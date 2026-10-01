from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsampling followed by concatenation with skip connection and convolution."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle slight odd-size padding mismatch if any
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    """ResNet-34 U-Net multi-task architecture for solar filament instance segmentation.
    
    Heads:
    1. fg_logits: [B, 1, H, W] - Filament semantic segmentation logits
    2. bnd_logits: [B, 1, H, W] - Instance boundary separation logits
    3. ctr_logits: [B, 1, H, W] - Gaussian instance center proposal logits
    4. off_pred: [B, 2, H, W] - Continuous normalized 2D offset vector field
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        base_decoder_channels: int = 64,
    ):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Encoder stages
        if in_channels != 3:
            self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.encoder_conv1 = resnet.conv1

        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels, 1/4 (after maxpool)
        self.layer2 = resnet.layer2  # 128 channels, 1/8
        self.layer3 = resnet.layer3  # 256 channels, 1/16
        self.layer4 = resnet.layer4  # 512 channels, 1/32

        # Decoder stages (upsampling back to 1/1)
        # layer4 (512) -> dec4 with layer3 (256) -> 256
        self.dec4 = DecoderBlock(512, 256, 256)
        # dec4 (256) -> dec3 with layer2 (128) -> 128
        self.dec3 = DecoderBlock(256, 128, 128)
        # dec3 (128) -> dec2 with layer1 (64) -> 64
        self.dec2 = DecoderBlock(128, 64, 64)
        # dec2 (64) -> dec1 with stem (64) -> 32
        self.dec1 = DecoderBlock(64, 64, 32)
        # Final upsample to native 1/1 resolution
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvBlock(32, base_decoder_channels),
        )

        # Multi-task output heads
        self.head_fg = nn.Conv2d(base_decoder_channels, 1, kernel_size=1)
        self.head_bnd = nn.Conv2d(base_decoder_channels, 1, kernel_size=1)
        self.head_ctr = nn.Conv2d(base_decoder_channels, 1, kernel_size=1)
        self.head_off = nn.Conv2d(base_decoder_channels, 2, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]

        # Encoder
        x0 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x)))  # 64 ch, H/2, W/2
        x1 = self.encoder_maxpool(x0)                                    # 64 ch, H/4, W/4
        x1 = self.layer1(x1)                                             # 64 ch, H/4, W/4
        x2 = self.layer2(x1)                                             # 128 ch, H/8, W/8
        x3 = self.layer3(x2)                                             # 256 ch, H/16, W/16
        x4 = self.layer4(x3)                                             # 512 ch, H/32, W/32

        # Decoder
        d4 = self.dec4(x4, x3)  # 256 ch, H/16
        d3 = self.dec3(d4, x2)  # 128 ch, H/8
        d2 = self.dec2(d3, x1)  # 64 ch, H/4
        d1 = self.dec1(d2, x0)  # 32 ch, H/2
        features = self.dec0(d1)  # base_decoder_channels, H, W

        # Ensure exact match with input spatial dims
        if features.shape[-2:] != (H, W):
            features = F.interpolate(features, size=(H, W), mode="bilinear", align_corners=False)

        # Heads
        fg_logits = self.head_fg(features)
        bnd_logits = self.head_bnd(features)
        ctr_logits = self.head_ctr(features)
        off_pred = self.head_off(features)

        return {
            "fg_logits": fg_logits,
            "bnd_logits": bnd_logits,
            "ctr_logits": ctr_logits,
            "off_pred": off_pred,
        }
