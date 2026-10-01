from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn
from src.models.resnet_unet import ResNet34UNet


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate a segmentation model from configuration."""
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "resnet34_unet")
    in_channels = model_cfg.get("in_channels", 3)
    pretrained = model_cfg.get("pretrained", False)

    if "resnet34" in name:
        return ResNet34UNet(in_channels=in_channels, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model architecture: {name}")


__all__ = ["ResNet34UNet", "build_model"]
