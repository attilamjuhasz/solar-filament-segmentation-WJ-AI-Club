"""U-Net filament segmenter, loss, and checkpoint I/O."""
from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


def build_model(encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    """Single-channel input (the images are grayscale), single-channel logit output."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,
        classes=1,
    )


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict) -> None:
    """Weights are stored in fp16 to stay well under GitHub's 100 MB file limit."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["encoder"], pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), meta
