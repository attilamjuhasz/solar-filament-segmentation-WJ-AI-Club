"""Full-resolution inference: image -> probability map -> filament instances."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from .postprocess import PostParams, to_instances
from .preprocess import normalize


@torch.no_grad()
def predict_lowres(model: nn.Module, x: np.ndarray, device, tta: bool = True) -> np.ndarray:
    """Network input (size x size float32) -> probability map at the same size."""
    t = torch.from_numpy(x)[None, None].to(device)
    flips = [(), (3,), (2,), (2, 3)] if tta else [()]
    prob = torch.zeros_like(t)
    for dims in flips:
        inp = t.flip(dims) if dims else t
        out = torch.sigmoid(model(inp))
        prob += out.flip(dims) if dims else out
    return (prob / len(flips))[0, 0].float().cpu().numpy()


def upsample(prob: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def predict_prob(model: nn.Module, img: np.ndarray, size: int, device, tta: bool = True):
    """uint8 full-res image -> (full-res probability map, solar disk mask)."""
    contrast, disk = normalize(img)
    x = cv2.resize(contrast, (size, size), interpolation=cv2.INTER_AREA)
    prob = upsample(predict_lowres(model, x, device, tta), img.shape)
    return prob, disk.mask(img.shape)


def predict_instances(model, img, size, device, params: PostParams, tta: bool = True) -> np.ndarray:
    prob, disk_mask = predict_prob(model, img, size, device, tta)
    return to_instances(prob, params, disk_mask)
