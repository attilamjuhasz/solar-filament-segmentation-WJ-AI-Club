"""Overlays for eyeballing predictions next to the ground truth."""
from __future__ import annotations

import cv2
import numpy as np


def overlay(img: np.ndarray, labels: np.ndarray | None = None, gt_mask: np.ndarray | None = None) -> np.ndarray:
    """RGB image with each predicted filament in its own colour and GT outlined in white."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if labels is not None and labels.max() > 0:
        rng = np.random.default_rng(0)
        colours = rng.integers(60, 255, size=(int(labels.max()) + 1, 3)).astype(np.float32)
        fg = labels > 0
        rgb[fg] = 0.45 * rgb[fg] + 0.55 * colours[labels[fg]]
    out = rgb.clip(0, 255).astype(np.uint8)
    if gt_mask is not None:
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out
