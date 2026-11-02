"""Solar-disk detection and limb-darkening correction for GONG H-alpha images.

The disk is brighter at the centre than at the limb, and brightness varies
between observatories and observing conditions. Dividing by the radial
intensity profile flattens the disk so filaments appear as a consistent
negative contrast, which makes thresholds and models transfer across images.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter

CONTRAST_SCALE = 0.1  # a filament ~20-30% darker than its surroundings maps to about -2..-3
CLIP = 4.0


@dataclass
class Disk:
    cx: float
    cy: float
    r: float

    def mask(self, shape: tuple[int, int], margin: float = 0.0) -> np.ndarray:
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        return (xx - self.cx) ** 2 + (yy - self.cy) ** 2 <= (self.r - margin) ** 2


def find_disk(img: np.ndarray) -> Disk:
    """Fit the solar disk as the enclosing circle of the largest bright blob."""
    h, w = img.shape
    scale = 512 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.1 * small.size:
            (cx, cy), r = cv2.minEnclosingCircle(largest)
            return Disk(cx / scale, cy / scale, r / scale)
    return Disk(w / 2, h / 2, 0.45 * min(h, w))  # fallback: centred disk


def radial_profile(img: np.ndarray, disk: Disk, bin_px: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity as a function of radius (smoothed), plus each pixel's radial bin."""
    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx - disk.cx) ** 2 + (yy - disk.cy) ** 2)
    bins = (r / bin_px).astype(np.int32)
    n_bins = int(disk.r / bin_px) + 2
    inside = bins < n_bins
    sums = np.bincount(bins[inside], weights=img[inside].astype(np.float64), minlength=n_bins)
    counts = np.bincount(bins[inside], minlength=n_bins)
    profile = sums / np.maximum(counts, 1)
    # Median filtering keeps the profile from being pulled down by large filaments or sunspots.
    profile = median_filter(profile, size=9, mode="nearest")
    return profile, np.minimum(bins, n_bins - 1)


def normalize(img: np.ndarray) -> tuple[np.ndarray, Disk]:
    """uint8 H-alpha image -> float32 contrast image (disk background ~0, filaments < 0).

    Pixels outside the disk are set to 0.
    """
    disk = find_disk(img)
    profile, bins = radial_profile(img, disk)
    background = np.maximum(profile[bins], 1.0)
    contrast = (img.astype(np.float32) / background - 1.0) / CONTRAST_SCALE
    contrast = np.clip(contrast, -CLIP, CLIP)
    contrast[~disk.mask(img.shape, margin=2)] = 0.0
    return contrast.astype(np.float32), disk
