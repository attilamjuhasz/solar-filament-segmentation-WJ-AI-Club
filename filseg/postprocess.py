"""Turn a filament probability map into individual filament instances.

PQ counts every extra fragment as a false positive and every merge of two
filaments as at least one miss, so this step matters as much as the network:

1. Hysteresis threshold: keep weak pixels only if connected to a confident one.
2. Drop small specks.
3. Group fragments closer than ``merge_dist`` px into one filament.
4. Morphologically close each filament so it is a single contiguous piece.
5. Drop filaments that are still too small.

All parameters are tuned on the validation set with ``python -m filseg.tune``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PostParams:
    thr_high: float = 0.5
    thr_low: float = 0.3
    min_area: int = 50  # px at full resolution, applied to fragments before grouping
    merge_dist: int = 12  # px; fragments closer than this become one filament
    close_radius: int = 3  # px; closing applied to each filament
    min_instance_area: int = 300  # px; final filaments smaller than this are removed

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PostParams":
        return cls(**json.loads(Path(path).read_text()))


def _disk_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def hysteresis(prob: np.ndarray, low: float, high: float) -> np.ndarray:
    n, labels = cv2.connectedComponents((prob > low).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[prob > high])] = True
    keep[0] = False
    return keep[labels]


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def to_instances(prob: np.ndarray, params: PostParams, disk_mask: np.ndarray | None = None) -> np.ndarray:
    """Probability map (HxW, 0..1) -> instance label image (0 = background, 1..N)."""
    binary = hysteresis(prob, params.thr_low, max(params.thr_high, params.thr_low))
    if disk_mask is not None:
        binary &= disk_mask
    binary = remove_small(binary, params.min_area)

    grouping = binary.astype(np.uint8)
    if params.merge_dist > 0:
        # Dilating by d/2 joins fragments whose gap is <= merge_dist.
        grouping = cv2.dilate(grouping, _disk_kernel(max(1, params.merge_dist // 2)))
    n, groups, stats, _ = cv2.connectedComponentsWithStats(grouping, connectivity=8)

    labels = np.zeros(prob.shape, np.int32)
    pad = params.close_radius + 1
    h, w = prob.shape
    next_id = 1
    for k in range(1, n):
        x, y, bw, bh = stats[k, :4]
        y0, y1 = max(0, y - pad), min(h, y + bh + pad)
        x0, x1 = max(0, x - pad), min(w, x + bw + pad)
        inst = ((groups[y0:y1, x0:x1] == k) & binary[y0:y1, x0:x1]).astype(np.uint8)
        if params.close_radius > 0:
            inst = cv2.morphologyEx(inst, cv2.MORPH_CLOSE, _disk_kernel(params.close_radius))
        inst = inst.astype(bool) & (labels[y0:y1, x0:x1] == 0)
        if inst.sum() < params.min_instance_area:
            continue
        labels[y0:y1, x0:x1][inst] = next_id
        next_id += 1
    return labels
