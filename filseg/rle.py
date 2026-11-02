"""COCO RLE helpers. The submission stores only the RLE ``counts`` string;
the size is always the image size (2048 x 2048)."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from .data import IMG_SIZE


def encode(mask: np.ndarray) -> str:
    """Binary HxW mask -> RLE counts string."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def to_rle(counts: str, height: int = IMG_SIZE, width: int = IMG_SIZE) -> dict:
    return {"size": [height, width], "counts": counts.encode("ascii")}


def decode(counts: str, height: int = IMG_SIZE, width: int = IMG_SIZE) -> np.ndarray:
    return mask_utils.decode(to_rle(counts, height, width)).astype(bool)


def labels_to_rles(labels: np.ndarray) -> list[dict]:
    """Instance label image (0 = background, 1..N = filaments) -> list of COCO RLEs."""
    rles = []
    for k in range(1, int(labels.max()) + 1):
        mask = labels == k
        if mask.any():
            rles.append(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
    return rles
