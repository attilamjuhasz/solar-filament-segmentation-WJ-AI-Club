"""PyTorch dataset: preprocessed, downscaled images with soft filament targets.

Preprocessing a 2048 px image takes a moment, so inputs and targets are cached
as ``.npy`` files after the first epoch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import AnnotatedImage, read_image
from .preprocess import normalize


def model_input(img: np.ndarray, size: int) -> np.ndarray:
    """uint8 full-res image -> float32 (size x size) network input."""
    contrast, _ = normalize(img)
    return cv2.resize(contrast, (size, size), interpolation=cv2.INTER_AREA)


def soft_target(item: AnnotatedImage, size: int) -> np.ndarray:
    """Fraction of each low-res pixel covered by filament. Keeps thin barbs that a
    hard threshold would erase when downscaling."""
    mask = item.union_mask().astype(np.float32)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)


class FilamentDataset(Dataset):
    def __init__(
        self,
        items: list[AnnotatedImage],
        image_paths: dict[str, Path],
        size: int = 1024,
        crop: int | None = 512,
        train: bool = True,
        cache_dir: str | Path | None = "cache",
    ):
        self.items = items
        self.image_paths = image_paths
        self.size = size
        self.crop = crop if train else None
        self.train = train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.items)

    def _cached(self, key: str, compute) -> np.ndarray:
        if self.cache_dir is None:
            return compute()
        path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"
        if path.exists():
            return np.load(path).astype(np.float32)
        arr = compute()
        np.save(path, arr.astype(np.float16))
        return arr

    def load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        item = self.items[idx]
        x = self._cached(
            f"img|{item.stem}|{self.size}",
            lambda: model_input(read_image(self.image_paths[item.stem]), self.size),
        )
        y = self._cached(f"tgt|{item.image_id}|{self.size}", lambda: soft_target(item, self.size))
        return x, y

    def _random_crop(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        c = self.crop
        h, w = x.shape
        if rng.random() < 0.7 and (y > 0.5).any():
            # Centre most crops on a filament so batches are not mostly empty disk.
            ys, xs = np.nonzero(y > 0.5)
            i = rng.integers(len(ys))
            top = np.clip(ys[i] - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, h - c)
            left = np.clip(xs[i] - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, w - c)
        else:
            top, left = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
        return x[top : top + c, left : left + c], y[top : top + c, left : left + c]

    def _augment(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        k = int(rng.integers(4))
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        # Contrast/brightness jitter and noise mimic seeing and instrument differences.
        x = x * rng.uniform(0.8, 1.25) + rng.uniform(-0.3, 0.3)
        x = x + rng.normal(0, rng.uniform(0, 0.15), x.shape).astype(np.float32)
        return x, y

    def __getitem__(self, idx: int):
        x, y = self.load(idx)
        if self.train:
            rng = np.random.default_rng()
            if self.crop:
                x, y = self._random_crop(x, y, rng)
            x, y = self._augment(x, y, rng)
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None]
        return x, y
