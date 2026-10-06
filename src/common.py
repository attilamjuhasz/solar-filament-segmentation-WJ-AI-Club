import os

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "MAGFiLO_1.0_Kaggle_2026")
TRAIN_IMG = os.path.join(DATA, "train", "train_images")
TEST_IMG = os.path.join(DATA, "test", "test_images")
CACHE = os.path.join(ROOT, "data", "cache")
RUNS = os.path.join(ROOT, "runs")
FULL = 2048


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_meta():
    return pd.read_csv(os.path.join(CACHE, "meta.csv"))


def load_inst(image_id):
    """Per-reading instance map (0 = background). The uint8 .npy cache loads ~25x faster than the PNG."""
    npy = os.path.join(CACHE, "inst_u8", image_id + ".npy")
    if os.path.exists(npy):
        return np.load(npy)
    return cv2.imread(os.path.join(CACHE, "inst", image_id + ".png"), cv2.IMREAD_UNCHANGED)


_DISK = None


def disk_info(stem):
    global _DISK
    if _DISK is None:
        import json
        _DISK = json.load(open(os.path.join(CACHE, "disk.json")))
    return _DISK[stem]


def radius_map(info, shape, x0=0.0, y0=0.0, step=1.0):
    """r/R for a grid whose pixel (i, j) center sits at native (x0 + (j+.5)*step - .5, y0 + ...)."""
    h, w = shape
    xs = x0 + (np.arange(w, dtype=np.float32) + 0.5) * step - 0.5
    ys = y0 + (np.arange(h, dtype=np.float32) + 0.5) * step - 0.5
    return np.hypot(xs[None, :] - info["cx"], ys[:, None] - info["cy"]) / info["r"]


def make_planes(img, info, x0=0.0, y0=0.0, step=1.0):
    """uint8 gray region -> float32 (3, h, w): normalized intensity, limb-flattened contrast, r/R.

    The same function serves full-disk 1024 inputs (step=2) and native or resampled crops.
    """
    x = img.astype(np.float32)
    rr = radius_map(info, x.shape, x0, y0, step)
    prof = np.asarray(info["prof"], np.float32)
    ref = np.interp(np.minimum(rr, 0.999) * len(prof), np.arange(len(prof)) + 0.5, prof)
    p0 = np.clip((x - info["med"]) / (2 * info["iqr"]), -3, 3)
    p1 = np.clip((x / np.maximum(ref, 1.0) - 1) * 4, -3, 3)
    p2 = np.minimum(rr, 1.5)
    return np.stack([p0, p1, p2]).astype(np.float32)


def disk_mask(info, shape=(FULL, FULL), x0=0.0, y0=0.0, step=1.0, margin=0.0):
    return radius_map(info, shape, x0, y0, step) <= 1.0 + margin
