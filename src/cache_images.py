"""Cache grayscale images as .npy (native 2048 and 1024 downsample) for fast random access.

python src/cache_images.py              # img2048 + img1024
python src/cache_images.py --res 1536   # extra downsample from the 2048 cache (stage-1 at 1536)
"""
import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, TEST_IMG, TRAIN_IMG, read_gray  # noqa: E402


def extra_res(res):
    os.makedirs(os.path.join(CACHE, f"img{res}"), exist_ok=True)
    src = os.path.join(CACHE, "img2048")
    for f in tqdm(sorted(f for f in os.listdir(src) if f.endswith(".npy")), desc=f"img{res}"):
        out = os.path.join(CACHE, f"img{res}", f)
        if not os.path.exists(out):
            np.save(out, cv2.resize(np.load(os.path.join(src, f)), (res, res), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=None)
    a = ap.parse_args()
    if a.res:
        return extra_res(a.res)
    for split, src in [("train", TRAIN_IMG), ("test", TEST_IMG)]:
        for res in (2048, 1024):
            os.makedirs(os.path.join(CACHE, f"img{res}"), exist_ok=True)
        for f in tqdm(sorted(os.listdir(src)), desc=split):
            stem = os.path.splitext(f)[0]
            out = os.path.join(CACHE, "img2048", stem + ".npy")
            if os.path.exists(out):
                continue
            img = read_gray(os.path.join(src, f))
            assert img.shape == (2048, 2048), (f, img.shape)
            np.save(out, img)
            np.save(os.path.join(CACHE, "img1024", stem + ".npy"),
                    cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
