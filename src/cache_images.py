"""Cache grayscale images as .npy (native 2048 and 1024 downsample) for fast random access."""
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, TEST_IMG, TRAIN_IMG, read_gray  # noqa: E402


def main():
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
