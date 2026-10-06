"""Cache per-reading instance maps (data/cache/inst/*.png, uint16) as uint8 .npy for fast loading.

python src/cache_inst.py   -> data/cache/inst_u8/<reading_id>.npy   (labels are < 255 per reading)
"""
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE  # noqa: E402


def main():
    src, dst = os.path.join(CACHE, "inst"), os.path.join(CACHE, "inst_u8")
    os.makedirs(dst, exist_ok=True)
    for f in tqdm(sorted(f for f in os.listdir(src) if f.endswith(".png"))):
        out = os.path.join(dst, f[:-4] + ".npy")
        if os.path.exists(out):
            continue
        lab = cv2.imread(os.path.join(src, f), cv2.IMREAD_UNCHANGED)
        assert lab.max() < 255, f
        np.save(out[:-4] + ".tmp.npy", lab.astype(np.uint8))
        os.replace(out[:-4] + ".tmp.npy", out)


if __name__ == "__main__":
    main()
