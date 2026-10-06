"""Average stage-1 probability maps of several runs (seed/fold/resolution ensemble).

python src/ens_probs.py --runs s1_a,s1_b --probs probs_tta --out s1_ens   -> runs/s1_ens/probs_tta/<stem>.npy
Maps of different resolutions are resized (bilinear) to the largest one before averaging.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import RUNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--probs", default="probs_tta")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = a.runs.split(",")
    dirs = [os.path.join(RUNS, r, a.probs) for r in runs]
    stems = sorted(set.intersection(*[{f[:-4] for f in os.listdir(d) if f.endswith(".npy")} for d in dirs]))
    odir = os.path.join(RUNS, a.out, a.probs)
    os.makedirs(odir, exist_ok=True)
    for s in tqdm(stems):
        maps = [np.load(os.path.join(d, s + ".npy")).astype(np.float32) for d in dirs]
        r = max(m.shape[-1] for m in maps)
        maps = [m if m.shape[-1] == r else np.stack([cv2.resize(c, (r, r), interpolation=cv2.INTER_LINEAR) for c in m])
                for m in maps]
        np.save(os.path.join(odir, s + ".npy"), np.round(np.mean(maps, 0)).astype(np.uint8))
    print(f"averaged {len(runs)} runs over {len(stems)} stems -> {odir}")


if __name__ == "__main__":
    main()
