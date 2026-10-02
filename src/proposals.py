"""Blob proposals from stage-1 probability maps (the 'split everything into blobs' step).

Levels (all at native 2048, inside the disk):
  A  hysteresis on the union head u (seed >= .5, extent >= .3)        raw blobs
  P  hysteresis on the precise head p (seed >= .5, extent >= .35)     may split differently from A
  B  A-blobs rejoined across gaps <= 2*GAP px (only when >= 2 merged)  fixes fragmentation
  C  faint blobs: u >= .15 with peak >= .3, not already covered by A   recall for small/faint filaments
Near-duplicates (mask IoU > .9) are dropped; at most MAX_PROPS per image, ranked by peak u.

python src/proposals.py --run s1_r34_f0 --probs probs_plain     -> runs/<run>/props_<probs>/<stem>.pkl
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from pycocotools import mask as mu
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import RUNS, disk_info, disk_mask  # noqa: E402

MIN_AREA = 30
GAP = 12
MAX_PROPS = 80


def _up(p):
    return cv2.resize(p.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)


def _hyst(p, hi, lo):
    n, lab = cv2.connectedComponents((p > lo).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[p > hi])] = True
    keep[0] = False
    return keep[lab]


def _components(binary):
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        x, y, w, h = st[i, :4]
        out.append((lab[y:y + h, x:x + w] == i, (x, y, w, h)))
    return out


def _full(local, box):
    x, y, w, h = box
    m = np.zeros((2048, 2048), np.uint8)
    m[y:y + h, x:x + w] = local
    return m


def make_proposals(probs, stem):
    """probs: float array (2, 1024, 1024) in [0, 1]. Returns list of dicts with native RLE masks."""
    p, u = _up(probs[0]), _up(probs[1])
    dm = disk_mask(disk_info(stem))
    p[~dm] = 0
    u[~dm] = 0
    props = []

    A = _hyst(u, 0.5, 0.3)
    for loc, box in _components(A):
        props.append(("A", loc, box))
    for loc, box in _components(_hyst(p, 0.5, 0.35)):
        props.append(("P", loc, box))
    # B: rejoin A blobs across small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GAP + 1, 2 * GAP + 1))
    _, glab = cv2.connectedComponents(cv2.dilate(A.astype(np.uint8), k), connectivity=8)
    _, alab = cv2.connectedComponents(A.astype(np.uint8), connectivity=8)
    gl = np.where(A, glab, 0)
    for g in np.unique(gl[gl > 0]):
        sel = gl == g
        if len(np.unique(alab[sel])) >= 2:
            ys, xs = np.nonzero(sel)
            x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            if sel.sum() >= MIN_AREA:
                props.append(("B", sel[y:y + h, x:x + w], (x, y, w, h)))
    # C: faint blobs not covered by A
    for loc, box in _components(_hyst(u, 0.3, 0.15)):
        x, y, w, h = box
        cov = (A[y:y + h, x:x + w] & loc).sum() / loc.sum()
        if cov < 0.5:
            props.append(("C", loc & ~A[y:y + h, x:x + w] if cov > 0 else loc, box))

    out = []
    for level, loc, (x, y, w, h) in props:
        if loc.sum() < MIN_AREA:
            continue
        sl = (slice(y, y + h), slice(x, x + w))
        out.append(dict(level=level, box=(int(x), int(y), int(w), int(h)), area=int(loc.sum()),
                        peak_u=float(u[sl][loc].max()), mean_u=float(u[sl][loc].mean()),
                        peak_p=float(p[sl][loc].max()), mean_p=float(p[sl][loc].mean()),
                        rle=mu.encode(np.asfortranarray(_full(loc.astype(np.uint8), (x, y, w, h))))))
    out.sort(key=lambda d: -d["peak_u"])
    # dedupe near-identical masks (A vs P often coincide)
    keep = []
    for d in out:
        if keep:
            iou = np.asarray(mu.iou([d["rle"]], [k["rle"] for k in keep], [0] * len(keep)))[0]
            if iou.max() > 0.9:
                continue
        keep.append(d)
        if len(keep) >= MAX_PROPS:
            break
    return keep


def load_probs(path):
    return np.load(path).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probs", default="probs_plain")
    a = ap.parse_args()
    pdir = os.path.join(RUNS, a.run, a.probs)
    odir = os.path.join(RUNS, a.run, "props_" + a.probs.replace("probs_", ""))
    os.makedirs(odir, exist_ok=True)
    for f in tqdm(sorted(f for f in os.listdir(pdir) if f.endswith(".npy"))):
        stem = f[:-4]
        props = make_proposals(load_probs(os.path.join(pdir, f)), stem)
        pickle.dump(props, open(os.path.join(odir, stem + ".pkl"), "wb"))


if __name__ == "__main__":
    main()
