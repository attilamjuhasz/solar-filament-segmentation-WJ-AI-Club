"""Noise floor of val-fold PQ: stem-level bootstrap (marginal and paired) with a PQ~0.4 proxy predictor.

Proxy = perturbed GT of a random reading (parity.sc_perturb). Paired change = +1 px dilation (like `grow`).
Also: the FastGT duplicate-GT discrepancy on the one reading with a near-duplicate polygon pair.
"""
import gc
import os
import sys

import cv2
import numpy as np

ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official as O  # noqa: E402
from assemble import FastGT, to_local  # noqa: E402
from common import load_meta  # noqa: E402
from metric import load_gt_rles  # noqa: E402
from parity import ANN, gt_dataframe, pred_dataframe, readings_of, sc_perturb  # noqa: E402
from pycocotools import mask as mu  # noqa: E402


def per_stem_stats(preds, stems, transform=None, chunk=25):
    out = {}
    stems = sorted(stems)
    for i in range(0, len(stems), chunk):
        ch = stems[i:i + chunk]
        g = FastGT(ch)
        for s in ch:
            fin = [to_local(r) for r in preds.get(s, [])]
            if transform:
                fin = transform(fin)
            out[s] = np.array(g.stats(s, fin), float)
        del g
        gc.collect()
    return out


def grow1(fin):
    owner = np.zeros((2048, 2048), bool)
    for x, y, m in fin:
        owner[y:y + m.shape[0], x:x + m.shape[1]] |= m
    res = []
    for x, y, m in fin:
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        x1, y1 = min(x + m.shape[1] + 1, 2048), min(y + m.shape[0] + 1, 2048)
        big = np.zeros((y1 - y0, x1 - x0), np.uint8)
        big[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] = m
        d = (cv2.dilate(big, np.ones((3, 3), np.uint8)) > 0) & ~owner[y0:y1, x0:x1]
        owner[y0:y1, x0:x1] |= d
        res.append((x0, y0, d | big.astype(bool)))
    return res


def pq_of(t):
    return t[0] / (t[1] + 0.5 * t[2] + 0.5 * t[3])


def main():
    meta = load_meta()
    stems = sorted(meta[meta.fold == 0].file_name.str[:-5].unique())
    gt = load_gt_rles(ANN, set(stems))
    preds = sc_perturb(readings_of(gt), np.random.default_rng(7))
    A = per_stem_stats(preds, stems)
    B = per_stem_stats(preds, stems, transform=grow1)
    a = np.array([A[s] for s in stems])
    b = np.array([B[s] for s in stems])
    print(f"proxy PQ={pq_of(a.sum(0)):.4f}  grow1 PQ={pq_of(b.sum(0)):.4f}  delta={pq_of(b.sum(0)) - pq_of(a.sum(0)):+.4f}")
    rng = np.random.default_rng(0)
    pa, pd_ = [], []
    for _ in range(4000):
        idx = rng.integers(0, len(stems), len(stems))
        x, y = pq_of(a[idx].sum(0)), pq_of(b[idx].sum(0))
        pa.append(x)
        pd_.append(y - x)
    print(f"bootstrap SE (141 stems): marginal PQ {np.std(pa):.4f}; paired delta(grow1) {np.std(pd_):.4f}")
    # split-half: how often does the sign of a delta flip between halves?
    half = []
    for _ in range(2000):
        perm = rng.permutation(len(stems))
        h1, h2 = perm[: len(stems) // 2], perm[len(stems) // 2:]
        half.append(pq_of(a[h1].sum(0)) - pq_of(a[h2].sum(0)))
    print(f"split-half |PQ(h1)-PQ(h2)| median {np.median(np.abs(half)):.4f}")

    # ---- FastGT vs official on the near-duplicate GT reading (fold 3)
    stem = "20160810234034Lh"
    gt3 = load_gt_rles(ANN, {stem})
    gdf = gt_dataframe({stem})
    rid = "010403-20160810234034Lh"
    rles = gt3[rid][1]
    iou = np.asarray(mu.iou(rles, rles, [0] * len(rles)))
    np.fill_diagonal(iou, 0)
    i, j = np.unravel_index(iou.argmax(), iou.shape)
    pick = rles[i] if mu.area(rles[i]) >= mu.area(rles[j]) else rles[j]
    p = {stem: [pick]}
    ov = O.get_overlap_df(gdf, pred_dataframe(p))
    print("dup-GT reading: official", O.get_pq_score_counts(ov))
    g = FastGT([stem])
    t = np.zeros(4)
    t += g.stats(stem, [to_local(pick)])
    print("dup-GT reading: FastGT  ", dict(S=t[0], tp=int(t[1]), fp=int(t[2]), fn=int(t[3])), f"(pair IoU {iou.max():.3f})")


if __name__ == "__main__":
    main()
