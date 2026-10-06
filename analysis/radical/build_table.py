"""Per-candidate x per-reading table on fold-0 val (v2 candidates: runs/s2_r34/cands_val_tta_last).

For every candidate (all levels) and mask threshold t in THRS, record per reading: IoU with best GT, label, y=IoU*1(IoU>.5).
Also pairwise pixel overlaps between candidates (thr .5) for set-selection decoding.
Output: table.pkl
"""
import os, sys, pickle, time
import numpy as np
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "analysis/score_agent"))
import cv2
cv2.setNumThreads(1)
from ana import halves
from assemble import load_cands, FastGT

THRS = [0.3, 0.4, 0.5, 0.6, 0.7]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table.pkl")
A, B = halves()
stems = sorted(A + B)
C = load_cands(os.path.join(ROOT, "runs/s2_r34/cands_val_tta_last"), stems)
t0 = time.time()
rows = []   # one dict per candidate
G = FastGT(stems)
gtinfo = {}
for si, s in enumerate(stems):
    reads = G.by_stem[s]
    gtinfo[s] = [areas.copy() for lab, areas in reads]
    cs = C[s]
    for k, c in enumerate(cs):
        r = dict(stem=s, k=k, level=c["level"], q=c["q"], mean_p=c["mean_p"], peak_u=c["peak_u"], x=c["x"], y=c["y"],
                 shape=c["soft"].shape, half="A" if s in A else "B")
        for t in THRS:
            m = c["soft"] > int(t * 255)
            area = int(m.sum())
            ious, labs, inters = [], [], []
            for lab, areas in reads:
                h, w = m.shape
                inter = np.bincount(lab[c["y"]:c["y"] + h, c["x"]:c["x"] + w][m], minlength=256)
                inter[0] = 0
                iou = inter / np.maximum(area + areas - inter, 1)
                j = int(iou.argmax())
                ious.append(float(iou[j])); labs.append(j); inters.append(int(inter[j]))
            r[f"area_{t}"] = area
            r[f"iou_{t}"] = np.array(ious)
            r[f"lab_{t}"] = np.array(labs)
            if t == 0.5:
                r["mprob"] = float(c["soft"][m].mean()) / 255 if m.any() else 0.0
        rows.append(r)
    # pairwise overlaps at thr .5
    ms = [(c["x"], c["y"], c["soft"] > 127) for c in cs]
    ov = {}
    for i in range(len(cs)):
        xi, yi, mi = ms[i]
        for j in range(i + 1, len(cs)):
            xj, yj, mj = ms[j]
            x0, y0 = max(xi, xj), max(yi, yj)
            x1, y1 = min(xi + mi.shape[1], xj + mj.shape[1]), min(yi + mi.shape[0], yj + mj.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            a = mi[y0 - yi:y1 - yi, x0 - xi:x1 - xi] & mj[y0 - yj:y1 - yj, x0 - xj:x1 - xj]
            n = int(a.sum())
            if n:
                ov[(i, j)] = n
    for r in rows[-len(cs):]:
        pass
    gtinfo[s + "#ov"] = ov
    if si % 20 == 0:
        print(si, s, len(cs), f"{time.time() - t0:.0f}s", flush=True)
pickle.dump(dict(rows=rows, gt=gtinfo, A=A, B=B, stems=stems, thrs=THRS), open(OUT, "wb"))
print("done", len(rows), f"{time.time() - t0:.0f}s")
