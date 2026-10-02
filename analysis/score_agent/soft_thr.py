"""Model-like soft consensus (Gaussian sigma=1 blur of the mean of 2 readers) -> threshold t -> CC,
optionally gated by a 'core' (component must contain soft >= core, i.e. both readers drew it).
Scored vs the held-out 3rd reader (3-reading stems, every 2nd stem). Mimics S2's thr / existence coupling."""
import os, sys
import cv2, numpy as np
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("STRIDE", "2")
from align import counts, pair_iou, pq, stems_readings
from common import load_inst

TS = [0.2, 0.3, 0.4, 0.5, 0.6]
var = [(t, g) for t in TS for g in (None, 0.75)]
tot = {v: np.zeros(4) for v in var}
for s, rds in stems_readings(3).items():
    labs = [load_inst(r).astype(np.int32) for r in rds]
    for h in range(3):
        o = [labs[i] > 0 for i in range(3) if i != h]
        soft = sum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0) for m in o) / 2
        for t, g in var:
            n, L = cv2.connectedComponents((soft > t).astype(np.uint8), connectivity=8)
            if g is not None and n > 1:
                mx = np.zeros(n, np.float32)
                np.maximum.at(mx, L.ravel(), soft.ravel())
                keep = mx >= g
                keep[0] = False
                L = np.where(keep[L], L, 0)
            tot[(t, g)] += counts(pair_iou(L, labs[h])[0])
for v in var:
    p, info = pq(tot[v])
    print(f"thr={v[0]:.1f} gate={'core>=.75' if v[1] else 'none     '}  PQ={p:.4f} {info}", flush=True)
