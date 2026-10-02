"""Resolution ceilings of a PERFECT model (outputs exactly its training target) + q-label fidelity.

S1: reading fg -> INTER_AREA 1024 -> (perfect net) -> linear 2048 -> >.5 -> CC  vs the same reading.
S2: each GT instance used as its own prior -> s2.window -> crop -> resize_to(S) -> (perfect net)
    -> resize_to(side, linear) -> >.5  vs the instance. Also the s2_loss q label for that perfect pred.
python res.py [S] [MAX_SIDE] [every]
"""
import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(1)
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2  # noqa: E402
from align import BINS, counts, pair_iou, pq  # noqa: E402
from common import load_inst, load_meta  # noqa: E402

S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
MAXS = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 1
s2.S = S
s2.MAX_SIDE = MAXS


def s2_roundtrip(g, box):
    x0, y0, side = s2.window(box)
    y = s2.resize_to(s2.crop_pad(g, x0, y0, side).astype(np.float32), S)
    big = s2.resize_to(y, side, area=False)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, 2048), min(y0 + side, 2048)
    m = np.zeros((2048, 2048), bool)
    m[ya:yb, xa:xb] = big[ya - y0:yb - y0, xa - x0:xb - x0] > 0.5
    step = side / S
    A = (y > 0.5).sum() * step ** 2  # s2_loss: it = pa = |hard & tgt| * step^2 for the perfect pred
    G = g.sum()
    q_iou = A / max(A + G - A, 1)
    return m, step, q_iou


def main():
    meta = load_meta()
    rids = sorted(meta.image_id)[::EVERY]
    t1 = np.zeros(4)
    t2 = np.zeros(4)
    rows = []
    for rid in rids:
        L = load_inst(rid).astype(np.int32)
        # ---- S1 perfect-net ceiling
        fg = (L > 0).astype(np.float32)
        d = cv2.resize(fg, (1024, 1024), interpolation=cv2.INTER_AREA)
        d = np.round(d * 255) / 255  # uint8 target quantisation as in prep
        up = cv2.resize(d, (2048, 2048), interpolation=cv2.INTER_LINEAR) > 0.5
        P = cv2.connectedComponents(up.astype(np.uint8), connectivity=8)[1]
        t1 += counts(pair_iou(P, L)[0])
        # ---- S2 perfect-net ceiling, each instance its own prior
        n_inst = 0
        S_, TP = 0.0, 0
        for lab in np.unique(L)[1:]:
            g = L == lab
            ys, xs = np.nonzero(g)
            box = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            m, step, q_iou = s2_roundtrip(g, box)
            inter = (m & g).sum()
            iou = inter / max((m | g).sum(), 1)
            rows.append((g.sum(), step, iou, q_iou, max(box[2], box[3])))
            n_inst += 1
            if iou > 0.5:
                S_ += iou
                TP += 1
        t2 += np.array([S_, TP, n_inst - TP, n_inst - TP])
    R = np.array(rows)
    p1, i1 = pq(t1)
    p2, i2 = pq(t2)
    print(f"readings={len(rids)} instances={len(R)}  S={S} MAX_SIDE={MAXS}")
    print(f"S1 perfect-net @1024 self-PQ = {p1:.4f} {i1}")
    print(f"S2 perfect-net @window->{S} self-PQ = {p2:.4f} {i2}")
    b = np.digitize(R[:, 0], BINS) - 1
    print("  area bin | share | median step | S2 mean IoU | P(IoU<=.5) | q-label!=native(>.5) | q-label IoU median")
    for i in range(len(BINS) - 1):
        sel = b == i
        if not sel.any():
            continue
        r = R[sel]
        dis = ((r[:, 3] > 0.5) != (r[:, 2] > 0.5)).mean()
        print(f"  <{BINS[i + 1]:>10} | {sel.mean():.3f} | {np.median(r[:, 1]):.2f} | {r[:, 2].mean():.3f} | "
              f"{(r[:, 2] <= 0.5).mean():.3f} | {dis:.3f} | {np.median(r[:, 3]):.3f}")
    sb = np.digitize(R[:, 1], [0, 1.01, 2.01, 3.01, 4.01, 100]) - 1
    print("  step bin | share | S2 mean IoU | P(IoU<=.5)")
    for i, nm in enumerate(["<=1", "1-2", "2-3", "3-4", ">4"]):
        sel = sb == i
        if sel.any():
            print(f"  {nm:>5} | {sel.mean():.3f} | {R[sel, 2].mean():.3f} | {(R[sel, 2] <= .5).mean():.3f}")


if __name__ == "__main__":
    main()
