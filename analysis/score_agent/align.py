"""Objective-alignment experiments on annotations only (no model needed).

Exact official-PQ counting on label maps via a joint histogram (valid for disjoint predictions).
  A  annotator-as-predictor vs OTHER readings, with per-instance erosion/dilation k
  B  consensus-of-two vs held-out third reading (3-reading stems)
  C  oracle emit gating (q vs q*s) + mean q*s by predicted area
  D  existence agreement by size
python align.py A|B|C|D
"""
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import load_inst, load_meta  # noqa: E402

BINS = [0, 200, 400, 1000, 3000, 8000, 10**9]


def pair_iou(P, G):
    """P, G int label maps (0 = bg). -> iou (nP, nG) over labels 1..; areas."""
    nP, nG = int(P.max()) + 1, int(G.max()) + 1
    j = np.bincount(P.ravel().astype(np.int64) * nG + G.ravel(), minlength=nP * nG).reshape(nP, nG)
    aP, aG = j.sum(1), j.sum(0)
    inter = j[1:, 1:].astype(np.float64)
    union = aP[1:, None] + aG[None, 1:] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    keepP = aP[1:] > 0
    keepG = aG[1:] > 0
    return iou[keepP][:, keepG], aP[1:][keepP], aG[1:][keepG]


def counts(iou):
    nP, nG = iou.shape
    if nG == 0:
        return np.array([0.0, 0, nP, 0])
    if nP == 0:
        return np.array([0.0, 0, 0, nG])
    hit = iou > 0.5
    return np.array([iou[hit].sum(), hit.sum(), (hit.sum(1) == 0).sum(), (hit.sum(0) == 0).sum()], float)


def pq(t):
    S, TP, FP, FN = t
    d = TP + 0.5 * FP + 0.5 * FN
    return S / d, dict(sq=round(S / max(TP, 1), 4), tp=int(TP), fp=int(FP), fn=int(FN))


def relabel(L):
    """Compact uint16 labels, drop empty."""
    u = np.unique(L)
    lut = np.zeros(int(u.max()) + 1, np.int32)
    lut[u[u > 0]] = np.arange(1, (u > 0).sum() + 1)
    return lut[L]


def morph(L, k):
    if k == 0:
        return L
    L16 = L.astype(np.uint16)
    ker = np.ones((2 * abs(k) + 1,) * 2, np.uint8)
    if k < 0:
        keep = (cv2.erode(L16, ker) == L16) & (cv2.dilate(L16, ker) == L16)
        return np.where(keep, L16, 0).astype(np.int32)
    d = cv2.dilate(L16, ker)
    return np.where(L16 > 0, L16, d).astype(np.int32)


def stems_readings(min_r=1):
    meta = load_meta()
    g = meta.groupby(meta.file_name.str[:-5]).image_id.apply(sorted)
    stride = int(os.environ.get("STRIDE", "1"))
    return {s: r for i, (s, r) in enumerate(g.items()) if len(r) >= min_r and i % stride == 0}


# ------------------------------------------------------------------------------------------ A
def exp_A():
    ks = [-2, -1, 0, 1, 2, 3]
    tot = {k: np.zeros(4) for k in ks}
    for s, rds in stems_readings(2).items():
        labs = [load_inst(r).astype(np.int32) for r in rds]
        for a in range(len(rds)):
            for k in ks:
                P = relabel(morph(labs[a], k)) if k else labs[a]
                for b in range(len(rds)):
                    if a != b:
                        tot[k] += counts(pair_iou(P, labs[b])[0])
    for k in ks:
        p, info = pq(tot[k])
        print(f"A  annotator-as-pred vs other readings, width shift {k:+d}px: PQ={p:.4f} {info}")


# ------------------------------------------------------------------------------------------ B
def exp_B():
    var = ["one_other", "union_cc", "inter_cc", "union_cc_inter_dil1"]
    tot = {v: np.zeros(4) for v in var}
    for s, rds in stems_readings(3).items():
        labs = [load_inst(r).astype(np.int32) for r in rds]
        for h in range(len(rds)):
            o = [labs[i] for i in range(len(rds)) if i != h]
            a, b = o[0] > 0, o[1] > 0
            preds = {
                "one_other": o[0],
                "union_cc": cv2.connectedComponents((a | b).astype(np.uint8), connectivity=8)[1],
                "inter_cc": cv2.connectedComponents((a & b).astype(np.uint8), connectivity=8)[1],
            }
            # intersection, dilated by 1 but only inside the union (a 'median' shape for 2 readers)
            inter_d = (cv2.dilate((a & b).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (a | b)
            preds["union_cc_inter_dil1"] = cv2.connectedComponents(inter_d.astype(np.uint8), connectivity=8)[1]
            for v, P in preds.items():
                tot[v] += counts(pair_iou(P, labs[h])[0])
    for v in var:
        p, info = pq(tot[v])
        print(f"B  consensus-of-2 vs held-out 3rd [{v:20s}] PQ={p:.4f} {info}")


# ------------------------------------------------------------------------------------------ C
def exp_C():
    """Predictions = each reading's instances (and union-CC), scored vs the OTHER readings.
    Per prediction: n readings, m matched, c = sum IoU over matches. Oracle gating curves."""
    recs = {"reading_vs_others": [], "unionCC_vs_all": []}
    base = {k: np.zeros(4) for k in recs}
    for s, rds in stems_readings(2).items():
        labs = [load_inst(r).astype(np.int32) for r in rds]
        for a in range(len(rds)):
            others = [labs[b] for b in range(len(rds)) if b != a]
            _collect(labs[a], others, recs["reading_vs_others"], base["reading_vs_others"])
        U = cv2.connectedComponents(np.any([l > 0 for l in labs], 0).astype(np.uint8), connectivity=8)[1]
        _collect(U, labs, recs["unionCC_vs_all"], base["unionCC_vs_all"])
    for name, R in recs.items():
        R = np.array(R)  # n, m, c, area
        S, TP, FP, FN = base[name]
        p0 = S / (TP + 0.5 * FP + 0.5 * FN)
        print(f"C  [{name}] all emitted: PQ={p0:.4f} preds={len(R)}")
        q = R[:, 1] / R[:, 0]
        qs = R[:, 2] / R[:, 0]
        for lab, key in (("q", q), ("q*s", qs)):
            best = (p0, None)
            for t in np.unique(np.round(key, 3)):
                drop = key < t
                d = drop
                S2 = S - R[d, 2].sum()
                TP2 = TP - R[d, 1].sum()
                FP2 = FP - (R[d, 0] - R[d, 1]).sum()
                FN2 = FN + R[d, 1].sum()
                p = S2 / (TP2 + 0.5 * FP2 + 0.5 * FN2)
                if p > best[0]:
                    best = (p, t)
            print(f"     oracle gate on {lab:4s}: best PQ={best[0]:.4f} at threshold {best[1]}  (rule: q*s > PQ/2 = {best[0] / 2:.3f})")
        b = np.digitize(R[:, 3], BINS) - 1
        row = []
        for i in range(len(BINS) - 1):
            sel = b == i
            if sel.any():
                row.append(f"<{BINS[i + 1]}:{qs[sel].mean():.3f}(q={q[sel].mean():.2f},n={sel.sum()})")
        print("     mean q*s by pred area:", " ".join(row))


def _collect(P, gts, recs, base):
    nP = int(P.max())
    if nP == 0:
        for G in gts:
            base += counts(np.zeros((0, int(G.max()))))
        return
    m = np.zeros(nP + 1)
    c = np.zeros(nP + 1)
    area = np.bincount(P.ravel(), minlength=nP + 1)
    for G in gts:
        nG = int(G.max()) + 1
        j = np.bincount(P.ravel().astype(np.int64) * nG + G.ravel(), minlength=(nP + 1) * nG).reshape(nP + 1, nG)
        aP, aG = j.sum(1), j.sum(0)
        inter = j[1:, 1:].astype(float)
        iou = inter / np.maximum(aP[1:, None] + aG[None, 1:] - inter, 1)
        iou[:, aG[1:] == 0] = 0
        base += counts(iou[aP[1:] > 0][:, aG[1:] > 0])
        hit = iou > 0.5
        m[1:] += hit.any(1)
        c[1:] += (iou * hit).sum(1)
    for i in range(1, nP + 1):
        if area[i]:
            recs.append((len(gts), m[i], c[i], area[i]))


# ------------------------------------------------------------------------------------------ D
def exp_D():
    """Each GT instance of reading A: does another reading have a match (IoU>.5)? any overlap (IoU>.1)?"""
    rows = []
    for s, rds in stems_readings(2).items():
        labs = [load_inst(r).astype(np.int32) for r in rds]
        for a in range(len(rds)):
            for b in range(len(rds)):
                if a == b:
                    continue
                iou, aA, _ = pair_iou(labs[a], labs[b])
                if iou.shape[1] == 0:
                    best = np.zeros(iou.shape[0])
                else:
                    best = iou.max(1)
                for ar, bi in zip(aA, best):
                    rows.append((ar, bi))
    R = np.array(rows)
    b = np.digitize(R[:, 0], BINS) - 1
    print("D  GT instance of reader A vs reader B: share of instances | P(match IoU>.5) | P(any IoU>.1) | mean IoU if matched")
    for i in range(len(BINS) - 1):
        sel = b == i
        mt = R[sel, 1] > 0.5
        print(f"   area<{BINS[i + 1]:>10}: {sel.mean():.3f} | {mt.mean():.3f} | {(R[sel, 1] > 0.1).mean():.3f} | "
              f"{R[sel, 1][mt].mean() if mt.any() else 0:.3f}")
    mt = R[:, 1] > 0.5
    print(f"   all: P(match)={mt.mean():.3f} meanIoU|match={R[mt, 1].mean():.3f}")


if __name__ == "__main__":
    {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D}[sys.argv[1]]()
