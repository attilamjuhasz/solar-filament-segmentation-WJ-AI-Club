"""Human-consensus ceiling: predict a held-out reading from the OTHER readings of the same image (GT only).
P1 = one other annotator's drawing as-is; P2 (3-reading images) = consensus decoding of 2 other annotators.
Run over all multi-reading stems (train+val) and the fold-0 val subset separately."""
import os, sys
import numpy as np, pandas as pd, cv2
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import load_meta, load_inst

m = load_meta(); m["stem"] = m.file_name.str[:-5]
by = m.groupby("stem").image_id.apply(list).to_dict()
fold = dict(zip(m.stem, m.fold))


def inst_list(lab):
    ids = np.unique(lab); ids = ids[ids > 0]
    return {int(i): (lab == i) for i in ids}


def iou_mat(la, lb):
    """IoU matrix between instances of two label maps (uint8)."""
    pair = la.astype(np.int32) * 256 + lb.astype(np.int32)
    bc = np.bincount(pair.ravel(), minlength=65536).reshape(256, 256)
    aa = bc.sum(1); ab = bc.sum(0)
    inter = bc.astype(float)
    union = aa[:, None] + ab[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0)
    iou[0, :] = 0; iou[:, 0] = 0
    return iou, aa, ab


def score_preds(preds, lab):
    """preds: list of bool masks (non-overlapping enforced by caller). -> S,TP,FP,FN vs label map."""
    areas = np.bincount(lab.ravel(), minlength=256)
    n_gt = int((areas[1:] > 0).sum())
    S = TP = FP = 0; matched = set()
    for pm in preds:
        inter = np.bincount(lab[pm], minlength=256); inter[0] = 0
        iou = inter / np.maximum(pm.sum() + areas - inter, 1)
        j = int(iou.argmax())
        if iou[j] > 0.5:
            S += iou[j]; TP += 1; matched.add(j)
        else:
            FP += 1
    return np.array([S, TP, FP, n_gt - len(matched)], float)


def pq(t):
    return t[0] / max(t[1] + 0.5 * t[2] + 0.5 * t[3], 1e-9)


acc = {}
def add(key, v):
    acc[key] = acc.get(key, np.zeros(4)) + v


multi = [s for s, r in by.items() if len(r) > 1]
print(len(multi), "multi-reading stems", flush=True)
for n_i, s in enumerate(multi):
    labs = [load_inst(r) for r in by[s]]
    grp = "val" if fold[s] == 0 else "trn"
    n = len(labs)
    for h in range(n):
        others = [i for i in range(n) if i != h]
        # P1: each single other annotator
        for o in others:
            preds = [labs[o] == i for i in np.unique(labs[o]) if i > 0]
            add((grp, n, "P1 single other annotator"), score_preds(preds, labs[h]))
        if n == 3:
            o1, o2 = others
            iou, a1, a2 = iou_mat(labs[o1], labs[o2])
            # greedy match o1<->o2 at IoU > .5 (unique since non-overlapping)
            pairs = [(i, j) for i in range(1, 256) for j in range(1, 256) if iou[i, j] > 0.5]
            m1 = {i for i, j in pairs}; m2 = {j for i, j in pairs}
            cons = {k: [] for k in ("o1", "union", "inter", "o1|o2 larger")}
            for i, j in pairs:
                A_, B_ = labs[o1] == i, labs[o2] == j
                cons["o1"].append(A_); cons["union"].append(A_ | B_); cons["inter"].append(A_ & B_)
                cons["o1|o2 larger"].append(A_ if A_.sum() >= B_.sum() else B_)
            singles = [labs[o1] == i for i in range(1, 256) if a1[i] > 0 and i not in m1] + \
                      [labs[o2] == j for j in range(1, 256) if a2[j] > 0 and j not in m2]
            for k, v in cons.items():
                add((grp, n, f"P2 consensus-only ({k} mask)"), score_preds(v, labs[h]))
            # consensus + all of o1's singles (non-overlapping: o1 singles do not overlap o1 consensus masks)
            s1 = [labs[o1] == i for i in range(1, 256) if a1[i] > 0 and i not in m1]
            add((grp, n, "P2 consensus(o1 mask) + o1 singles (= o1)"), score_preds(cons["o1"] + s1, labs[h]))
            # consensus (o1 masks) + singles of both, dropping singles overlapping already kept
            own = np.zeros_like(labs[0], bool); kept = []
            for msk in cons["o1"] + singles:
                if (msk & own).sum() > 0.2 * msk.sum():
                    continue
                kept.append(msk & ~own); own |= msk
            add((grp, n, "P2 consensus + all singles (union of drawings)"), score_preds(kept, labs[h]))
    if n_i % 50 == 0:
        print(n_i, flush=True)

rows = []
for (grp, n, k), v in sorted(acc.items()):
    rows.append((grp, n, k, pq(v), int(v[1]), int(v[2]), int(v[3]), v[0] / max(v[1], 1)))
    print(f"{grp} n={n} {k:55s} PQ={pq(v):.4f} tp={int(v[1])} fp={int(v[2])} fn={int(v[3])} sq={v[0] / max(v[1], 1):.3f}")
