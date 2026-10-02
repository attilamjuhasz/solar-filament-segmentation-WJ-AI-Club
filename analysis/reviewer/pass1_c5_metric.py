import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pycocotools import mask as mu
sys.path.insert(0, "src")
from common import DATA
from metric import load_gt_rles, reading_stats, evaluate
from scipy.optimize import linear_sum_assignment

ANN = os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
gt = load_gt_rles(ANN)
n_pairs = n_overlap = n_big = 0
worst = []
for rid, (stem, g) in gt.items():
    if len(g) < 2:
        continue
    inter_area = []
    a = mu.area(g).astype(float)
    iou = np.asarray(mu.iou(g, g, [0] * len(g)))
    np.fill_diagonal(iou, 0)
    n_pairs += len(g) * (len(g) - 1) // 2
    ov = np.triu(iou > 0)
    n_overlap += ov.sum()
    if (iou > 0.2).any():
        n_big += 1
        i, j = np.unravel_index(iou.argmax(), iou.shape)
        worst.append((round(float(iou.max()), 3), rid, int(a[i]), int(a[j])))
print("GT pairs within a reading:", n_pairs, "overlapping pairs:", n_overlap, "readings with a pair IoU>0.2:", n_big)
print("worst:", sorted(worst, reverse=True)[:8])


# Hungarian vs threshold-only matching on adversarial case: pred covering two overlapping GTs
def hung(p, g, thr=0.5):
    if not p or not g:
        return 0.0, 0, len(p), len(g)
    iou = np.asarray(mu.iou(p, g, [0] * len(g)))
    r, c = linear_sum_assignment(-iou)
    ok = iou[r, c] > thr
    tp = int(ok.sum())
    return float(iou[r, c][ok].sum()), tp, len(p) - tp, len(g) - tp


# compare on real GT readings using another annotator as prediction (overlaps allowed in that "pred")
diff = 0
tot = [np.zeros(4), np.zeros(4)]
stems = {}
for rid, (s, g) in gt.items():
    stems.setdefault(s, []).append(rid)
for s, rids in stems.items():
    if len(rids) < 2:
        continue
    p = gt[rids[0]][1]
    for rid in rids[1:]:
        a = reading_stats(p, gt[rid][1]); b = hung(p, gt[rid][1])
        tot[0] += a; tot[1] += b
        if a[1:] != b[1:]:
            diff += 1
print("readings where threshold-matching != Hungarian:", diff, "totals thr", tot[0], "hung", tot[1])

# edge cases
e = np.zeros((2048, 2048), np.uint8); e[10:20, 10:20] = 1
E = mu.encode(np.asfortranarray(e))
print("no preds:", reading_stats([], [E]), "| no gt:", reading_stats([E], []))
gt1 = {"r1": ("s1", [E]), "r2": ("s1", [E]), "r3": ("s2", [E])}
print("evaluate preds for s1 only (s2 missing), extra stem s9 ignored:", evaluate({"s1": [e.astype(bool)], "s9": [e]}, gt1))
print("evaluate empty preds:", evaluate({}, gt1)[0], "| empty gt:", evaluate({"s1": [e]}, {})[0])
