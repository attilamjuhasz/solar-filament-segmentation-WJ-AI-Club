"""Q3b: val PQ by annotator; reweight to the all-fold annotator mix. Pairwise annotator PQ by fold."""
from base import *
import pandas as pd
from common import load_meta, load_inst
ctx = load_ctx(load_s1=False)
fin = {s: assemble(ctx.C[s], V2) for s in ctx.stems}
meta = load_meta()
meta["stem"] = meta.file_name.str[:-5]
meta["ann"] = meta.image_id.str.split("-").str[0]
mv = meta[meta.fold == 0]
rows = []
for s in ctx.stems:
    rids = list(mv[mv.stem == s].image_id)  # FastGT reading order follows meta order
    for (lab, areas), rid in zip(ctx.G.by_stem[s], rids):
        t = np.zeros(4)
        # stats for one reading
        g = type("G", (), {})()
        n_gt = int((areas[1:] > 0).sum()); matched = set(); S = TP = FP = 0
        for x, y, m in fin[s]:
            h, w = m.shape
            inter = np.bincount(lab[y:y + h, x:x + w][m], minlength=256); inter[0] = 0
            iou = inter / np.maximum(m.sum() + areas - inter, 1); j = int(iou.argmax())
            if iou[j] > .5:
                S += iou[j]; TP += 1; matched.add(j)
            else:
                FP += 1
        rows.append((s, rid.split("-")[0], S, TP, FP, n_gt - len(matched), n_gt))
df = pd.DataFrame(rows, columns=["stem", "ann", "S", "TP", "FP", "FN", "ngt"])
df["grp"] = df.ann.str[:4]
def pqg(g): return g.S.sum() / (g.TP.sum() + .5 * g.FP.sum() + .5 * g.FN.sum())
print("overall", round(pqg(df), 4), "readings", len(df))
out = []
for k, g in df.groupby("grp"):
    out.append((k, len(g), round(pqg(g), 3), round(g.ngt.mean(), 1), round(g.FP.sum() / max(len(g), 1), 2)))
print("by annotator group (grp, n readings, PQ, mean #GT, FP/reading):")
for o in out: print("  ", o)
# reweight to all-fold annotator-group mix
allmix = meta.ann.str[:4].value_counts(normalize=True)
vmix = df.grp.value_counts(normalize=True)
w = df.grp.map(lambda k: allmix.get(k, 0) / vmix[k])
S, TP, FP, FN = [(df[c] * w).sum() for c in ("S", "TP", "FP", "FN")]
print(f"val PQ reweighted to all-fold annotator-group mix: {S / (TP + .5 * FP + .5 * FN):.4f}")
# per-reading PQ macro (alternative metric variants)
df["pq_r"] = df.S / (df.TP + .5 * df.FP + .5 * df.FN).clip(lower=1e-9)
print(f"metric variants on val: pooled {pqg(df):.4f}; macro over readings {df.pq_r.mean():.4f}; "
      f"macro over images {df.groupby('stem').apply(pqg).mean():.4f}")
# pairwise annotator agreement (GT vs GT) by fold: is fold 0 more consistent?
import itertools
res = {}
for f in range(5):
    mf = meta[meta.fold == f]
    t = np.zeros(4)
    for s, g in mf.groupby("stem"):
        rids = list(g.image_id)
        if len(rids) < 2:
            continue
        labs = [load_inst(r) for r in rids]
        for a, b in itertools.permutations(range(len(rids)), 2):
            la, lb = labs[a], labs[b]
            areas_b = np.bincount(lb.ravel(), minlength=256)
            n_gt = int((areas_b[1:] > 0).sum()); matched = set()
            for j in np.unique(la)[1:]:
                m = la == j
                inter = np.bincount(lb[m], minlength=256); inter[0] = 0
                iou = inter / np.maximum(m.sum() + areas_b - inter, 1); k = int(iou.argmax())
                if iou[k] > .5:
                    t[0] += iou[k]; t[1] += 1; matched.add(k)
                else:
                    t[2] += 1
            t[3] += n_gt - len(matched)
    res[f] = pq(t)
print("inter-annotator pooled PQ by fold:", {k: round(v, 3) for k, v in res.items()})
