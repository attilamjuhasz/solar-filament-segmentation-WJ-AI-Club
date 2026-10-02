"""Q1: reliability of q. y = mean over readings of IoU*1[IoU>.5] of the candidate's (unclipped) mask."""
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

from ana import TAG, Ctx, cand_mask

ctx = Ctx(load_s1=False)
rows = []  # stem, half, level, q, peak_u, mean_p, mprob5, area5, y5, m5, y6, m6
for s in ctx.stems:
    half = 0 if s in set(ctx.A) else 1
    for c in ctx.C[s]:
        r = [s, half, c["level"], c["q"], c["peak_u"], c["mean_p"]]
        for thr in (0.5, 0.6):
            P = dict(thr=thr, ring=0.0, rel=0.0)
            m, area, mprob = cand_mask(c, P)
            if area == 0:
                ious = [0.0] * len(ctx.G.by_stem[s])
            else:
                ious = [b for b, _, _ in ctx.obj_ious(s, c["x"], c["y"], m)]
            ious = np.array(ious)
            if thr == 0.5:
                r += [mprob, area]
            r += [float((ious * (ious > 0.5)).mean()), float((ious > 0.5).mean())]
        rows.append(r)
pickle.dump(rows, open(f"q1_rows_{TAG}.pkl", "wb"))
R = np.array([r[3:] for r in rows], float)
lv = np.array([r[2] for r in rows])
q, pu, mp, mprob, area, y5, m5, y6, m6 = R.T


def table(sel, name, y, m, score=None, edges=(0, .1, .2, .3, .35, .4, .45, .5, .55, .6, 1.01)):
    sc = q if score is None else score
    print(f"\n[{name}] n={sel.sum()}  mean score={sc[sel].mean():.3f} mean y={y[sel].mean():.3f}  "
          f"spearman(score,y)={spearmanr(sc[sel], y[sel])[0]:.3f}")
    print("   score bin   |    n  | mean score | mean y=E[IoU*1(IoU>.5)] | P(match) | y/score")
    b = np.digitize(sc, edges) - 1
    for i in range(len(edges) - 1):
        k = sel & (b == i)
        if k.sum() < 5:
            continue
        print(f"  [{edges[i]:.2f},{edges[i + 1]:.2f}) | {k.sum():5d} | {sc[k].mean():.3f}      | {y[k].mean():.3f}"
              f"                   | {m[k].mean():.3f}    | {y[k].mean() / max(sc[k].mean(), 1e-6):.2f}")


allm = np.ones(len(q), bool)
table(allm, "ALL levels, mask thr .5", y5, m5)
for L in "APBC":
    table(lv == L, f"level {L}, mask thr .5", y5, m5)
table(allm, "ALL levels, mask thr .6", y6, m6)
table(lv == "A", "level A, mask thr .6", y6, m6)
print("\nalternative scores vs y (thr .5), spearman, all levels / level A:")
for nm, sc in [("q", q), ("qpu", q * (0.5 + 0.5 * pu)), ("q*mprob", q * mprob), ("peak_u", pu),
               ("mean_p", mp), ("mprob", mprob), ("log area", np.log1p(area))]:
    print(f"  {nm:9s} {spearmanr(sc, y5)[0]:.3f} / {spearmanr(sc[lv == 'A'], y5[lv == 'A'])[0]:.3f}")
small = area < 400
table(small, "area<400 (thr .5), all levels", y5, m5)
table(~small, "area>=400 (thr .5), all levels", y5, m5)
