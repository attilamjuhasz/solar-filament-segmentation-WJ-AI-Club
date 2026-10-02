"""Q3b: per object, S2 final mask vs the best-overlapping S1-only instance. Which matches readings better?
Oracle swap upper bound + simple swap rules, reported on both halves (BEST config)."""
import numpy as np

from ana import BEST, S1BEST, Ctx, assemble_prov, pq

ctx = Ctx()


def qpu(c, m, area, mprob):
    return c["q"] * (0.5 + 0.5 * c["peak_u"])


def yval(s, x, y, m):
    ious = np.array([b for b, _, _ in ctx.obj_ious(s, x, y, m)])
    return float((ious * (ious > 0.5)).mean()), float((ious > 0.5).mean())


def inter(a, b):
    ax, ay, am = a
    bx, by, bm = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + am.shape[1], bx + bm.shape[1]), min(ay + am.shape[0], by + bm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).sum())


recs = []  # stem, idx, q, s2 y, s1 y, s2 area, s1 area, pair iou, s1 index
fin2, fin1 = {}, {}
for s in ctx.stems:
    f2, prov = assemble_prov(ctx.C[s], dict(BEST), score_fn=qpu)
    f1 = ctx.s1_inst(s, **S1BEST)
    fin2[s], fin1[s] = f2, f1
    for k, ((x, y, m), (c, sc)) in enumerate(zip(f2, prov)):
        a2 = int(m.sum())
        best, bj = 0.0, -1
        for j, (x1, y1, m1) in enumerate(f1):
            io = inter((x, y, m), (x1, y1, m1))
            iou = io / max(a2 + int(m1.sum()) - io, 1)
            if iou > best:
                best, bj = iou, j
        y2 = yval(s, x, y, m)[0]
        y1v = yval(s, *f1[bj])[0] if bj >= 0 else np.nan
        a1 = int(f1[bj][2].sum()) if bj >= 0 else 0
        recs.append((s, k, c["q"], y2, y1v, a2, a1, best, bj))

R = np.array([r[2:] for r in recs], float)
q, y2, y1, a2, a1, piou, bj = R.T
pair = piou >= 0.5
print(f"S2 finals={len(R)}, with an S1 instance at mask-IoU>=.5: {pair.sum()} ({pair.mean():.2f}); mean pair IoU {piou[pair].mean():.3f}")
print(f"  on paired objects: mean y S2 {y2[pair].mean():.4f} vs S1 {y1[pair].mean():.4f}; "
      f"S2 better {np.mean(y2[pair] > y1[pair] + 1e-9):.2f}, S1 better {np.mean(y1[pair] > y2[pair] + 1e-9):.2f}, tie {np.mean(np.abs(y1[pair] - y2[pair]) < 1e-9):.2f}")
print(f"  area ratio S1/S2 median {np.median(a1[pair] / a2[pair]):.2f}")
for lo, hi in [(0, 0.4), (0.4, 0.5), (0.5, 0.55), (0.55, 1)]:
    k = pair & (q >= lo) & (q < hi)
    if k.sum():
        print(f"  q in [{lo},{hi}): n={k.sum()} y S2 {y2[k].mean():.3f} S1 {y1[k].mean():.3f}")
for lo, hi in [(0, 0.8), (0.8, 1.0), (1.0, 1.25), (1.25, 99)]:
    r = a1 / np.maximum(a2, 1)
    k = pair & (r >= lo) & (r < hi)
    if k.sum():
        print(f"  S1/S2 area ratio in [{lo},{hi}): n={k.sum()} y S2 {y2[k].mean():.3f} S1 {y1[k].mean():.3f}")


def swapped(rule):
    out = {}
    i = 0
    for s in ctx.stems:
        f2 = fin2[s]
        owner = np.zeros((2048, 2048), bool)
        objs = []
        for k in range(len(f2)):
            r = R[i]
            i += 1
            use1 = r[5] >= 0.5 and rule(r)
            objs.append(fin1[s][int(r[6])] if use1 else f2[k])
        res = []
        for x, y, m in objs:  # keep disjoint (earlier = higher score wins)
            h, w = m.shape
            free = m & ~owner[y:y + h, x:x + w]
            if free.sum() >= 0.5 * m.sum() and free.sum() > 0:
                owner[y:y + h, x:x + w] |= free
                res.append((x, y, free))
        out[s] = res
    return out


rules = {
    "S2 only (BEST)": lambda r: False,
    "oracle: better of S1/S2 per object": lambda r: r[2] > r[1],
    "always S1 mask when paired": lambda r: True,
    "S1 mask if S1 larger": lambda r: r[4] > r[3],
    "S1 mask if S1 smaller": lambda r: r[4] < r[3],
    "S1 mask if q<.5": lambda r: r[0] < 0.5,
}
for nm, rule in rules.items():
    fin = swapped(rule)
    tA, tB = ctx.pq_counts(fin, ctx.A), ctx.pq_counts(fin, ctx.B)
    print(f"{nm:40s} A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f}")
