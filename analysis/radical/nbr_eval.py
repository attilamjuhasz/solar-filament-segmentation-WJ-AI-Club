"""Evaluate idea (b): how much do warped neighbour labels / predictions say about the target frame?"""
import os, sys, pickle
import numpy as np
from decode import *   # C, G, R, NR, A, B, stems, stats_reading, assemble_sel, yv, pq, cnt
HERE = os.path.dirname(os.path.abspath(__file__))
N = pickle.load(open(os.path.join(HERE, "nbr.pkl"), "rb"))
BINS = [(0, 1), (1, 6), (6, 12), (12, 24), (24, 48)]

print("=== val stems with >=1 labelled neighbour by max hours")
for h in (1, 6, 12, 24, 48):
    n = sum(any(dh <= h and k == "lab" for dh, t, k, f in N["nbinfo"][s]) for s in stems)
    n2 = sum(any(dh <= h and k == "lab" and f != 0 for dh, t, k, f in N["nbinfo"][s]) for s in stems)
    print(f"  <= {h:2d} h: {n} stems (of which with an other-fold neighbour: {n2})")

print("=== target GT instances: best IoU with warped neighbour GT (nearest labelled neighbour per bin)")
for lo, hi in BINS:
    ious, covs, n = [], [], 0
    for s in stems:
        for ri, gl in enumerate(N["gtcov"][s]):
            for g, lst in enumerate(gl):
                L = [z for z in lst if lo < z[0] <= hi or (lo == 0 and z[0] <= hi)]
                if not L:
                    continue
                dmin = min(z[0] for z in L)
                L = [z for z in L if z[0] == dmin and z[4] > 0.5]   # nearest neighbour, visible
                if not L:
                    continue
                ious.append(max(z[2] for z in L)); covs.append(max(z[3] for z in L))
    ious, covs = np.array(ious), np.array(covs)
    if len(ious):
        print(f"  dt {lo:2d}-{hi:2d}h: n={len(ious):4d} IoU>.5 {np.mean(ious > .5):.3f}  IoU>.3 {np.mean(ious > .3):.3f}  "
              f"cov3px>.5 {np.mean(covs > .5):.3f}  median IoU {np.median(ious):.3f}")

print("=== direct transfer PQ (warped neighbour reading as prediction vs target reading)")
for lo, hi in BINS:
    t = np.zeros(4)
    for s, ri, tt, dh, S, TP, FP, FN in N["direct"]:
        if lo <= dh < hi:
            t += [S, TP, FP, FN]
    if t[1] + t[2] + t[3]:
        print(f"  dt {lo:2d}-{hi:2d}h PQ={pq(t):.4f} {cnt(t)}")


# ---- candidate features
def feat(s, k, hmax, how):
    L = [d for d in N["cand"][s][k] if d["kind"] == "lab" and d["dh"] <= hmax and d["vfrac"] > 0.5]
    if not L:
        return np.nan
    dmin = min(d["dh"] for d in L)
    L = [d for d in L if d["dh"] <= dmin + 1e-6]
    if how == "match3":
        return float(np.mean([np.mean(d["iou"] > 0.3) for d in L]))
    if how == "match2":
        return float(np.mean([np.mean(d["iou"] > 0.2) for d in L]))
    if how == "cov":
        return float(np.mean([np.mean(d["cov"]) for d in L]))
    raise ValueError


def s1feat(s, k, hmax):
    L = [d for d in N["cand"][s][k] if d["kind"] == "s1" and d["dh"] <= hmax and d["vfrac"] > 0.5]
    if not L:
        return np.nan
    return float(np.mean([d["mpd"][0] for d in L]))


print("=== candidate-level: does the neighbour feature separate y (A/P candidates)?")
for hmax in (12, 24, 48):
    for how in ("match3", "cov"):
        F, Y, Q = [], [], []
        for s in stems:
            for k, r in enumerate(R[s]):
                if r["level"] not in "AP":
                    continue
                f = feat(s, k, hmax, how)
                if np.isnan(f):
                    continue
                F.append(f); Y.append(yv(r).mean()); Q.append(r["q"] * r["mean_p"])
        F, Y, Q = map(np.array, (F, Y, Q))
        from scipy.stats import spearmanr
        if len(F) > 20:
            # partial info: within score bins, does F separate Y?
            near = (Q > 0.12) & (Q < 0.35)
            print(f"  h<={hmax} {how}: n={len(F)} rho(F,y)={spearmanr(F, Y)[0]:.3f} rho(Q,y)={spearmanr(Q, Y)[0]:.3f} "
                  f"rho(Q+0.3F,y)={spearmanr(Q + 0.3 * F, Y)[0]:.3f} | near-margin n={near.sum()} "
                  f"rho(F,y)={spearmanr(F[near], Y[near])[0]:.3f} rho(Q,y)={spearmanr(Q[near], Y[near])[0]:.3f}")


def run_score(name, score_fn, lam, sub):
    tA, tB = np.zeros(4), np.zeros(4)
    for s in sub:
        items = []
        for k, r in enumerate(R[s]):
            if r["level"] not in "AP":
                continue
            sc = score_fn(s, k, r)
            if sc >= lam:
                items.append((sc, k, 0.5))
        fin = assemble_sel(s, items)
        v = sum(stats_reading(s, ri, fin) for ri in range(NR[s]))
        if s in A: tA += v
        else: tB += v
    print(f"  {name:60s} A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} {cnt(tA + tB)}", flush=True)
    return pq(tA), pq(tB), pq(tA + tB)


print("=== assembly with neighbour-label feature (stems with a labelled neighbour <= h)")
for hmax in (12, 24, 48):
    sub = [s for s in stems if any(dh <= hmax and k == "lab" for dh, t, k, f in N["nbinfo"][s])]
    print(f" h<={hmax}: {len(sub)} stems, {sum(NR[s] for s in sub)} readings")
    base = lambda s, k, r: r["q"] * r["mean_p"]
    run_score("v2 q*mean_p", base, 0.225, sub)
    for how in ("match3", "cov"):
        for b in (0.1, 0.2, 0.3, 0.5):
            def sc(s, k, r, b=b, how=how, hmax=hmax):
                f = feat(s, k, hmax, how)
                f = 0.5 if np.isnan(f) else f
                return r["q"] * r["mean_p"] * (1 - b + 2 * b * f)
            run_score(f"qmp*(1-b+2bF) {how} b={b}", sc, 0.225, sub)

print("=== assembly with neighbour S1 (out-of-sample val/test neighbours)")
for hmax in (6, 24, 48):
    sub = [s for s in stems if any(dh <= hmax and (k == "test" or f == 0) for dh, t, k, f in N["nbinfo"][s])]
    print(f" h<={hmax}: {len(sub)} stems")
    base = lambda s, k, r: r["q"] * r["mean_p"]
    run_score("v2 q*mean_p", base, 0.225, sub)
    for w in (0.2, 0.35, 0.5):
        def sc(s, k, r, w=w, hmax=hmax):
            f = s1feat(s, k, hmax)
            mp = r["mean_p"] if np.isnan(f) else (1 - w) * r["mean_p"] + w * f
            return r["q"] * mp
        run_score(f"q*((1-w)mean_p + w*nbr_mean_p) w={w}", sc, 0.225, sub)
