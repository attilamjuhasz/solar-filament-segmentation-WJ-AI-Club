"""E1 follow-up: old masks + averaged q; paired bootstrap of every variant vs v2."""
from base import *
from ana import Ctx
co = Ctx(cdir=os.path.join(ROOT, "runs/s2_r34/cands_val_tta_last"), load_s1=False)
cn = Ctx(cdir=os.path.join(ROOT, "runs/s2_r34_tta/cands_val_tta_last"), load_s1=False)
qn = {(s, c["idx"], c["level"]): c["q"] for s in cn.stems for c in cn.C[s]}
qo = {(s, c["idx"], c["level"]): c["q"] for s in co.stems for c in co.C[s]}
pqf = lambda t: t[0] / (t[1] + .5 * t[2] + .5 * t[3])
base = np.array([co.G.stats(s, assemble(co.C[s], V2)) for s in co.stems], float)


def variant(ctx, qmap, how):
    for s in ctx.stems:
        for c in ctx.C[s]:
            c.setdefault("q_orig", c["q"])
            o = qmap.get((s, c["idx"], c["level"]), c["q_orig"])
            c["q"] = {"own": c["q_orig"], "avg": 0.5 * (c["q_orig"] + o), "geo": np.sqrt(c["q_orig"] * o), "other": o}[how]
    return np.array([ctx.G.stats(s, assemble(ctx.C[s], V2)) for s in ctx.stems], float)


rng = np.random.default_rng(0)
I = [rng.integers(0, len(base), len(base)) for _ in range(3000)]
A = np.isin(co.stems, co.A)
for nm, ctx, qmap, how in (("E1 masks, E1 q", cn, qo, "own"), ("E1 masks, avg q", cn, qo, "avg"), ("old masks, avg q", co, qn, "avg"),
                           ("old masks, geo-mean q", co, qn, "geo"), ("old masks, E1 q", co, qn, "other")):
    t = variant(ctx, qmap, how)
    d = np.array([pqf(t[i].sum(0)) - pqf(base[i].sum(0)) for i in I])
    print(f"{nm:24s} A={pqf(t[A].sum(0)):.4f} B={pqf(t[~A].sum(0)):.4f} all={pqf(t.sum(0)):.4f}  delta vs v2 {pqf(t.sum(0)) - pqf(base.sum(0)):+.4f} "
          f"(A {pqf(t[A].sum(0)) - pqf(base[A].sum(0)):+.4f}, B {pqf(t[~A].sum(0)) - pqf(base[~A].sum(0)):+.4f}) paired SE {d.std():.4f}", flush=True)
