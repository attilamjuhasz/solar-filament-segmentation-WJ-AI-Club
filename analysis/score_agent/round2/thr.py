"""Per-candidate threshold variants with the v2 selection held fixed (pure shape effect), plus re-selected."""
from base import *
ctx = load_ctx(load_s1=False)
QMP = lambda c, m, a, mp: c["q"] * c["mean_p"]
FP = {s: assemble_prov(ctx.C[s], V2, score_fn=QMP) for s in ctx.stems}


def fixed_sel(P):
    out = {}
    for s in ctx.stems:
        F, prov = FP[s]
        owner = np.zeros((2048, 2048), bool)
        res = []
        for (c, sc) in prov:
            m, a, _ = cand_mask(c, P)
            if a == 0:
                continue
            x, y = c["x"], c["y"]
            free = m & ~owner[y:y + m.shape[0], x:x + m.shape[1]]
            owner[y:y + m.shape[0], x:x + m.shape[1]] |= free
            res.append((x, y, free))
        out[s] = res
    return out


for thr in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
    P = dict(thr=thr, ring=0.0, rel=0.0)
    half_report(ctx, fixed_sel(P), f"fixed selection, thr {thr}")
for rel in (0.5, 0.6, 0.7, 0.8):
    for thr in (0.35, 0.45):
        P = dict(thr=thr, ring=0.0, rel=rel)
        half_report(ctx, fixed_sel(P), f"fixed selection, thr max({thr}, {rel}*p95)")
for ring in (0.3, 0.4):
    P = dict(thr=0.5, ring=ring, rel=0.0)
    half_report(ctx, fixed_sel(P), f"fixed selection, thr .5 ring {ring}")
for thr in (0.4, 0.45, 0.55, 0.6):
    half_report(ctx, {s: assemble(ctx.C[s], dict(V2, thr=thr)) for s in ctx.stems}, f"v2 reselected, thr {thr}")
