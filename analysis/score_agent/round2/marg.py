"""Why is lam=.228 optimal when y ~ 1.2*score in the .2-.3 band? Realized value of the marginal additions."""
from base import *
ctx = load_ctx(load_s1=False)
QMP = lambda c, m, a, mp: c["q"] * c["mean_p"]
for lo, hi in ((0.19, 0.228), (0.228, 0.26)):
    vals, uncl, n_block = [], [], 0
    for s in ctx.stems:
        f_hi, p_hi = assemble_prov(ctx.C[s], dict(V2, lam=hi, lam_small=hi), score_fn=QMP)
        f_lo, p_lo = assemble_prov(ctx.C[s], dict(V2, lam=lo, lam_small=lo), score_fn=QMP)
        ids_hi = {id(c) for c, _ in p_hi}
        band = [c for c in ctx.C[s] if c["level"] in "AP" and lo <= c["q"] * c["mean_p"] < hi]
        added = [(m, c) for m, (c, _) in zip(f_lo, p_lo) if id(c) not in ids_hi]
        n_block += len(band) - len(added)
        for (x, y, m), c in added:
            v = np.array([b for b, _, _ in ctx.obj_ious(s, x, y, m)])
            vals.append(float((v * (v > .5)).mean())); uncl.append(c["yv"])
    vals = np.array(vals)
    print(f"score band [{lo},{hi}): cands in band (A/P) {len(vals) + n_block}, actually added {len(vals)}, blocked (own_frac/a_min/overlap) {n_block}; "
          f"realized y of added (clipped mask) mean {vals.mean():.3f}, unclipped y mean {np.mean(uncl):.3f}; break-even PQ/2 = .228")
