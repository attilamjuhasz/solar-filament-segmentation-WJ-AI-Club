"""Fixed configs through src/assemble.assemble (with a 'qmp' = q*mean_p score added), no tuning."""
import assemble as AS
from ana import Ctx, cnt, pq

_orig = AS.cand_score


def cand_score(c, mode, mprob):
    if mode == "qmp":
        return c["q"] * c["mean_p"]
    return _orig(c, mode, mprob)


AS.cand_score = cand_score
ctx = Ctx(load_s1=False)
base = dict(score="qmp", thr=0.5, ring=0.0, rel=0.0, lam=0.3, lam_small=0.3, small=0, a_min=300, own_frac=0.8,
            levels="AP", grow=0)
cfgs = [("CURRENT best_params (qpu thr.6 lam.35 a400 own.8 A)",
         dict(score="qpu", thr=0.6, ring=0.0, rel=0.0, lam=0.35, lam_small=0.2, small=400, a_min=400, own_frac=0.8, levels="A", grow=0))]
for thr in (0.45, 0.5, 0.55, 0.6):
    for lam in (0.25, 0.3, 0.35):
        cfgs.append((f"qmp thr{thr} lam{lam} a300 own.8 AP", dict(base, thr=thr, lam=lam)))
for lv in ("A", "APB", "APBC"):
    cfgs.append((f"qmp thr.5 lam.3 a300 own.8 {lv}", dict(base, levels=lv)))
for a in (100, 200, 400):
    cfgs.append((f"qmp thr.5 lam.3 a{a} own.8 AP", dict(base, a_min=a)))
for of in (0.6,):
    cfgs.append((f"qmp thr.5 lam.3 a300 own{of} AP", dict(base, own_frac=of)))
cfgs.append(("qmp thr.5 lam.3 a300 own.8 AP grow1", dict(base, grow=1)))
for nm, P in cfgs:
    fin = {s: AS.assemble(ctx.C[s], P) for s in ctx.stems}
    tA, tB = ctx.pq_counts(fin, ctx.A), ctx.pq_counts(fin, ctx.B)
    print(f"{nm:55s} A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} {cnt(tA + tB)}", flush=True)
