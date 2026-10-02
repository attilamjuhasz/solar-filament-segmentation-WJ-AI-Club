"""ep8: fixed simple configs through src/assemble (no tuning) + ep4/ep8 q-averaging (checkpoint ensemble, cross-fit)."""
import os
import pickle

import assemble as AS
import numpy as np

_orig = AS.cand_score


def cand_score(c, mode, mprob):
    if mode == "qmp":
        return c["q"] * c["mean_p"]
    if mode == "q48":
        return 0.5 * (c["q"] + c["q4"])
    if mode == "q48mp":
        return 0.5 * (c["q"] + c["q4"]) * c["mean_p"]
    return _orig(c, mode, mprob)


AS.cand_score = cand_score
import q3  # noqa: E402  (CANDS/TAG env -> ep8 context)
from ana import ROOT, cnt, pq  # noqa: E402

ctx = q3.ctx
E4 = os.path.join(ROOT, "runs/s2_r34/cands_val_tta_ep4")
miss = 0
for s in ctx.stems:
    q4 = {(c["idx"], c["level"]): c["q"] for c in pickle.load(open(os.path.join(E4, s + ".pkl"), "rb"))}
    for c in ctx.C[s]:
        k = (c["idx"], c["level"])
        c["q4"] = q4.get(k, c["q"])
        miss += k not in q4
print("ep8 cands without an ep4 twin:", miss)

base = dict(score="q", thr=0.5, ring=0.0, rel=0.0, lam=0.225, lam_small=0.0, small=0, a_min=100, own_frac=0.8,
            levels="AP", grow=0)
cfgs = [("ep4 best_params (qpu thr.6 lam.35 a400 A)",
         dict(score="qpu", thr=0.6, ring=0.0, rel=0.0, lam=0.35, lam_small=0.2, small=400, a_min=400, own_frac=0.8, levels="A", grow=0)),
        ("M0 tuned-on-all ep8 (qm thr.65 lam.3 a100 AP small400/.3)",
         dict(score="qm", thr=0.65, ring=0.0, rel=0.0, lam=0.3, lam_small=0.3, small=400, a_min=100, own_frac=0.8, levels="AP", grow=0))]
for sc in ("q", "qmp", "q48", "q48mp"):
    for lam in (0.2, 0.225, 0.25):
        for thr in (0.5, 0.6):
            cfgs.append((f"{sc} lam{lam} thr{thr} a100 own.8 AP", dict(base, score=sc, lam=lam, thr=thr)))
cfgs.append(("q lam.225 thr.5 a100 own.8 A", dict(base, levels="A")))
cfgs.append(("q lam.225 thr.5 a300 own.8 AP", dict(base, a_min=300)))
cfgs.append(("qmp lam.225 thr.5 a300 own.8 AP", dict(base, score="qmp", a_min=300)))
for nm, P in cfgs:
    fin = {s: AS.assemble(ctx.C[s], P) for s in ctx.stems}
    tA, tB = ctx.pq_counts(fin, ctx.A), ctx.pq_counts(fin, ctx.B)
    print(f"{nm:58s} A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} {cnt(tA + tB)}", flush=True)

G = dict(q3.G1, lam=[0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3])
q3.split_report("M8 q averaged over ep4+ep8 checkpoints", q3.M1, lambda st: (lambda c: 0.5 * (c["q"] + c["q4"])), dict(q3.P1, lam=0.22), G)
q3.split_report("M8b (q4+q8)/2 * mean_p", q3.M1, lambda st: (lambda c: 0.5 * (c["q"] + c["q4"]) * c["mean_p"]), dict(q3.P1, lam=0.2), G)
q3.split_report("M9 plain q (ep8), theory lam grid", q3.M1, lambda st: (lambda c: c["q"]), dict(q3.P1, lam=0.22), G)
