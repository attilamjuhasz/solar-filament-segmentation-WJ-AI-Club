"""Evaluate a new S2 candidate dir vs the current one: v2 config, lam curve, calibration, rank quality, 2-model ensemble.
usage: python e1_eval.py runs/s2_r34_tta/cands_val_tta_last"""
import sys
from base import *
from ana import Ctx
from scipy.stats import spearmanr
NEW = sys.argv[1]
OLD = "runs/s2_r34/cands_val_tta_last"
P5 = dict(thr=.5, ring=0, rel=0)


def add_y(ctx):
    for s in ctx.stems:
        for c in ctx.C[s]:
            m, a, _ = cand_mask(c, P5)
            if a == 0:
                c["yv"] = 0.0; continue
            v = np.array([b for b, _, _ in ctx.obj_ious(s, c["x"], c["y"], m)])
            c["yv"] = float((v * (v > .5)).mean())


co = Ctx(cdir=os.path.join(ROOT, OLD), load_s1=False); add_y(co)
cn = Ctx(cdir=os.path.join(ROOT, NEW), load_s1=False); add_y(cn)
for tag, ctx in (("OLD s2_r34", co), ("NEW", cn)):
    print(f"\n=== {tag}")
    half_report(ctx, {s: assemble(ctx.C[s], V2) for s in ctx.stems}, f"{tag} v2 config")
    for lam in (0.18, 0.2, 0.25, 0.275, 0.3):
        half_report(ctx, {s: assemble(ctx.C[s], dict(V2, lam=lam, lam_small=lam)) for s in ctx.stems}, f"{tag} lam {lam}")
    cs = [c for s in ctx.stems for c in ctx.C[s] if c["level"] in "AP"]
    q = np.array([c["q"] for c in cs]); mp = np.array([c["mean_p"] for c in cs]); y = np.array([c["yv"] for c in cs])
    print(f"  A/P cands {len(cs)}: mean q {q.mean():.3f}, mean q*mean_p {(q * mp).mean():.3f}, mean y {y.mean():.3f}; spearman q {spearmanr(q, y)[0]:.3f}, q*mean_p {spearmanr(q * mp, y)[0]:.3f}")
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .6), (.6, 1)):
        k = (q * mp >= lo) & (q * mp < hi)
        if k.sum():
            print(f"    score [{lo},{hi}) n={k.sum():4d} mean score {(q * mp)[k].mean():.3f} mean y {y[k].mean():.3f}")
# ensemble (same proposals -> key (idx, level))
qo = {(s, c["idx"], c["level"]): c["q"] for s in co.stems for c in co.C[s]}
miss = 0
for s in cn.stems:
    for c in cn.C[s]:
        k = (s, c["idx"], c["level"])
        c["q_old"] = qo.get(k, c["q"]); miss += k not in qo
print(f"\nensemble: new cands without an old twin: {miss}")
import assemble as AS
_o = AS.cand_score
AS.cand_score = lambda c, mode, mp: (0.5 * (c["q"] + c["q_old"]) * c["mean_p"] if mode == "ens" else
                                     np.sqrt(c["q"] * c["q_old"]) * c["mean_p"] if mode == "ensg" else _o(c, mode, mp))
for mode in ("ens", "ensg"):
    for lam in (0.2, 0.225, 0.25):
        half_report(cn, {s: AS.assemble(cn.C[s], dict(V2, score=mode, lam=lam, lam_small=lam)) for s in cn.stems}, f"ensemble {mode} (new masks) lam {lam}")
cs = [c for s in cn.stems for c in cn.C[s] if c["level"] in "AP"]
y = np.array([c["yv"] for c in cs]); e = np.array([0.5 * (c["q"] + c["q_old"]) * c["mean_p"] for c in cs])
print(f"  ensemble spearman {spearmanr(e, y)[0]:.3f}, mean score {e.mean():.3f}")
