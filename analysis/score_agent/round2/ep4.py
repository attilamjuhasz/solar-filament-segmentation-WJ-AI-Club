from base import *
from scipy.stats import spearmanr
from ana import Ctx
import ana
r = {}
for tag, cd in (("ep4", "runs/s2_r34/cands_val_tta_ep4"), ("ep8", "runs/s2_r34/cands_val_tta_last")):
    ctx = Ctx(cdir=os.path.join(ROOT, cd), load_s1=False)
    fin = {s: assemble(ctx.C[s], V2) for s in ctx.stems}
    half_report(ctx, fin, f"{tag} v2 config (q*mean_p, lam .225)")
    for lam in (0.25, 0.275, 0.3):
        half_report(ctx, {s: assemble(ctx.C[s], dict(V2, lam=lam)) for s in ctx.stems}, f"{tag} lam {lam}")
    rows = pickle.load(open(f"../score/q1_rows_{tag}.pkl", "rb"))
    R = np.array([x[3:] for x in rows], float); lv = np.array([x[2] for x in rows])
    q, pu, mp, mprob, area, y5 = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4], R[:, 5]
    k = np.isin(lv, list("AP"))
    print(f"  {tag}: A/P cands {k.sum()}, spearman(q,y) {spearmanr(q[k], y5[k])[0]:.3f}, spearman(q*mean_p,y) {spearmanr((q * mp)[k], y5[k])[0]:.3f}, mean q {q[k].mean():.3f}, mean y {y5[k].mean():.3f}")
