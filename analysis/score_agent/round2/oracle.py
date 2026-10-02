"""Headroom: oracle keep/reject with realized y, and PQ vs scorer quality (y + noise in logit space)."""
from rs3 import *
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
sel = np.isin(LV, list("AP"))
r0 = spearmanr(QMP[sel], Y[sel])[0]
print(f"current q*mean_p: rho={r0:.3f}")
report("ORACLE score = realized y (lam .228)", Y)
# mix: score = calibrated blend of y and current score, to trace PQ vs rho
iso = crossfit(iso_fit(QMP))
for a in (0.1, 0.2, 0.3, 0.5):
    vals = []
    for rep in range(3):
        z = (1 - a) * iso + a * Y + rng.normal(0, 0.0, len(Y))
        tA, tB = assemble_eval(z)
        vals.append((pq(tA), pq(tB), pq(tA + tB), spearmanr(z[sel], Y[sel])[0]))
        break
    v = np.array(vals).mean(0)
    print(f"  blend {a:.1f}*y + {1 - a:.1f}*iso(qmp): rho={v[3]:.3f}  A={v[0]:.4f} B={v[1]:.4f} all={v[2]:.4f}")
# oracle on FN side only: add candidates with y>lam that are currently rejected (keep current kept)
z = QMP.copy(); z[(Y > 0.228) & (QMP < 0.225)] = 1.0
report("oracle rescue: force-keep rejected cands with y>.228", z)
z = QMP.copy(); z[(Y < 0.05) & (QMP >= 0.225)] = 0.0
report("oracle prune: drop kept cands with y<.05", z)
