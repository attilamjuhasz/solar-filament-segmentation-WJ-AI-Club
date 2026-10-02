import numpy as np
from gbm import GBM
from scipy.stats import spearmanr
rng = np.random.default_rng(1)
X = rng.normal(size=(3000, 10)); p = 1 / (1 + np.exp(-(X[:, 0] * X[:, 1] + np.sin(2 * X[:, 2]) + 0.5 * X[:, 3])))
y = (rng.random(3000) < p).astype(float)
m = GBM(n_trees=300).fit(X[:2000], y[:2000])
pr = m.predict(X[2000:])
print("synthetic: held-out spearman(pred, true p)", round(spearmanr(pr, p[2000:])[0], 3), " logistic-linear would miss the x0*x1 term")
from rs3 import *
cols = ["q", "mean_p", "peak_u", "s1u_mean", "s1p_p90", "mprob", "log_area", "lvA", "lvP"]
X0 = F[:, [ix[c] for c in cols]]
for tr in (HA, ~HA):
    g = GBM().fit(X0[tr], Y[tr], W[tr])
    pi, po = g.predict(X0[tr]), g.predict(X0[~tr])
    print(f"in-half rho {spearmanr(pi, Y[tr])[0]:.3f} wMSE {np.sum(W[tr] * (pi - Y[tr]) ** 2) / W[tr].sum():.4f} | held-out rho {spearmanr(po, Y[~tr])[0]:.3f} wMSE {np.sum(W[~tr] * (po - Y[~tr]) ** 2) / W[~tr].sum():.4f} | qmp held-out rho {spearmanr(QMP[~tr], Y[~tr])[0]:.3f}")
