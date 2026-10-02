"""Stacked rescorer: soft-target logistic regression on candidate features, fit on one half, applied to the other."""
import pickle
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

import q3
from q3 import TAG, ctx, split_report, M1, P1, G1

D = pickle.load(open(f"feats_{TAG}.pkl", "rb"))
NAMES, F = D["names"], D["F"]
k = 0
for s in ctx.stems:
    for c in ctx.C[s]:
        c["f"] = F[k]
        k += 1
Y = np.array([c["yv"] for s in ctx.stems for c in ctx.C[s]])
HALF = np.array([0 if s in set(ctx.A) else 1 for s in ctx.stems for c in ctx.C[s]])
SETS = {
    "q": ["q"],
    "q+level": ["q", "lvA", "lvP", "lvB", "lvC"],
    "q+level+S1": ["q", "logit_q", "lvA", "lvP", "lvB", "lvC", "peak_u", "mean_p", "s1p_mean", "s1p_p90", "s1u_mean"],
    "q+level+S1+S2": ["q", "logit_q", "lvA", "lvP", "lvB", "lvC", "peak_u", "mean_p", "s1p_mean", "s1p_p90",
                      "s1u_mean", "mprob", "p95"],
    "+agree": ["q", "logit_q", "lvA", "lvP", "lvB", "lvC", "peak_u", "mean_p", "s1p_mean", "s1p_p90", "s1u_mean",
               "mprob", "p95", "n_agree", "max_q_other"],
    "all": NAMES,
}


def fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]

    def f(b):
        z = Z @ b
        p = 1 / (1 + np.exp(-z))
        l = -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).sum() + alpha * (b[1:] ** 2).sum()
        g = Z.T @ (p - y) + 2 * alpha * np.r_[0, b[1:]]
        return l, g

    b = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ b))), b


idx = {n: i for i, n in enumerate(NAMES)}


def spearman_table():
    print("held-out Spearman(score, realized y)  [fit A -> eval B | fit B -> eval A]")
    for nm, cols in SETS.items():
        ci = [idx[c] for c in cols]
        out = []
        for tr, te in ((0, 1), (1, 0)):
            pred, b = fit(F[HALF == tr][:, ci], Y[HALF == tr])
            out.append(spearmanr(pred(F[HALF == te][:, ci]), Y[HALF == te])[0])
        print(f"  {nm:15s} {out[0]:.3f} | {out[1]:.3f}")
    ci = [idx[c] for c in NAMES]
    pred, b = fit(F[:, ci], Y)
    print("standardized coefficients (all features, fit on all val):")
    for n, v in sorted(zip(NAMES, b[1:]), key=lambda t: -abs(t[1])):
        print(f"  {n:12s} {v:+.3f}")


def make_fit(cols):
    ci = [idx[c] for c in cols]

    def fit_stems(stems):
        st = set(stems)
        sel = np.array([s in st for s in ctx.stems for c in ctx.C[s]])
        pred, _ = fit(F[sel][:, ci], Y[sel])
        return lambda c: float(pred(c["f"][ci][None])[0])
    return fit_stems


if __name__ == "__main__":
    spearman_table()
    todo = sys.argv[1:] or ["q+level+S1+S2", "+agree", "all"]
    for nm in todo:
        split_report(f"M6 rescorer [{nm}]", M1, make_fit(SETS[nm]), P1, G1)
