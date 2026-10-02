"""Q2b: cross-fit rescorers (numpy GBM / logistic+interactions / isotonic) vs q*mean_p, assembled with v2 params."""
from base import *
import assemble as AS
from gbm import GBM
from scipy.optimize import minimize
from scipy.stats import spearmanr

_orig = AS.cand_score
AS.cand_score = lambda c, mode, mprob: c["rs"] if mode == "rs" else _orig(c, mode, mprob)
ctx = load_ctx(load_s1=False)
D = pickle.load(open("feats2_val.pkl", "rb"))
N, F = list(D["names"]), D["F"]
ix = {n: i for i, n in enumerate(N)}
CANDS = [c for s in ctx.stems for c in ctx.C[s]]
Y = np.array([c["yv"] for c in CANDS])
STEM = np.array([s for s in ctx.stems for c in ctx.C[s]])
W = np.array([len(ctx.G.by_stem[s]) for s in STEM], float)
HA = np.isin(STEM, ctx.A)
LV = np.array([c["level"] for c in CANDS])
QMP = F[:, ix["qmp"]]


def assemble_eval(score, lam=0.228, P=None):
    for c, v in zip(CANDS, score):
        c["rs"] = float(v)
    P = dict(V2, score="rs", lam=lam, lam_small=lam) if P is None else P
    fin = {s: AS.assemble(ctx.C[s], P) for s in ctx.stems}
    tA, tB = ctx.pq_counts(fin, ctx.A), ctx.pq_counts(fin, ctx.B)
    return tA, tB


def lam_curve(score, lams=(0.18, 0.2, 0.21, 0.22, 0.228, 0.24, 0.25, 0.26, 0.28, 0.3)):
    res = {}
    for l in lams:
        res[l] = assemble_eval(score, l)
    return res


def report(name, score):
    tA, tB = assemble_eval(score)
    cur = lam_curve(score)
    # cross-fit lam: pick lam on A -> score B, and vice versa
    lA = max(cur, key=lambda l: pq(cur[l][0])); lB = max(cur, key=lambda l: pq(cur[l][1]))
    cross = cur[lB][0] + cur[lA][1]
    sel = np.isin(LV, list("AP"))
    wmse = lambda m: np.sum((W * (score - Y) ** 2)[m]) / W[m].sum()
    print(f"{name:52s} fixed lam .228: A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} | lam-crossfit {pq(cross):.4f} (lamA {lA}, lamB {lB}) | "
          f"wMSE A {wmse(HA & sel):.4f} B {wmse(~HA & sel):.4f} | rho {spearmanr(score[sel], Y[sel])[0]:.3f}", flush=True)
    return pq(tA), pq(tB)


def crossfit(fitfn):
    out = np.zeros(len(Y))
    for tr in (HA, ~HA):
        f = fitfn(tr)
        out[~tr] = f(~tr)
    return out



def pav_fit(x, y, w=None):
    w = np.ones(len(x)) if w is None else w
    o = np.argsort(x, kind="mergesort")
    sums, cnts, xs = [], [], []
    for xi, yi, wi in zip(x[o], y[o], w[o]):
        sums.append(yi * wi); cnts.append(wi); xs.append(xi * wi)
        while len(sums) > 1 and sums[-2] / cnts[-2] > sums[-1] / cnts[-1]:
            s_, c_, x_ = sums.pop(), cnts.pop(), xs.pop()
            sums[-1] += s_; cnts[-1] += c_; xs[-1] += x_
    return np.array(xs) / np.array(cnts), np.array(sums) / np.array(cnts)

def iso_fit(x):
    def fit(tr):
        kx, ky = pav_fit(x[tr], Y[tr], W[tr])
        return lambda te: np.interp(x[te], kx, ky)
    return fit


def logit_fit(cols, inter=False, alpha=3.0):
    X0 = np.nan_to_num(F[:, [ix[c] for c in cols]], nan=-1)
    if inter:
        k = X0.shape[1]
        X0 = np.c_[X0] if k < 2 else np.c_[X0, np.stack([X0[:, i] * X0[:, j] for i in range(k) for j in range(i + 1, k)], 1)]

    def fit(tr):
        X = X0[tr]; y = Y[tr]; w = W[tr]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - mu) / sd]

        def f(b):
            p = 1 / (1 + np.exp(-(Z @ b)))
            l = -(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum() + alpha * (b[1:] ** 2).sum()
            g = Z.T @ (w * (p - y)) + 2 * alpha * np.r_[0, b[1:]]
            return l, g
        b = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return lambda te: 1 / (1 + np.exp(-(np.c_[np.ones(te.sum()), (X0[te] - mu) / sd] @ b)))
    return fit


def gbm_fit(cols, **kw):
    X0 = F[:, [ix[c] for c in cols]]

    def fit(tr):
        m = GBM(**kw).fit(X0[tr], Y[tr], W[tr])
        return lambda te: m.predict(X0[te])
    return fit


if __name__ == "__main__":
    import sys
    report("q*mean_p (v2)", QMP)
    report("q", F[:, ix["q"]])
    report("isotonic(q*mean_p) cross-fit", crossfit(iso_fit(QMP)))
    BASEC = ["q", "mean_p", "peak_u", "s1u_mean", "s1p_p90", "mprob", "log_area", "lvA", "lvP", "lvB", "lvC"]
    SHAPE = ["dtw", "skl", "skl_over_area", "elong", "width", "contrast", "rR", "sharp", "area_over_prior", "p95"]
    CTXF = ["img_ncand", "img_nconf", "img_sumqmp", "img_top5q", "img_top5mp", "img_p0mass", "img_p0frac", "img_p1mass",
            "img_p1frac", "rank_qmp", "conflict_qmp", "n_agree", "max_q_other"]
    SITE = ["disk_r", "disk_med", "disk_iqr", "year"] + [f"site_{c}" for c in "BCLMTU"]
    report("logistic [q,mean_p] + interactions", crossfit(logit_fit(["q", "mean_p"], inter=True)))
    report("logistic BASE + interactions", crossfit(logit_fit(BASEC[:7], inter=True)))
    report("logistic BASE+SHAPE+CTX", crossfit(logit_fit(BASEC + SHAPE + CTXF)))
    for nm, cols in [("GBM BASE", BASEC), ("GBM BASE+SHAPE", BASEC + SHAPE), ("GBM BASE+CTX", BASEC + CTXF),
                     ("GBM BASE+SHAPE+CTX", BASEC + SHAPE + CTXF), ("GBM ALL(+site/disk/year)", BASEC + SHAPE + CTXF + SITE)]:
        report(nm, crossfit(gbm_fit(cols)))
    for kw in (dict(n_trees=150, lr=0.05, depth=2), dict(n_trees=600, lr=0.02, depth=3, min_leaf=50), dict(n_trees=300, lr=0.03, depth=4, min_leaf=40)):
        report(f"GBM BASE+SHAPE+CTX {kw}", crossfit(gbm_fit(BASEC + SHAPE + CTXF, **kw)))
