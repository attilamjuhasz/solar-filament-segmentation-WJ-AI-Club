"""Extrapolate the reachable keep/reject ceiling: on 3-reading val stems, score each held-out reading with
(model), (1 other annotator), (2 other annotators), and blends. PQ(n others) ~ PQinf - c/n."""
from decode import *

M3 = [s for s in stems if NR[s] == 3]
M2 = [s for s in stems if NR[s] == 2]


def loo_one(which):
    def mk(s, ri):
        others = [i for i in range(NR[s]) if i != ri]
        o = others[which]
        return lambda k, t: float(yv(R[s][k], t)[o])
    return mk


def blend_n(alpha, n_use):
    def mk(s, ri):
        others = [i for i in range(NR[s]) if i != ri][:n_use]
        def f(k, t):
            v = yv(R[s][k], t)[others].mean()
            return alpha * v + (1 - alpha) * R[s][k]["q"] * R[s][k]["mean_p"]
        return f
    return mk


def run_avg(name, mks, lam, stems_):
    """average counts over several score makers (e.g. LOO with other #0 and other #1)."""
    t = np.zeros(4)
    for mk in mks:
        for s in stems_:
            for ri in range(NR[s]):
                t += stats_reading(s, ri, decode(s, mk(s, ri), lam))
    print(f"{name:66s} lam={lam:.3f} PQ={pq(t):.4f} {cnt(t)}", flush=True)
    return pq(t)


if __name__ == "__main__":
    for lam in (0.2, 0.228, 0.26):
        print(f"--- 3-reading stems ({len(M3)} stems, 93 readings), lam {lam}")
        run_avg("model q*mean_p", [model_score], lam, M3)
        p1 = run_avg("1 other annotator (avg over which)", [loo_one(0), loo_one(1)], lam, M3)
        p2 = run_avg("2 other annotators (mean y)", [loo_score], lam, M3)
        print(f"   extrapolated PQ(inf annotators) = 2*P2 - P1 = {2 * p2 - p1:.4f}")
        for a in (0.2, 0.35, 0.5):
            b1 = run_avg(f"blend {a}*1-other + model", [blend_n(a, 1)], lam, M3)
            b2 = run_avg(f"blend {a}*2-others + model", [blend_n(a, 2)], lam, M3)
    print(f"--- 2-reading stems ({len(M2)})")
    run_avg("model", [model_score], 0.228, M2)
    run_avg("1 other annotator", [loo_score], 0.228, M2)
    for a in (0.2, 0.35):
        run_avg(f"blend {a}*1-other + model", [blend_n(a, 1)], 0.228, M2)
