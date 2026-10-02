"""Q4: no-fit score products (S1 head blends etc.) + lam sensitivity, split-half."""
from rs3 import *
props_mu = {}
for s in ctx.stems:
    pr = props_of(s)
    for c in ctx.C[s]:
        d = pr[c["idx"]]
        c["mean_u"] = d["mean_u"]; c["peak_p"] = d["peak_p"]
MU = np.array([c["mean_u"] for c in CANDS]); MP = F[:, ix["mean_p"]]; Q = F[:, ix["q"]]
S1P = np.nan_to_num(F[:, ix["s1p_mean"]], nan=0); S1U = np.nan_to_num(F[:, ix["s1u_mean"]], nan=0); PU = F[:, ix["peak_u"]]
print("lam sensitivity for q*mean_p:")
cur = lam_curve(QMP, lams=(0.15, 0.17, 0.19, 0.2, 0.21, 0.22, 0.228, 0.24, 0.25, 0.26, 0.28, 0.3))
for l, (tA, tB) in cur.items():
    print(f"  lam {l:.3f}: A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} {cnt(tA + tB)}")
cands = {
    "q*mean_u (head1 in proposal)": Q * MU,
    "q*(mean_p+mean_u)/2": Q * (MP + MU) / 2,
    "q*sqrt(mean_p*mean_u)": Q * np.sqrt(MP * MU),
    "q*s1p_mean (head0 in S2 mask)": Q * S1P,
    "q*s1u_mean (head1 in S2 mask)": Q * S1U,
    "q*(mean_p+s1p_mean)/2": Q * (MP + S1P) / 2,
    "q*mean_p^2": Q * MP ** 2,
    "sqrt(q)*mean_p": np.sqrt(Q) * MP,
    "q*mean_p*peak_u": QMP * PU,
}
for nm, sc in cands.items():
    report(nm + " [raw]", sc)
    report(nm + " [isotonic cross-fit]", crossfit(iso_fit(sc)))
