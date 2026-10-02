"""Minimal rescorer sets, a no-fit product score, and S1 instances gated by the rescored S2 score."""
import sys
import numpy as np
import rescore as R
from q3 import M1, M3, P1, G1, split_report

R.SETS.update({
    "q,mean_p": ["q", "mean_p"],
    "q,mean_p,peak_u": ["q", "mean_p", "peak_u"],
    "q,mean_p,peak_u,s1u_mean": ["q", "mean_p", "peak_u", "s1u_mean"],
    "q,logit_q,mean_p,peak_u,s1u_mean,lvA,lvP": ["q", "logit_q", "mean_p", "peak_u", "s1u_mean", "lvA", "lvP"],
})
for nm in ["q,mean_p", "q,mean_p,peak_u", "q,mean_p,peak_u,s1u_mean", "q,logit_q,mean_p,peak_u,s1u_mean,lvA,lvP"]:
    split_report(f"M6 rescorer [{nm}]", M1, R.make_fit(R.SETS[nm]), P1, G1)
G1p = dict(G1, lam=[0.1, 0.13, 0.16, 0.2, 0.23, 0.26, 0.3, 0.35, 0.4])
split_report("M7 no-fit score q*mean_p", M1, lambda st: (lambda c: c["q"] * c["mean_p"]), dict(P1, lam=0.2), G1p)
split_report("M3r S1-only inst gated by rescored [q,mean_p,peak_u,s1u_mean] S2 cand", M3,
             R.make_fit(R.SETS["q,mean_p,peak_u,s1u_mean"]),
             dict(s1="best", levels="AP", match_iou=0.5, d0=0.0, lam=0.22, a_min=0, own_frac=0.0),
             dict(s1=["best", "loose"], lam=[0.14, 0.17, 0.2, 0.22, 0.25, 0.28, 0.32], match_iou=[0.3, 0.5],
                  levels=["A", "AP"], a_min=[0, 100, 300, 400]))
