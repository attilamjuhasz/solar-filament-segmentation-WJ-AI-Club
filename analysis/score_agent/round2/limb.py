from rs3 import *
rR = F[:, ix["rR"]]; la = F[:, ix["log_area"]]
sel = np.isin(LV, list("AP")) & np.isfinite(rR)
print("calibration y/score by r/R (A/P cands with score >= .15):")
for lo, hi in ((0, .5), (.5, .7), (.7, .85), (.85, 1.01)):
    for h, nm in ((HA, "A"), (~HA, "B")):
        k = sel & h & (rR >= lo) & (rR < hi) & (QMP >= .15)
        print(f"  r/R [{lo},{hi}) half {nm}: n={k.sum():4d} mean score {QMP[k].mean():.3f} mean y {Y[k].mean():.3f} ratio {Y[k].mean() / QMP[k].mean():.2f}")
for f_ in (0.8, 0.9):
    z = QMP * np.where(rR > 0.85, f_, 1.0)
    report(f"score * {f_} if r/R > .85", np.nan_to_num(z))
