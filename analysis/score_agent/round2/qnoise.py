"""How noisy is a single-reading q target vs the all-readings mean (what PQ actually rewards)?"""
from base import *
from scipy.stats import spearmanr
ctx = load_ctx(load_s1=False)
P5 = dict(thr=.5, ring=0, rel=0)
per, qs, means = [], [], []
for s in ctx.stems:
    if len(ctx.G.by_stem[s]) < 2:
        continue
    for c in ctx.C[s]:
        if c["level"] not in "AP":
            continue
        m, a, _ = cand_mask(c, P5)
        if a == 0:
            continue
        v = np.array([b if b > .5 else 0.0 for b, _, _ in ctx.obj_ious(s, c["x"], c["y"], m)])
        per.append(v); qs.append(c["q"] * c["mean_p"]); means.append(v.mean())
qs = np.array(qs); means = np.array(means)
tot = np.var(np.concatenate(per))
within = np.mean([np.var(v) for v in per])
print(f"multi-reading A/P cands {len(per)}: per-reading target variance {tot:.4f}, within-candidate (annotator) variance {within:.4f} -> {within / tot:.0%} of target variance is annotator noise")
one = np.array([v[np.random.default_rng(i).integers(len(v))] for i, v in enumerate(per)])
print(f"spearman(score, single-reading y) {spearmanr(qs, one)[0]:.3f} vs spearman(score, mean-over-readings y) {spearmanr(qs, means)[0]:.3f}")
print(f"share of cands where readings disagree on match (some >.5, some not): {np.mean([(v > 0).any() and (v == 0).any() for v in per]):.2f}")
