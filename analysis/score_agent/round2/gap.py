"""Q3: val vs test distributions of model outputs; per-year / per-site val PQ."""
from base import *
from assemble import load_cands
from common import disk_info, disk_mask
import pandas as pd
ctx = load_ctx(load_s1=False)
fin = {s: assemble(ctx.C[s], V2) for s in ctx.stems}
# per stem counts
per = {s: ctx.G.stats(s, fin[s]) for s in ctx.stems}
df = pd.DataFrame([(s, int(s[:4]), s[-2], len(ctx.G.by_stem[s]), *per[s]) for s in ctx.stems], columns=["stem", "year", "site", "nr", "S", "TP", "FP", "FN"])
def pqg(g):
    return g.S.sum() / (g.TP.sum() + .5 * g.FP.sum() + .5 * g.FN.sum())
df["yb"] = pd.cut(df.year, [2010, 2012, 2014, 2016, 2017, 2020, 2022], labels=["11-12", "13-14", "15-16", "17", "18-20", "21-22"])
print("val PQ by year bin:", {k: (round(pqg(g), 3), len(g)) for k, g in df.groupby("yb", observed=True)})
print("val PQ by site:", {k: (round(pqg(g), 3), len(g)) for k, g in df.groupby("site")})
print("val PQ by #readings:", {k: (round(pqg(g), 3), len(g)) for k, g in df.groupby("nr")})
# reweight val to test year distribution (stem weights)
test = sorted(f.rsplit(".", 1)[0] for f in os.listdir(os.path.join(ROOT, "data/MAGFiLO_1.0_Kaggle_2026/test/test_images")))
ty = pd.Series([int(s[:4]) for s in test])
tyb = pd.cut(ty, [2010, 2012, 2014, 2016, 2017, 2020, 2022], labels=["11-12", "13-14", "15-16", "17", "18-20", "21-22"]).value_counts(normalize=True)
vyb = df.yb.value_counts(normalize=True)
w = df.yb.map(lambda b: tyb[b] / vyb[b]).astype(float)
S, TP, FP, FN = [(df[c] * w).sum() for c in ("S", "TP", "FP", "FN")]
print(f"val PQ reweighted to test year mix: {S / (TP + .5 * FP + .5 * FN):.4f} (unweighted {pqg(df):.4f})")
tsb = pd.Series([s[-2] for s in test]).value_counts(normalize=True); vsb = df.site.value_counts(normalize=True)
w2 = df.site.map(lambda b: tsb[b] / vsb[b]).astype(float)
S, TP, FP, FN = [(df[c] * w2).sum() for c in ("S", "TP", "FP", "FN")]
print(f"val PQ reweighted to test site mix: {S / (TP + .5 * FP + .5 * FN):.4f}")
df.to_csv("val_per_stem.csv", index=False)

# model-output distributions val vs test
Ct = load_cands(os.path.join(ROOT, "runs/s2_r34/cands_test_tta_last"), test)
PD = os.path.join(ROOT, "runs/s1_r34_f0/probs_tta")
def stats(stems, C):
    rows = []
    for s in stems:
        cs = C[s]
        f = assemble(cs, V2)
        kept_scores = sorted([c["q"] * c["mean_p"] for c in cs], reverse=True)
        sc = np.array(kept_scores) if kept_scores else np.zeros(0)
        info = disk_info(s)
        pr = np.load(os.path.join(PD, s + ".npy")).astype(np.float32) / 255
        dm = disk_mask(info, (1024, 1024), step=2.0)
        img = np.load(os.path.join(ROOT, "data/cache/img2048", s + ".npy"), mmap_mode="r")
        rows.append(dict(stem=s, year=int(s[:4]), ncand=len(cs), nkept=len(f), kept_area=sum(int(m.sum()) for _, _, m in f),
                         mean_kept_score=float(sc[sc >= .225].mean()) if (sc >= .225).any() else np.nan,
                         n_marg=int(((sc > .15) & (sc < .3)).sum()), mean_q=np.mean([c["q"] for c in cs]) if cs else np.nan,
                         p0mass=float(pr[0][dm].mean()), p1mass=float(pr[1][dm].mean()), p0frac=float((pr[0][dm] > .5).mean()),
                         r=info["r"], med=info["med"], iqr=info["iqr"], img_mean=float(np.asarray(img[::8, ::8]).mean())))
    return pd.DataFrame(rows)
V = stats(ctx.stems, ctx.C); T = stats(test, Ct)
V.to_csv("val_img_stats.csv", index=False); T.to_csv("test_img_stats.csv", index=False)
cols = ["ncand", "nkept", "kept_area", "mean_kept_score", "n_marg", "mean_q", "p0mass", "p1mass", "p0frac", "r", "med", "iqr", "img_mean"]
print(f"{'stat':16s} {'val mean':>10s} {'test mean':>10s} {'val med':>9s} {'test med':>9s}  KS-p")
from scipy.stats import ks_2samp
for c in cols:
    print(f"{c:16s} {V[c].mean():10.3f} {T[c].mean():10.3f} {V[c].median():9.3f} {T[c].median():9.3f}  {ks_2samp(V[c].dropna(), T[c].dropna()).pvalue:.3f}")
