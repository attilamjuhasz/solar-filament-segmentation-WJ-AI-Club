"""Q2a: what separates scoring misses (good cands left out) from kept FPs. Per-image residual structure."""
from base import *
from scipy.stats import spearmanr
ctx = load_ctx(load_s1=False)
D = pickle.load(open("feats2_val.pkl", "rb"))
N, F = D["names"], D["F"]
ix = {n: i for i, n in enumerate(N)}
QMP = lambda c, m, a, mp: c["q"] * c["mean_p"]
rows = []
k = 0
for s in ctx.stems:
    fin, prov = assemble_prov(ctx.C[s], V2, score_fn=QMP)
    sel_ids = {id(c) for c, _ in prov}
    for c in ctx.C[s]:
        m, a, _ = cand_mask(c, dict(thr=.5, ring=0, rel=0))
        # overlap with any selected final
        ov = 0.0
        for (x, y, fm) in fin:
            x0, y0 = max(x, c["x"]), max(y, c["y"])
            x1, y1 = min(x + fm.shape[1], c["x"] + m.shape[1]), min(y + fm.shape[0], c["y"] + m.shape[0])
            if x1 > x0 and y1 > y0 and a:
                io = (fm[y0 - y:y1 - y, x0 - x:x1 - x] & m[y0 - c["y"]:y1 - c["y"], x0 - c["x"]:x1 - c["x"]]).sum()
                ov = max(ov, io / a)
        rows.append((s, id(c) in sel_ids, c["yv"], ov, len(ctx.G.by_stem[s]), c["level"]))
        k += 1
sel = np.array([r[1] for r in rows]); yv = np.array([r[2] for r in rows]); ov = np.array([r[3] for r in rows])
nr = np.array([r[4] for r in rows]); lv = np.array([r[5] for r in rows])
lam = 0.228
missed = (~sel) & (yv > lam) & (ov < 0.2) & np.isin(lv, list("AP"))
keptbad = sel & (yv < 0.05)
keptgood = sel & (yv > lam)
rejbad = (~sel) & (yv < 0.05) & (ov < 0.2) & np.isin(lv, list("AP"))
print(f"cands {len(rows)}; missed-good (unselected, y>{lam}, not covered, A/P) {missed.sum()}  kept-bad (sel, y<.05) {keptbad.sum()}  kept-good {keptgood.sum()}  rejected-bad {rejbad.sum()}")
print(f"value at stake: missed-good sum(y - lam) weighted by readings = {((yv - lam) * nr)[missed].sum():.1f};  kept-bad sum(lam - y)*nr = {((lam - yv) * nr)[keptbad].sum():.1f}")
cols = ["q", "mean_p", "peak_u", "mprob", "log_area", "s1u_mean", "s1p_p90", "n_agree", "rR", "elong", "width", "contrast",
        "img_nconf", "img_p0mass", "img_top5q", "disk_med", "year", "rank_qmp", "sharp", "dtw", "skl", "conflict_qmp", "area_over_prior", "qmp"]
print(f"{'feature':16s} {'missed-good':>12s} {'kept-bad':>10s} {'kept-good':>10s} {'rej-bad':>10s}")
for cn in cols:
    v = F[:, ix[cn]]
    print(f"{cn:16s} {np.nanmedian(v[missed]):12.3f} {np.nanmedian(v[keptbad]):10.3f} {np.nanmedian(v[keptgood]):10.3f} {np.nanmedian(v[rejbad]):10.3f}")
print("level mix missed-good:", {L: int((missed & (lv == L)).sum()) for L in "AP"}, " kept-bad:", {L: int((keptbad & (lv == L)).sum()) for L in "AP"})
print("readings/img missed-good:", np.bincount(nr[missed], minlength=4)[1:], " kept-bad:", np.bincount(nr[keptbad], minlength=4)[1:], " all cands:", np.bincount(nr, minlength=4)[1:])
# per-image calibration: residual y - qmp averaged per image among cands with qmp>.1; ICC-ish check
qmp = F[:, ix["qmp"]]
stem_arr = np.array([r[0] for r in rows])
res = yv - qmp
mask = (qmp > 0.1) & np.isin(lv, list("AP"))
st = sorted(set(stem_arr[mask]))
mres = np.array([res[mask & (stem_arr == s)].mean() for s in st])
cnts = np.array([(mask & (stem_arr == s)).sum() for s in st])
# permutation null: shuffle residuals across images
rng = np.random.default_rng(0)
null = []
r_m = res[mask]; s_m = stem_arr[mask]
for _ in range(300):
    pr = rng.permutation(r_m)
    null.append(np.var([pr[s_m == s].mean() for s in st]))
print(f"per-image mean residual (y - qmp) variance {np.var(mres):.4f} vs permutation null {np.mean(null):.4f} (p95 {np.quantile(null, .95):.4f})")
# correlations of per-image residual with image features
for cn in ["img_nconf", "img_p0mass", "img_p0frac", "img_p1mass", "img_top5q", "img_top5mp", "disk_med", "disk_iqr", "year", "img_ncand"]:
    v = np.array([F[mask & (stem_arr == s), ix[cn]][0] for s in st])
    print(f"  spearman(img residual, {cn:11s}) = {spearmanr(v, mres)[0]:+.3f}")
nrs = np.array([len(ctx.G.by_stem[s]) for s in st])
print("  mean residual by #readings:", {k: round(float(mres[nrs == k].mean()), 3) for k in (1, 2, 3)})
