"""Near-miss part 2: DT-based width, per-annotator spread, cross-reading status of near-miss preds."""
from base import *
from skimage.morphology import skeletonize

ctx = load_ctx(load_s1=False)
recs = pickle.load(open("near_recs.pkl", "rb"))
fin = {s: assemble(ctx.C[s], V2) for s in ctx.stems}


def dt_width(m):
    if m.sum() < 3:
        return np.nan
    d = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    sk = skeletonize(m)
    return float(np.median(2 * d[sk] - 1)) if sk.any() else np.nan


out = {"TP": [], "FPnear": [], "FNnear": []}
for kind, half, s, ri, j, k, st in recs:
    readings = ctx.G.by_stem[s]
    lab, areas = readings[ri]
    g = lab == j
    x, y, m = fin[s][k]
    ys, xs = np.nonzero(g)
    pad = 8
    x0, y0 = max(min(x, xs.min()) - pad, 0), max(min(y, ys.min()) - pad, 0)
    x1 = min(max(x + m.shape[1], xs.max() + 1) + pad, 2048); y1 = min(max(y + m.shape[0], ys.max() + 1) + pad, 2048)
    P = np.zeros((y1 - y0, x1 - x0), bool); P[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] = m
    G = g[y0:y1, x0:x1]
    wP, wG = dt_width(P), dt_width(G)
    # other readings: best-IoU instance of the same filament, and pred status there
    other_iou, pred_tp_other, other_w = [], [], []
    for r2, (lab2, areas2) in enumerate(readings):
        if r2 == ri:
            continue
        v = np.bincount(lab2[g], minlength=256).astype(float); v[0] = 0
        iou = v / np.maximum(g.sum() + areas2 - v, 1)
        l2 = int(iou.argmax())
        other_iou.append(iou[l2])
        if iou[l2] > 0:
            other_w.append(dt_width(lab2[y0:y1, x0:x1] == l2))
        vp = np.bincount(lab2[y:y + m.shape[0], x:x + m.shape[1]][m], minlength=256).astype(float); vp[0] = 0
        ip = vp / np.maximum(m.sum() + areas2 - vp, 1)
        pred_tp_other.append(ip.max() > 0.5)
    out[kind].append(dict(wP=wP, wG=wG, nread=len(readings), oiou=other_iou, ptp=pred_tp_other, ow=other_w, iou=st["iou"]))

for kind, R in out.items():
    wP = np.array([r["wP"] for r in R]); wG = np.array([r["wG"] for r in R])
    ok = np.isfinite(wP) & np.isfinite(wG)
    print(f"\n== {kind} n={len(R)}: DT width median pred {np.nanmedian(wP):.1f} GT {np.nanmedian(wG):.1f}; ratio median {np.median(wP[ok] / wG[ok]):.2f} "
          f"[q25 {np.quantile(wP[ok] / wG[ok], .25):.2f} q75 {np.quantile(wP[ok] / wG[ok], .75):.2f}]")
    multi = [r for r in R if r["nread"] > 1]
    oi = np.concatenate([r["oiou"] for r in multi]) if multi else np.zeros(0)
    pt = np.concatenate([r["ptp"] for r in multi]) if multi else np.zeros(0)
    print(f"  multi-reading cases {len(multi)}/{len(R)}; same filament in other readings: IoU with this GT median {np.median(oi):.2f}, "
          f"share absent(IoU=0) {np.mean(oi == 0):.2f}, share IoU>.5 {np.mean(oi > .5):.2f}; pred is TP in the other reading {np.mean(pt):.2f}")
    ow = [ (r["wG"], w) for r in multi for w in r["ow"] if np.isfinite(w) and np.isfinite(r["wG"])]
    if ow:
        a = np.array(ow)
        print(f"  annotator width spread: |log(wG/wG_other)| median {np.median(np.abs(np.log(a[:, 0] / a[:, 1]))):.2f}  (ratio iqr {np.quantile(a[:, 0] / a[:, 1], .25):.2f}-{np.quantile(a[:, 0] / a[:, 1], .75):.2f})")
# per-annotator widths over all GT (val)
meta = load_meta() if False else None
