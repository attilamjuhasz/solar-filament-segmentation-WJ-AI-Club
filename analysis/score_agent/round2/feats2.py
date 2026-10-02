"""Extended per-candidate features (candidate order = ctx.stems x ctx.C[s]) incl. image-level context."""
from base import *
from common import disk_info, disk_mask
from skimage.morphology import skeletonize
import sys

CD = sys.argv[1] if len(sys.argv) > 1 else None   # optional candidate dir (e.g. test) -> features without y
OUT = sys.argv[2] if len(sys.argv) > 2 else "feats2_val.pkl"
SITES = "BCLMTU"

if CD is None:
    ctx = load_ctx(load_s1=False)
    stems, C = ctx.stems, ctx.C
else:
    from assemble import load_cands
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, CD)) if f.endswith(".pkl"))
    C = load_cands(os.path.join(ROOT, CD), stems)
base = pickle.load(open("feats_ep8.pkl", "rb")) if CD is None else None
PDIR = os.path.join(ROOT, "runs/s1_r34_f0/probs_tta")
P5 = dict(thr=0.5, ring=0.0, rel=0.0)
names = ["img_ncand", "img_nconf", "img_sumqmp", "img_top5q", "img_top5mp", "img_p0mass", "img_p0frac", "img_p1mass",
         "img_p1frac", "disk_r", "disk_med", "disk_iqr", "year"] + [f"site_{c}" for c in SITES] + \
        ["rank_qmp", "sharp", "dtw", "skl", "skl_over_area", "conflict_qmp", "area_over_prior", "qmp", "nread_dummy"]
rows, meta = [], []
for s in stems:
    info = disk_info(s)
    pr = np.load(os.path.join(PDIR, s + ".npy")).astype(np.float32) / 255.0
    dm = disk_mask(info, (1024, 1024), step=2.0)
    p0, p1 = pr[0][dm], pr[1][dm]
    cs = C[s]
    qmp = np.array([c["q"] * c["mean_p"] for c in cs]) if cs else np.zeros(0)
    q = np.array([c["q"] for c in cs]) if cs else np.zeros(0)
    mp = np.array([c["mean_p"] for c in cs]) if cs else np.zeros(0)
    order = np.argsort(-qmp)
    rank = np.empty(len(cs)); rank[order] = np.arange(len(cs))
    img = [len(cs), int((qmp > 0.225).sum()), qmp.sum(), np.sort(q)[::-1][:5].mean() if len(q) else 0,
           np.sort(mp)[::-1][:5].mean() if len(mp) else 0, p0.mean(), (p0 > 0.5).mean(), p1.mean(), (p1 > 0.5).mean(),
           info["r"], info["med"], info["iqr"], int(s[:4])] + [float(s[-2] == ch) for ch in SITES]
    masks = [cand_mask(c, P5) for c in cs]
    props = props_of(s)
    for i, c in enumerate(cs):
        m, area, mprob = masks[i]
        nz = c["soft"][c["soft"] > 127]
        sharp = float((nz > 204).mean()) if nz.size else 0.0
        if area >= 3:
            d = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
            sk = skeletonize(m)
            dtw = float(np.median(2 * d[sk] - 1)) if sk.any() else 0.0
            skl = float(sk.sum())
        else:
            dtw, skl = 0.0, 0.0
        # conflict: max qmp among higher-ranked candidates overlapping >= 20% of this mask
        conf = 0.0
        x, y = c["x"], c["y"]; h, w = m.shape
        for j in order[:int(rank[i])]:
            d_ = cs[j]; dm_ = masks[j][0]
            if masks[j][1] == 0 or area == 0:
                continue
            x0, y0 = max(x, d_["x"]), max(y, d_["y"])
            x1, y1 = min(x + w, d_["x"] + dm_.shape[1]), min(y + h, d_["y"] + dm_.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            io = int((m[y0 - y:y1 - y, x0 - x:x1 - x] & dm_[y0 - d_["y"]:y1 - d_["y"], x0 - d_["x"]:x1 - d_["x"]]).sum())
            if io >= 0.2 * area:
                conf = max(conf, qmp[j])
        pa = props[c["idx"]]["area"] if c["idx"] < len(props) else area
        rows.append(img + [rank[i], sharp, dtw, skl, skl / max(area, 1), conf, area / max(pa, 1), qmp[i], 0.0])
        meta.append((s, i))
F2 = np.array(rows, np.float64)
out = dict(names=names, F=F2, meta=meta)
if base is not None:
    assert base["F"].shape[0] == F2.shape[0]
    out = dict(names=base["names"] + names, F=np.c_[base["F"], F2], meta=meta)
pickle.dump(out, open(OUT, "wb"))
print(out["F"].shape)
