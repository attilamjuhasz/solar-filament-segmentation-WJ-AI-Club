"""Multi-frame (temporal TTA) evaluation on fold-0 val.
Fuse the target's S1 probs with S1 probs of GONG-archive frames taken minutes apart (same site and other sites),
warped into the target frame. Measure (1) S1-only postprocess PQ, (2) v2 assembly with mean_p recomputed from fused probs."""
import os, sys, glob, json, pickle
import numpy as np, cv2
from pycocotools import mask as mu
HERE = os.path.dirname(os.path.abspath(__file__))
from decode import *   # C, G, R, NR, A, B, stems, stats_reading, assemble_sel, pq, cnt
sys.path.insert(0, HERE)
from solgeo import warp_maps, stem_ts
from s1 import postprocess_s1
from common import disk_info
cv2.setNumThreads(2)
S1BEST = dict(t_hi=0.6, t_lo=0.45, gap=8, min_area=400, head=0)
PR = os.path.join(ROOT, "runs/s1_r34_f0")


def load_p(d, s):
    a = np.load(os.path.join(PR, d, s + ".npy")).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def nbr_probs(s):
    """-> list of (kind, warped (2,1024,1024) float probs, valid mask)"""
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "gong", s, "*.npz"))):
        st = os.path.basename(f)[:-4]
        z = np.load(f)
        p = z["p"].astype(np.float32) / 255.0
        info = json.loads(str(z["info"]))
        mx, my, valid, dt = warp_maps(disk_info(s), stem_ts(s), info, stem_ts(st), shape=(1024, 1024), step=2.0)
        w = np.stack([cv2.remap(p[c], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for c in (0, 1)])
        out.append(("same" if st[-2] == s[-2] else "other", w, valid))
    return out


def fuse(base, nb, w, kinds):
    L = [(p, v) for k, p, v in nb if k in kinds]
    if not L:
        return base
    num = sum(p * v for p, v in L); den = sum(v.astype(np.float32) for p, v in L)
    mean_nb = np.where(den > 0, num / np.maximum(den, 1e-6), base)
    return (1 - w) * base + w * mean_nb


def up(p):
    return cv2.resize(p.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)


DONE = [s for s in stems if glob.glob(os.path.join(HERE, "gong", s, "*.jpg")) and
        all(os.path.exists(f[:-4] + ".npz") for f in glob.glob(os.path.join(HERE, "gong", s, "*.jpg")))]
if os.environ.get("SUBSET"):
    stems = DONE
print("evaluating on", len(stems), "stems", flush=True)
PROPS = {s: pickle.load(open(os.path.join(PR, "props_tta", s + ".pkl"), "rb")) for s in stems}
# sanity: candidate mean_p equals its proposal's mean_p
bad = sum(abs(PROPS[s][c["idx"]]["mean_p"] - c["mean_p"]) > 1e-6 for s in stems for c in C[s])
print("mean_p/props mismatch:", bad, flush=True)

VARIANTS = {
    "TTA only (baseline)": ("tta", 0.0, ()),
    "TTA + same-site nbrs (w=.3)": ("tta", 0.3, ("same",)),
    "TTA + all nbrs (w=.3)": ("tta", 0.3, ("same", "other")),
    "TTA + all nbrs (w=.5)": ("tta", 0.5, ("same", "other")),
}
acc_s1 = {k: [np.zeros(4), np.zeros(4)] for k in VARIANTS}
acc_v2 = {k: [np.zeros(4), np.zeros(4)] for k in VARIANTS}
have = 0
for i, s in enumerate(stems):
    nb = nbr_probs(s)
    have += bool(nb)
    base = {"tta": load_p("probs_tta", s), "plain": load_p("probs_plain", s)}
    same = [p for k, p, v in nb if k == "same"]
    base["nbr1"] = same[0] if same else base["plain"]
    for name, (b, w, kinds) in VARIANTS.items():
        fp = fuse(base[b], nb, w, kinds)
        h = 0 if s in A else 1
        # S1-only
        inst = postprocess_s1(fp.copy(), s, **S1BEST)
        acc_s1[name][h] += sum(stats_reading(s, ri, inst) for ri in range(NR[s]))
        # v2 with fused mean_p (q from S2 unchanged; masks unchanged)
        P2 = up(fp[0])
        items = []
        for k, c in enumerate(C[s]):
            if c["level"] not in "AP":
                continue
            d = PROPS[s][c["idx"]]
            x, y, w_, h_ = d["box"]
            m = mu.decode(d["rle"])[y:y + h_, x:x + w_].astype(bool)
            mp = float(P2[y:y + h_, x:x + w_][m].mean())
            sc = c["q"] * mp
            if sc >= 0.225:
                items.append((sc, k, 0.5))
        fin = assemble_sel(s, items)
        acc_v2[name][h] += sum(stats_reading(s, ri, fin) for ri in range(NR[s]))
    if i % 20 == 0:
        print(i, s, len(nb), flush=True)
print(f"stems with archive neighbours: {have}/{len(stems)}")
for tag, acc in (("S1-only postprocess", acc_s1), ("v2 (q x fused mean_p)", acc_v2)):
    print("===", tag)
    for name, (tA, tB) in acc.items():
        print(f"  {name:45s} A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} {cnt(tA + tB)}")
