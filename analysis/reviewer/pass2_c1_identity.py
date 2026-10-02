"""Identity round trip: prior -> crop_pad -> resize_to(S) [make_input] -> resize_to(side, area=False) [refine] -> >thr.
Measures IoU vs the original, the best integer shift, and the q-target proxy for a perfect prediction."""
import os, sys, numpy as np, cv2
sys.path.insert(0, "src")
from common import load_meta, load_inst
from s2 import window, crop_pad, resize_to, S

def roundtrip(prior, x0, y0, side, thr=0.5):
    pr = resize_to(crop_pad(prior, x0, y0, side).astype(np.float32), S)
    big = resize_to(pr, side, area=False)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, 2048), min(y0 + side, 2048)
    out = np.zeros((2048, 2048), bool)
    out[ya:yb, xa:xb] = big[ya - y0:yb - y0, xa - x0:xb - x0] > thr
    return out, pr

def iou(a, b):
    u = (a | b).sum(); return (a & b).sum() / max(u, 1)

meta = load_meta()
rng = np.random.default_rng(0)
rows = []
for rid in meta.image_id.sample(40, random_state=0):
    lab = load_inst(rid)
    for l in np.unique(lab)[1:]:
        m = lab == l
        ys, xs = np.nonzero(m)
        box = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        x0, y0, side = window(box)
        out, pr = roundtrip(m, x0, y0, side)
        # best shift
        best = (iou(out, m), 0, 0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = iou(np.roll(np.roll(out, dy, 0), dx, 1), m)
                if v > best[0] + 1e-9: best = (v, dy, dx)
        step = side / S
        tgt = pr > 0.5
        proxy = tgt.sum() * step**2 / m.sum()   # perfect-prediction IoU in s2_loss (hard = tgt)
        rows.append((m.sum(), side, iou(out, m), best[1], best[2], best[0], proxy))
r = np.array(rows, float)
print("n inst", len(r))
for lo, hi in [(0, 400), (400, 1000), (1000, 3000), (3000, 8000), (8000, 1e9)]:
    s = r[(r[:, 0] >= lo) & (r[:, 0] < hi)]
    if len(s) == 0: continue
    print(f"area[{lo},{hi}) n={len(s)} side~{np.median(s[:,1]):.0f} roundtrip IoU mean={s[:,2].mean():.3f} min={s[:,2].min():.3f} "
          f"frac<0.5={np.mean(s[:,2]<0.5):.2f} | bestshift!=0 frac={np.mean((s[:,3]!=0)|(s[:,4]!=0)):.2f} gain={np.mean(s[:,5]-s[:,2]):.4f} "
          f"| q-proxy(perfect pred) mean={s[:,6].mean():.3f} frac<=0.5={np.mean(s[:,6]<=0.5):.2f}")
print("by side:")
for lo, hi in [(128, 256), (256, 512), (512, 768), (768, 1100), (1100, 1600)]:
    s = r[(r[:, 1] >= lo) & (r[:, 1] < hi)]
    if len(s) == 0: continue
    print(f"side[{lo},{hi}) n={len(s)} roundtrip IoU={s[:,2].mean():.3f} frac<0.5={np.mean(s[:,2]<0.5):.2f} q-proxy={s[:,6].mean():.3f} frac<=.5={np.mean(s[:,6]<=0.5):.2f}")
