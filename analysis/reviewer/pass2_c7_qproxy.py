"""s2_loss q target (S-scale counts * step^2 vs native gt_area) vs the native IoU the assembled mask would get."""
import sys, numpy as np, cv2
sys.path.insert(0, "src")
from common import load_meta, load_inst
from s2 import window, crop_pad, resize_to, S
meta = load_meta()
res = []
for rid in meta.image_id.sample(25, random_state=1):
    lab = load_inst(rid)
    for l in np.unique(lab)[1:]:
        g = lab == l
        ys, xs = np.nonzero(g); box = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        x0, y0, side = window(box); step = side / S
        for kind in ("shift2", "erode2", "dilate2"):
            gu = g.astype(np.uint8)
            if kind == "shift2": pred = np.roll(np.roll(gu, 2, 0), 2, 1)
            elif kind == "erode2": pred = cv2.erode(gu, np.ones((5, 5), np.uint8))
            else: pred = cv2.dilate(gu, np.ones((5, 5), np.uint8))
            if pred.sum() == 0: continue
            # "model output" at S = area-resized pred (what a perfectly calibrated net would emit)
            pS = resize_to(crop_pad(pred, x0, y0, side).astype(np.float32), S)
            yS = resize_to(crop_pad(g, x0, y0, side).astype(np.float32), S)
            hard, tgt = pS > 0.5, yS > 0.5
            it = (hard & tgt).sum() * step**2; pa = hard.sum() * step**2
            proxy = it / max(pa + g.sum() - it, 1)
            # native IoU of back-projected hard mask (refine path)
            big = resize_to(pS, side, area=False) > 0.5
            full = np.zeros_like(g); xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, 2048), min(y0 + side, 2048)
            full[ya:yb, xa:xb] = big[ya - y0:yb - y0, xa - x0:xb - x0]
            nat = (full & g).sum() / (full | g).sum()
            res.append((kind, side, proxy, nat))
for kind in ("shift2", "erode2", "dilate2"):
    r = np.array([(s, p, n) for k, s, p, n in res if k == kind])
    for lo, hi in [(128, 200), (200, 400), (400, 2000)]:
        q = r[(r[:, 0] >= lo) & (r[:, 0] < hi)]
        if len(q):
            print(f"{kind} side[{lo},{hi}) n={len(q)} proxy={q[:,1].mean():.3f} native={q[:,2].mean():.3f} "
                  f"mean|diff|={np.abs(q[:,1]-q[:,2]).mean():.3f} disagree@0.5={np.mean((q[:,1]>.5)!=(q[:,2]>.5)):.3f}")
