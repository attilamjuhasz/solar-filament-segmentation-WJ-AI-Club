"""Per-image solar disk geometry and radial intensity profile -> data/cache/disk.json.

Each entry: {cx, cy, r} in native 2048 px, and `prof`: median on-disk intensity in 256 bins of r/R
over [0, 1.0], used to flatten limb darkening.
"""
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE  # noqa: E402

NBINS = 256


def fit_disk(img):
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(th)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    # Least-squares circle on the boundary is robust to the filled interior; area radius as a check.
    pts = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(c + cx ** 2 + cy ** 2)
    s = img.shape[0] / 512
    return cx * s + (s - 1) / 2, cy * s + (s - 1) / 2, r * s, np.sqrt(filled.sum() / np.pi) * s


def _fit_circle(pts):
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cy, np.sqrt(c + cx ** 2 + cy ** 2)


def refine_limb(img, cx, cy, r=905.0, rmin=700, rmax=1100, iters=2, r_lo=885.0, r_hi=925.0):
    """Limb = outermost sharp outward intensity drop whose inner side is still disk-bright.

    Some sites show a gray halo ring outside the limb; the halo->sky drop has a dim inner side, so the
    brightness test rejects it even when the starting center is biased toward the halo.
    """
    sm = cv2.GaussianBlur(img, (0, 0), 2).astype(np.float32)
    h, w = img.shape
    c0 = sm[h // 2 - 300:h // 2 + 300, w // 2 - 300:w // 2 + 300]
    disk_med = np.median(c0)
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rr = np.arange(rmin, rmax, 1.0)
    for it in range(iters):
        xs = cx + np.cos(th)[:, None] * rr[None]
        ys = cy + np.sin(th)[:, None] * rr[None]
        prof = cv2.remap(sm, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                         borderValue=0)
        grad = np.zeros_like(prof)
        grad[:, 2:-2] = prof[:, 4:] - prof[:, :-4]
        inner = np.roll(prof, 8, axis=1)
        score = np.where(inner > 0.6 * disk_med, grad, 0)
        score[:, :8] = 0
        # up to 3 local-minimum drop candidates per ray; RANSAC picks the consistent circle
        pts = []
        for i in range(len(th)):
            s = score[i]
            idx = np.argsort(s)[:12]
            chosen = []
            for j in idx:
                if s[j] > -12 or len(chosen) == 3:
                    break
                if all(abs(j - c) > 10 for c in chosen):
                    chosen.append(j)
            pts += [(xs[i, j], ys[i, j]) for j in chosen]
        pts = np.asarray(pts)
        if len(pts) < 20:
            break
        rng = np.random.default_rng(0)
        best, best_in = None, -1
        for _ in range(1500):
            p = pts[rng.choice(len(pts), 3, replace=False)]
            try:
                c = _fit_circle(p)
            except np.linalg.LinAlgError:
                continue
            if not (r_lo <= c[2] <= r_hi):
                continue
            n_in = (np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2]) < 2.5).sum()
            if n_in > best_in:
                best, best_in = c, n_in
        if best is None:
            break
        inl = np.abs(np.hypot(pts[:, 0] - best[0], pts[:, 1] - best[1]) - best[2]) < 2.5
        cx, cy, r = _fit_circle(pts[inl])
    return cx, cy, r


def radial_profile(img, cx, cy, r):
    small = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
    s = img.shape[0] / 1024
    yy, xx = np.mgrid[:1024, :1024]
    rr = np.hypot(xx * s + (s - 1) / 2 - cx, yy * s + (s - 1) / 2 - cy) / r
    inside = rr < 1.0
    bins = np.minimum((rr[inside] * NBINS).astype(int), NBINS - 1)
    vals = small[inside]
    order = np.argsort(bins, kind="stable")
    bins, vals = bins[order], vals[order]
    edges = np.searchsorted(bins, np.arange(NBINS + 1))
    prof = np.array([np.median(vals[edges[i]:edges[i + 1]]) if edges[i + 1] > edges[i] else np.nan
                     for i in range(NBINS)])
    # fill empty bins (tiny r near center)
    idx = np.arange(NBINS)
    ok = ~np.isnan(prof)
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    return np.interp(idx, idx[ok], prof[ok]).tolist(), float(q50), float(q75 - q25)


def main():
    out = {}
    d = os.path.join(CACHE, "img2048")
    for f in tqdm(sorted(os.listdir(d))):
        img = np.load(os.path.join(d, f))
        cx, cy, r0, r_area = fit_disk(img)  # coarse center only; Otsu radius is biased by halos
        cx, cy, r = refine_limb(img, cx, cy, r=float(np.clip(r0, 885, 925)))
        prof, med, iqr = radial_profile(img, cx, cy, r)
        out[f[:-4]] = dict(cx=float(cx), cy=float(cy), r=float(r), r_area=float(r_area),
                           med=med, iqr=max(iqr, 1.0), prof=[round(v, 2) for v in prof])
    json.dump(out, open(os.path.join(CACHE, "disk.json"), "w"))
    r = np.array([v["r"] for v in out.values()])
    ra = np.array([v["r_area"] for v in out.values()])
    print("r pct", np.percentile(r, [0, 1, 50, 99, 100]).round(1), "| Otsu-area radius off by >15px on",
          int((np.abs(r - ra) > 15).sum()), "images (halo)")
    bad = [k for k, v in out.items() if not 890 < v["r"] < 915]
    print("suspicious", len(bad), bad[:10])


if __name__ == "__main__":
    main()
