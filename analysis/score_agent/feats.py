"""Per-candidate features for a stacked rescorer (same order as q1_rows.pkl)."""
import os
import pickle

import cv2
import numpy as np

from ana import ROOT, TAG, Ctx, cand_mask
from common import disk_info

ctx = Ctx()
M5 = dict(thr=0.5, ring=0.0, rel=0.0)
NAMES = ["q", "logit_q", "peak_u", "mean_p", "mprob", "p95", "log_area", "lvA", "lvP", "lvB", "lvC",
         "s1p_mean", "s1p_p90", "s1u_mean", "n_agree", "max_q_other", "rR", "elong", "width", "contrast",
         "n_cands_stem"]
F = []
for s in ctx.stems:
    up0 = cv2.resize(ctx.probs[s][0], (2048, 2048), interpolation=cv2.INTER_LINEAR)
    up1 = cv2.resize(ctx.probs[s][1], (2048, 2048), interpolation=cv2.INTER_LINEAR)
    img = np.load(os.path.join(ROOT, "data/cache/img2048", s + ".npy"), mmap_mode="r")
    info = disk_info(s)
    cs = ctx.C[s]
    masks = [cand_mask(c, M5) for c in cs]
    for i, c in enumerate(cs):
        m, area, mprob = masks[i]
        x, y = c["x"], c["y"]
        h, w = c["soft"].shape
        if area == 0:
            F.append([c["q"], 0] + [np.nan] * (len(NAMES) - 2))
            continue
        p0 = up0[y:y + h, x:x + w][m] / 255.0
        p1 = up1[y:y + h, x:x + w][m] / 255.0
        # agreement with other proposals (mask IoU >= .5)
        n_ag, mq = 0, 0.0
        for j, d in enumerate(cs):
            if j == i or masks[j][1] == 0:
                continue
            dm = masks[j][0]
            x0, y0 = max(x, d["x"]), max(y, d["y"])
            x1, y1 = min(x + w, d["x"] + dm.shape[1]), min(y + h, d["y"] + dm.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            io = int((m[y0 - y:y1 - y, x0 - x:x1 - x] & dm[y0 - d["y"]:y1 - d["y"], x0 - d["x"]:x1 - d["x"]]).sum())
            if io / max(area + masks[j][1] - io, 1) >= 0.5:
                n_ag += 1
                mq = max(mq, d["q"])
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean() + x, ys.mean() + y
        rR = float(np.hypot(cx - info["cx"], cy - info["cy"]) / info["r"])
        cov = np.cov(np.stack([xs, ys]).astype(np.float64)) if len(xs) > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(max(ev[0], 1e-6) / max(ev[1], 1e-6)))
        major = 4 * np.sqrt(max(ev[0], 1e-6))
        width = area / max(major, 1.0)
        # local contrast: mean intensity in mask vs a ring 3..8 px outside it
        pad = 10
        X0, Y0 = max(x - pad, 0), max(y - pad, 0)
        X1, Y1 = min(x + w + pad, 2048), min(y + h + pad, 2048)
        big = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        big[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = m
        ring = (cv2.dilate(big, np.ones((17, 17), np.uint8)) > 0) & ~(cv2.dilate(big, np.ones((7, 7), np.uint8)) > 0)
        crop = np.asarray(img[Y0:Y1, X0:X1], np.float32)
        inside = crop[big > 0].mean()
        outside = crop[ring].mean() if ring.any() else inside
        contrast = float(inside / max(outside, 1.0))
        q = c["q"]
        F.append([q, np.log(q + 1e-4) - np.log(1 - q + 1e-4), c["peak_u"], c["mean_p"], mprob, c["p95"],
                  np.log(area), c["level"] == "A", c["level"] == "P", c["level"] == "B", c["level"] == "C",
                  p0.mean(), np.percentile(p0, 90), p1.mean(), n_ag, mq, rR, elong, width, contrast, len(cs)])
F = np.array(F, np.float64)
pickle.dump(dict(names=NAMES, F=F), open(f"feats_{TAG}.pkl", "wb"))
print(F.shape, "nan rows", np.isnan(F).any(1).sum())
