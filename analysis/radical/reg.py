"""Verify GONG orientation conventions by registering frame pairs from different sites.
For each pair, warp the earlier frame into the later one with heliographic differential rotation, scanning
an assumed extra roll angle P (deg) and the rotation-rate multiplier. If images are solar-north-up (P-corrected),
the best roll is ~0 for every season; if they were celestial-north-up, best roll would track ephemeris P(date)."""
import os, sys, json, itertools
import numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
cv2.setNumThreads(1)
from common import make_planes
from solgeo import warp_maps, sun_angles, stem_ts

D = json.load(open(os.path.join(ROOT, "data/cache/disk.json")))
stems = sorted(D)
ts = {s: stem_ts(s) for s in stems}


def hp(stem):
    img = np.load(os.path.join(ROOT, "data/cache/img1024", stem + ".npy"))
    p = make_planes(img, D[stem], step=2.0)
    c = p[1]
    g = cv2.GaussianBlur(c, (0, 0), 1.0) - cv2.GaussianBlur(c, (0, 0), 6.0)
    return np.clip(g, -1.5, 1.5), p[2]


def corr(a, b, m):
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


pairs = []
for i, s in enumerate(stems):
    for t in stems[i + 1:]:
        dh = abs((ts[t] - ts[s]).total_seconds()) / 3600
        if s[-2] != t[-2] and dh <= 30:
            pairs.append((dh, s, t))
pairs.sort()
rng = np.random.default_rng(0)
near = [p for p in pairs if p[0] < 1]
far = [p for p in pairs if 3 < p[0] <= 30]
sel = list(rng.choice(len(near), min(8, len(near)), replace=False))
sel = [near[i] for i in sel] + [far[i] for i in rng.choice(len(far), min(28, len(far)), replace=False)]
print(f"{len(pairs)} cross-site pairs <=30h; {len(near)} within 1h; testing {len(sel)}", flush=True)
res = []
for dh, s, t in sel:
    a, rr = hp(t)  # target = t (later)
    b, _ = hp(s)
    Pe, B0, _ = sun_angles(ts[t])
    best = None
    out = {}
    for rate in (-1.0, 0.0, 0.5, 0.8, 1.0, 1.2):
        for P in (-30, -20, -12, -6, -3, 0, 3, 6, 12, 20, 30) if rate == 1.0 else (0,):
            mx, my, val, dt = warp_maps(D[t], ts[t], D[s], ts[s], shape=(1024, 1024), step=2.0, P=P, rate=rate)
            w = cv2.remap(b, mx, my, cv2.INTER_LINEAR)
            m = val & (rr < 0.92)
            c = corr(a, w, m)
            out[(rate, P)] = c
            if best is None or c > best[0]:
                best = (c, rate, P)
    # refine P around best at rate 1
    r1 = {P: out[(1.0, P)] for P in (-30, -20, -12, -6, -3, 0, 3, 6, 12, 20, 30)}
    bp = max(r1, key=r1.get)
    print(f"dt={dh:5.1f}h {s} -> {t} P_eph={Pe:6.1f} B0={B0:5.1f} | best rate={best[1]} P={best[2]} c={best[0]:.3f} | "
          f"c(rate1,P0)={out[(1.0, 0)]:.3f} c(rate0)={out[(0.0, 0)]:.3f} c(rate-1)={out[(-1.0, 0)]:.3f} "
          f"c(rate1,P=Peph~{int(round(Pe))})~{r1[min(r1, key=lambda p: abs(p - Pe))]:.3f}", flush=True)
    res.append(dict(dh=dh, s=s, t=t, Pe=Pe, rate=best[1], P=best[2], c=best[0], c0=out[(1.0, 0)], cr0=out[(0.0, 0)]))
pd.DataFrame(res).to_csv(os.path.join(HERE, "reg.csv"), index=False)
