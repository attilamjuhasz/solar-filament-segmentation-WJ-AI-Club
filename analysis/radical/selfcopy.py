"""Alignment control: archive copy of the TARGET frame itself (same exposure) and 1-min neighbours.
(1) download self copies; (2) S1 on MPS; (3) S1-only PQ of warped self-copy vs warped neighbour; (4) residual shift by xcorr."""
import os, sys, glob, json, subprocess
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__))
from decode import *
sys.path.insert(0, HERE)
from solgeo import warp_maps, stem_ts
from s1 import build_model, postprocess_s1
from disk import fit_disk, refine_limb, radial_profile
from common import make_planes, disk_info
torch.set_num_threads(3)
S1BEST = dict(t_hi=0.6, t_lo=0.45, gap=8, min_area=400, head=0)
sub = [s for s in stems if glob.glob(os.path.join(HERE, "gong", s, "*.npz"))][:44]
os.makedirs(os.path.join(HERE, "selfcopy"), exist_ok=True)
dev = torch.device("mps")
model = build_model(pretrained=False); model.load_state_dict(torch.load(os.path.join(ROOT, "runs/s1_r34_f0/best.pt"), map_location="cpu")); model = model.to(dev).eval()


def infer(img):
    cx, cy, r0, _ = fit_disk(img); cx, cy, r = refine_limb(img, cx, cy, r=float(np.clip(r0, 885, 925)))
    yy, xx = np.mgrid[:2048, :2048]; img = img.copy(); img[np.hypot(xx - cx, yy - cy) > 1.15 * r] = 0
    prof, med, iqr = radial_profile(img, cx, cy, r)
    info = dict(cx=float(cx), cy=float(cy), r=float(r), med=med, iqr=max(iqr, 1.0), prof=prof)
    x = make_planes(cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA), info, 0.0, 0.0, 2.0)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x)[None].to(dev).contiguous()))[0].cpu().numpy()
    return p, info


def warp(p, info, s, st):
    mx, my, valid, dt = warp_maps(disk_info(s), stem_ts(s), info, stem_ts(st), shape=(1024, 1024), step=2.0)
    return np.stack([cv2.remap(p[c], mx, my, cv2.INTER_LINEAR) for c in (0, 1)])


def xshift(a, b):
    """best integer+parabolic subpixel shift of b relative to a (1024 px) by xcorr in central region"""
    a = a[200:824, 200:824].astype(np.float32); best = None
    res = {}
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            bb = np.roll(np.roll(b, dy, 0), dx, 1)[200:824, 200:824]
            res[(dy, dx)] = float((a * bb).sum() / np.sqrt((a * a).sum() * (bb * bb).sum()))
    k = max(res, key=res.get)
    return k, res[k], res[(0, 0)]


acc = {k: np.zeros(4) for k in ("kaggle plain (cached)", "archive self-copy warped", "1-min same-site nbr warped", "nbr warped + best global shift")}
shifts = []
for s in sub:
    f = os.path.join(HERE, "selfcopy", s + ".jpg")
    if not os.path.exists(f):
        ts = stem_ts(s)
        subprocess.run(["curl", "-s", "-m", "120", "-o", f, f"https://gong2.nso.edu/HA/hag/{ts.strftime('%Y%m')}/{s[:8]}/{s}.jpg"])
    img = cv2.imread(f, 0)
    if img is None:
        continue
    pself, iself = infer(img)
    ws = warp(pself, iself, s, s)
    pk = np.load(os.path.join(ROOT, "runs/s1_r34_f0/probs_plain", s + ".npy")).astype(np.float32)
    pk = pk / 255 if pk.max() > 1.5 else pk
    same = sorted(g for g in glob.glob(os.path.join(HERE, "gong", s, "*.npz")) if os.path.basename(g)[-6] == s[-2])
    if not same:
        continue
    z = np.load(same[0]); pn = z["p"].astype(np.float32) / 255; inf_n = json.loads(str(z["info"]))
    wn = warp(pn, inf_n, s, os.path.basename(same[0])[:-4])
    k, c, c0 = xshift(pk[0], wn[0]); ks, cs_, cs0 = xshift(pk[0], ws[0])
    shifts.append((s, k, round(c, 3), round(c0, 3), ks, round(cs_, 3)))
    wn_shift = np.roll(np.roll(wn, k[0], 1), k[1], 2)
    for name, p in (("kaggle plain (cached)", pk), ("archive self-copy warped", ws), ("1-min same-site nbr warped", wn), ("nbr warped + best global shift", wn_shift)):
        inst = postprocess_s1(np.clip(p, 0, 1).copy(), s, **S1BEST)
        acc[name] += sum(stats_reading(s, ri, inst) for ri in range(NR[s]))
for name, t in acc.items():
    print(f"{name:35s} PQ={pq(t):.4f} {cnt(t)}")
print("xcorr shifts (stem, nbr best shift (dy,dx) @1024, corr best, corr at 0, self-copy best shift, corr):")
for r in shifts:
    print(" ", r)
