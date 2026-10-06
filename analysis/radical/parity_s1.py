import os, sys, glob, json, time
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_num_threads(3)
from s1 import build_model
from disk import fit_disk, refine_limb, radial_profile
from common import make_planes, disk_info
model = build_model(pretrained=False); model.load_state_dict(torch.load(os.path.join(ROOT, "runs/s1_r34_f0/best.pt"), map_location="cpu")); model.eval()
s = "20140609195854Bh"
ref = np.load(os.path.join(ROOT, "runs/s1_r34_f0/probs_plain", s + ".npy")).astype(np.float32)
ref = ref / 255 if ref.max() > 1.5 else ref
for src in ("kaggle", "archive"):
    if src == "kaggle":
        img = np.load(os.path.join(ROOT, "data/cache/img2048", s + ".npy"))
    else:
        img = cv2.imread(os.path.join(HERE, "parity/x", s + ".jpg"), 0)
    t0 = time.time()
    cx, cy, r0, _ = fit_disk(img); cx, cy, r = refine_limb(img, cx, cy, r=float(np.clip(r0, 885, 925)))
    yy, xx = np.mgrid[:2048, :2048]; img = img.copy(); img[np.hypot(xx - cx, yy - cy) > 1.15 * r] = 0
    prof, med, iqr = radial_profile(img, cx, cy, r)
    info = dict(cx=cx, cy=cy, r=r, med=med, iqr=max(iqr, 1.0), prof=prof)
    print(src, "disk", round(cx, 2), round(cy, 2), round(r, 2), "cached", {k: round(v, 2) for k, v in disk_info(s).items() if k in ("cx", "cy", "r", "med", "iqr")})
    x = make_planes(cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA), info, 0.0, 0.0, 2.0)
    t1 = time.time()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x)[None]))[0].numpy()
    print(src, f"prep {t1 - t0:.1f}s infer {time.time() - t1:.1f}s", "mean|dp|", np.abs(p - ref).mean(), "max", np.abs(p - ref).max(),
          "corr", np.corrcoef(p[0].ravel(), ref[0].ravel())[0, 1], "frac>.5 agree", ((p[0] > .5) == (ref[0] > .5)).mean())
