"""CPU S1 inference (plain, no TTA) on downloaded GONG archive frames -> gong/<val_stem>/<frame>.npz (probs u8, disk info)."""
import os, sys, glob, json, time
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_num_threads(int(os.environ.get("NT", "3")))
cv2.setNumThreads(1)
from s1 import build_model
from disk import fit_disk, refine_limb, radial_profile
from common import make_planes

model = build_model(pretrained=False)
sd = torch.load(os.path.join(ROOT, "runs/s1_r34_f0/best.pt"), map_location="cpu")
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
model.load_state_dict(sd)
DEV = torch.device(os.environ.get("DEV", "cpu"))
model = model.to(DEV).eval()
part, nparts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
files = sorted(glob.glob(os.path.join(HERE, "gong", "*", "*.jpg")))[part::nparts]
t0 = time.time()
for i, f in enumerate(files):
    out = f[:-4] + ".npz"
    if os.path.exists(out):
        continue
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (2048, 2048):
        print("bad", f); continue
    cx, cy, r0, _ = fit_disk(img)
    cx, cy, r = refine_limb(img, cx, cy, r=float(np.clip(r0, 885, 925)))
    yy, xx = np.mgrid[:2048, :2048]
    img = img.copy(); img[np.hypot(xx - cx, yy - cy) > 1.15 * r] = 0   # drop the archive's text overlays (outside the disk)
    prof, med, iqr = radial_profile(img, cx, cy, r)
    info = dict(cx=float(cx), cy=float(cy), r=float(r), med=med, iqr=max(iqr, 1.0), prof=prof)
    x = make_planes(cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA), info, 0.0, 0.0, 2.0)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x)[None].to(DEV).contiguous()))[0].cpu().numpy()
    np.savez_compressed(out, p=np.round(p * 255).astype(np.uint8), info=json.dumps(info))
    print(i, os.path.basename(f), f"r={r:.1f}", f"{time.time() - t0:.0f}s", flush=True)
print("done")
