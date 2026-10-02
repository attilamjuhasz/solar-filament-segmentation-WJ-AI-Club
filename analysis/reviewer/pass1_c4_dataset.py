"""Plant r/R into the target files so input plane 2 and target must coincide after crop + D4."""
import os, sys, json, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, "src")
import s1
from common import CACHE, disk_info, radius_map

SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakecache")
for sub in ("fg1024", "un1024"):
    os.makedirs(os.path.join(SCR, sub), exist_ok=True)
if not os.path.exists(os.path.join(SCR, "img1024")):
    os.symlink(os.path.join(CACHE, "img1024"), os.path.join(SCR, "img1024"))
if not os.path.exists(os.path.join(SCR, "boxes.json")):
    os.symlink(os.path.join(CACHE, "boxes.json"), os.path.join(SCR, "boxes.json"))

tr, va, readings = s1.split(0)
stems = tr[:4]
for s in stems:
    rr = np.minimum(radius_map(disk_info(s), (1024, 1024), 0, 0, 2.0), 1.5)
    t = np.round(rr * 100).astype(np.uint8)
    for rid in readings[s]:
        np.save(os.path.join(SCR, "fg1024", rid + ".npy"), t)
    np.save(os.path.join(SCR, "un1024", s + ".npy"), t)

s1.CACHE = SCR
ds = s1.S1Train(stems, readings, 64, seed=0)
pickle.dumps(ds)  # spawn picklability
worst = 0
for i in range(64):
    x, y = ds[i]
    assert x.shape == (3, 512, 512) and y.shape == (2, 512, 512) and x.dtype == y.dtype
    exp = np.round(x[2].numpy() * 100)
    worst = max(worst, float(np.abs(y[0].numpy() * 255 - exp).max()), float(np.abs(y[1].numpy() * 255 - exp).max()))
print("max |target - planted r/R| over 64 aug samples (should be <=1 rounding):", worst)
# duplicates across epochs / same i
a = ds[5][0]; b = ds[5][0]
print("same index twice gives different samples:", not np.allclose(a.numpy(), b.numpy()))
