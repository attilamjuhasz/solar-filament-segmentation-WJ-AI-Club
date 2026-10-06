import os, sys, json
import numpy as np, cv2, pandas as pd
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_meta, load_inst, disk_info
from solgeo import warp_maps, stem_ts
m = load_meta(); m["stem"] = m.file_name.str[:-5]; READ = m.groupby("stem").image_id.apply(list).to_dict()
def best_ious(la, lb):
    pair = la.astype(np.int32) * 256 + lb
    bc = np.bincount(pair.ravel(), minlength=65536).reshape(256, 256).astype(float)
    a = bc.sum(1); b = bc.sum(0)
    iou = bc / np.maximum(a[:, None] + b[None, :] - bc, 1); iou[0] = 0; iou[:, 0] = 0
    return iou[1:][a[1:] > 0].max(1)
for s, t in [("20160430133734Ch", "20160430133714Th"), ("20141013195834Ch", "20141013195854Bh"), ("20130303063154Uh", "20130303063134Lh")]:
    if s not in READ or t not in READ:
        print("skip", s, t, s in READ, t in READ); continue
    print(s, READ[s], t, READ[t], "disk s", {k: round(v, 1) for k, v in disk_info(s).items() if k in "cx cy r"}, "disk t", {k: round(v, 1) for k, v in disk_info(t).items() if k in "cx cy r"})
    ls = load_inst(READ[s][0])
    # same-frame other annotator, if any
    if len(READ[s]) > 1:
        print("  same frame other annotator: median best IoU", np.median(best_ious(ls, load_inst(READ[s][1]))).round(3))
    mx, my, valid, dt = warp_maps(disk_info(s), stem_ts(s), disk_info(t), stem_ts(t))
    for r in READ[t]:
        lt = load_inst(r)
        w = cv2.remap(lt, mx, my, cv2.INTER_NEAREST)
        bi = best_ious(ls, w)
        print("  warped", r, "median best IoU", np.median(bi).round(3), "frac>.5", (bi > .5).mean().round(3), "n", len(bi))
        bi0 = best_ious(ls, lt)
        print("  identity (no warp)  median", np.median(bi0).round(3), "frac>.5", (bi0 > .5).mean().round(3))
        best = (0, 0, 0)
        for dx in range(-6, 7, 2):
            for dy in range(-6, 7, 2):
                w2 = np.roll(np.roll(w, dy, 0), dx, 1)
                v = np.median(best_ious(ls, w2))
                if v > best[0]: best = (v, dx, dy)
        print("  best extra shift", best)
    # overlay picture
    img = np.load(os.path.join(ROOT, "data/cache/img2048", s + ".npy"))
    rgb = np.dstack([img] * 3).copy()
    rgb[ls > 0] = (0, 255, 0)
    w = cv2.remap(load_inst(READ[t][0]), mx, my, cv2.INTER_NEAREST)
    rgb[(w > 0) & (ls > 0)] = (255, 255, 0)
    rgb[(w > 0) & (ls == 0)] = (255, 0, 0)
    cv2.imwrite(f"pair_{s}.png", cv2.resize(rgb[300:1300, 300:1300], (1000, 1000))[:, :, ::-1])
