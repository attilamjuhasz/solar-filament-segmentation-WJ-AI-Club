"""Idea (b): temporal / cross-site neighbours. For each fold-0 val stem, warp labelled neighbours' GT (any other
labelled stem within 48 h; never the target's own labels) and out-of-sample neighbours' S1 probs (val/test stems)
into the target frame with heliographic differential rotation (images are solar-north-up, east-left: verified in reg.py).
Outputs nbr.pkl with per-candidate features, per-GT-instance neighbour coverage, and direct-transfer PQ counts."""
import os, sys, json, pickle, time
import numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
cv2.setNumThreads(2)
from common import load_meta, load_inst
from solgeo import warp_maps, stem_ts, img_to_helio, sun_angles

D = json.load(open(os.path.join(ROOT, "data/cache/disk.json")))
meta = load_meta(); meta["stem"] = meta.file_name.str[:-5]
READ = meta.groupby("stem").image_id.apply(list).to_dict()
FOLD = dict(zip(meta.stem, meta.fold))
TEST = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "data/MAGFiLO_1.0_Kaggle_2026/test/test_images")))
T = pickle.load(open(os.path.join(HERE, "table.pkl"), "rb"))
VAL = T["stems"]
ALL = sorted(set(READ) | set(TEST))
TS = {s: stem_ts(s) for s in ALL}
K7 = np.ones((7, 7), np.uint8)
MAXH = 48


def nbrs(s):
    out = []
    for t in ALL:
        if t == s:
            continue
        dh = (TS[s] - TS[t]).total_seconds() / 3600
        if abs(dh) <= MAXH:
            out.append((abs(dh), t))
    return sorted(out)


def warp_label(t, s, lab, mx, my):
    return cv2.remap(lab, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


cands = {}
import pickle as pk
for s in VAL:
    cands[s] = pk.load(open(os.path.join(ROOT, "runs/s2_r34/cands_val_tta_last", s + ".pkl"), "rb"))

res = dict(cand={}, gtcov={}, direct=[], nbinfo={})
t0 = time.time()
for si, s in enumerate(VAL):
    nb = nbrs(s)
    res["nbinfo"][s] = [(dh, t, "lab" if t in READ else "test", FOLD.get(t, -1)) for dh, t in nb]
    tgt_labs = [load_inst(r) for r in READ[s]]
    tgt_areas = [np.bincount(l.ravel(), minlength=256) for l in tgt_labs]
    cs = cands[s]
    cmasks = [(c["x"], c["y"], c["soft"] > 127) for c in cs]
    feats = [[] for _ in cs]   # per candidate: list of dicts per neighbour
    gtc = [[[] for _ in range(int(a.size))] for a in tgt_areas]
    nb_use = [z for z in nb if z[1] in READ][:4] + [z for z in nb if z[1] not in READ or FOLD.get(z[1]) == 0][:4]
    nb_use = sorted(set(nb_use))
    for dh, t in nb_use:
        mx, my, valid, dt = warp_maps(D[s], TS[s], D[t], TS[t], shape=(2048, 2048), step=1.0)
        # limit to where the neighbour view is not too foreshortened: neighbour mu > 0.3
        info = D[t]
        rr_n = np.hypot(mx - info["cx"], my - info["cy"]) / info["r"]
        valid &= rr_n < 0.95
        if t in READ:
            wl = [warp_label(t, s, load_inst(r), mx, my) * valid for r in READ[t]]
            wl = [w.astype(np.uint8) for w in wl]
            wareas = [np.bincount(w.ravel(), minlength=256) for w in wl]
            wdil = [cv2.dilate((w > 0).astype(np.uint8), K7) > 0 for w in wl]
            # candidates
            for k, (x, y, m) in enumerate(cmasks):
                h, w_ = m.shape
                area = int(m.sum())
                if area == 0:
                    feats[k].append(dict(t=t, dh=dh, kind="lab", vfrac=0.0, iou=np.zeros(len(wl)), cov=np.zeros(len(wl))))
                    continue
                vfrac = float(valid[y:y + h, x:x + w_][m].mean())
                ious, covs = [], []
                for w, a, dl in zip(wl, wareas, wdil):
                    inter = np.bincount(w[y:y + h, x:x + w_][m], minlength=256); inter[0] = 0
                    iou = inter / np.maximum(area + a - inter, 1)
                    ious.append(float(iou.max())); covs.append(float(dl[y:y + h, x:x + w_][m].mean()))
                feats[k].append(dict(t=t, dh=dh, kind="lab", vfrac=vfrac, iou=np.array(ious), cov=np.array(covs)))
            # target GT instances: best IoU with any warped neighbour instance, coverage within 3 px
            for ri, (tl, ta) in enumerate(zip(tgt_labs, tgt_areas)):
                for w, a, dl in zip(wl, wareas, wdil):
                    pair = tl.astype(np.int32) * 256 + w
                    bc = np.bincount(pair.ravel(), minlength=65536).reshape(256, 256).astype(float)
                    union = ta[:, None] + a[None, :] - bc
                    iou = np.where(union > 0, bc / np.maximum(union, 1), 0); iou[:, 0] = 0
                    cov = np.bincount(tl[dl], minlength=256) / np.maximum(ta, 1)
                    vf = np.bincount(tl[valid], minlength=256) / np.maximum(ta, 1)
                    for g in range(1, 256):
                        if ta[g] > 0:
                            gtc[ri][g].append((dh, t, float(iou[g].max()), float(cov[g]), float(vf[g])))
                    # direct transfer from the same IoU matrix: neighbour instances as predictions vs this target reading
                    S = TP = FP = 0; matched = set()
                    for g in range(1, 256):
                        if a[g] < 20:
                            continue
                        j = int(iou[:, g].argmax())
                        if iou[j, g] > 0.5:
                            S += iou[j, g]; TP += 1; matched.add(j)
                        else:
                            FP += 1
                    n_gt = int((ta[1:] > 0).sum())
                    res["direct"].append((s, ri, t, dh, S, TP, FP, n_gt - len(matched)))
        if (t in TEST) or FOLD.get(t, -1) == 0:   # out-of-sample S1 probs for the neighbour
            pr = np.load(os.path.join(ROOT, "runs/s1_r34_f0/probs_tta", t + ".npy"))
            mx2, my2 = (mx + 0.5) / 2 - 0.5, (my + 0.5) / 2 - 0.5
            wp = [cv2.remap(pr[ch], mx2, my2, cv2.INTER_LINEAR) * valid for ch in (0, 1)]
            for k, (x, y, m) in enumerate(cmasks):
                h, w_ = m.shape
                if not m.any():
                    continue
                vfrac = float(valid[y:y + h, x:x + w_][m].mean())
                mp = [float(p[y:y + h, x:x + w_][m].mean()) / 255 for p in wp]
                # tolerant: max-pool the warped prob by 5px before averaging
                mpd = [float(cv2.dilate(p[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w_ + 4], np.ones((5, 5), np.uint8))
                             [y - max(y - 4, 0):y - max(y - 4, 0) + h, x - max(x - 4, 0):x - max(x - 4, 0) + w_][m].mean()) / 255 for p in wp]
                feats[k].append(dict(t=t, dh=dh, kind="s1", vfrac=vfrac, mp=mp, mpd=mpd))
    res["cand"][s] = feats
    res["gtcov"][s] = gtc
    print(si, s, f"nb={len(nb)} lab={sum(t in READ for _, t in nb)} test={sum(t in TEST for _, t in nb)}", f"{time.time() - t0:.0f}s", flush=True)
pickle.dump(res, open(os.path.join(HERE, "nbr.pkl"), "wb"))
print("done")
