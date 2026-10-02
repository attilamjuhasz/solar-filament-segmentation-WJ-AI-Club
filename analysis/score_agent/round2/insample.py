"""Q3c: S1 in-sample (train folds) vs out-of-sample (val) behaviour -> what S2 sees in training vs at test time."""
from base import *
from assemble import FastGT
from common import load_meta
from s1 import postprocess_s1
import sys
PROBS = sys.argv[1] if len(sys.argv) > 1 else "plain"
meta = load_meta(); meta["stem"] = meta.file_name.str[:-5]
val = sorted(meta[meta.fold == 0].stem.unique())
tr_all = sorted(meta[meta.fold != 0].stem.unique())
rng = np.random.default_rng(0)
tr = sorted(rng.choice(tr_all, 150, replace=False))
S1BEST = dict(t_hi=0.6, t_lo=0.45, gap=8, min_area=400, head=0)


def run(stems, name):
    G = FastGT(stems)
    t = np.zeros(4)
    ys, ms, nprop, pin, pout, cal = [], [], [], [], [], []
    for s in stems:
        pr = np.load(os.path.join(ROOT, f"runs/s1_r34_f0/probs_{PROBS}", s + ".npy"))
        t += G.stats(s, postprocess_s1(pr, s, **S1BEST))
        props = pickle.load(open(os.path.join(ROOT, f"runs/s1_r34_f0/props_{PROBS}", s + ".pkl"), "rb"))
        props = [d for d in props if d["level"] in "AP"]
        nprop.append(len(props))
        for d in props:
            m = mu.decode(d["rle"]).astype(bool)
            x, y, w, h = d["box"]
            loc = m[y:y + h, x:x + w]
            ious = []
            for lab, areas in G.by_stem[s]:
                inter = np.bincount(lab[y:y + h, x:x + w][loc], minlength=256); inter[0] = 0
                iou = inter / np.maximum(loc.sum() + areas - inter, 1)
                ious.append(iou.max())
            ious = np.array(ious)
            ys.append(float((ious * (ious > .5)).mean())); ms.append(float((ious > .5).mean()))
            cal.append((d["mean_p"], float((ious > .5).mean())))
        p0 = cv2.resize(pr[0].astype(np.float32) / 255, (2048, 2048))
        un = np.zeros((2048, 2048), bool)
        for lab, _ in G.by_stem[s]:
            un |= lab > 0
        pin.append(p0[un].mean()); pout.append((p0[~un] > .5).mean())
    cal = np.array(cal)
    bins = [0, .5, .7, .85, 1.01]
    b = np.digitize(cal[:, 0], bins) - 1
    calib = {f"{bins[i]}-{bins[i+1]}": round(float(cal[b == i, 1].mean()), 3) for i in range(4) if (b == i).any()}
    print(f"[{name}] stems {len(stems)}: S1-only PQ {pq(t):.4f} ({cnt(t)}) | A/P props/img {np.mean(nprop):.1f}, "
          f"prop match rate {np.mean(ms):.3f}, mean y {np.mean(ys):.3f} | p0 mean on GT px {np.mean(pin):.3f}, "
          f"bg px with p0>.5 {np.mean(pout) * 1e3:.2f}e-3 | P(match | prop mean_p bin) {calib}", flush=True)


run(val, f"val (fold 0, OOF) probs_{PROBS}")
run(tr, f"train folds 1-4 (in-sample) probs_{PROBS}")
