"""End-to-end refine() with a fake model whose mask logit = logit(prior channel). Then assemble.cand_mask."""
import os, sys, pickle, numpy as np, torch, cv2
sys.path.insert(0, "src")
from pycocotools import mask as mu
import s2
from assemble import cand_mask, DEFAULT
from rle import _full

class Fake(torch.nn.Module):
    def eval(self): return self
    def forward(self, x):
        p = x[:, 3:4].clamp(1e-4, 1 - 1e-4)
        return torch.log(p / (1 - p)), torch.zeros(len(x), 1)

def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)

stems = sorted(f[:-4] for f in os.listdir("runs/s1_r34_f0/props_tta") if f.endswith(".pkl"))[:6]
res = []
for st in stems:
    props = pickle.load(open(f"runs/s1_r34_f0/props_tta/{st}.pkl", "rb"))
    cands = s2.refine(Fake(), st, props, torch.device("cpu"), tta=True)
    for c in cands:
        c["_cache"] = {}; c["p95"] = 0
        m, a, _ = cand_mask(c, dict(DEFAULT, thr=0.5))
        full = _full((c["x"], c["y"], m))
        prior = mu.decode(props[c["idx"]]["rle"]).astype(bool)
        res.append(iou(full, prior))
print("real props: n", len(res), "IoU mean %.4f min %.4f" % (np.mean(res), np.min(res)))

# synthetic blobs near frame edges / corners incl. windows that leave the frame
rng = np.random.default_rng(1)
st = stems[0]
out = []
for t in range(60):
    m = np.zeros((2048, 2048), np.uint8)
    L = int(rng.integers(30, 900)); wdt = int(rng.integers(3, 12))
    cx = int(rng.choice([rng.integers(0, 60), rng.integers(1988, 2048), rng.integers(0, 2048)]))
    cy = int(rng.choice([rng.integers(0, 60), rng.integers(1988, 2048), rng.integers(0, 2048)]))
    a = rng.uniform(0, np.pi)
    p1 = (int(cx - L / 2 * np.cos(a)), int(cy - L / 2 * np.sin(a))); p2 = (int(cx + L / 2 * np.cos(a)), int(cy + L / 2 * np.sin(a)))
    cv2.line(m, p1, p2, 1, wdt)
    m = m.astype(bool)
    if m.sum() < 30: continue
    ys, xs = np.nonzero(m)
    box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    d = dict(rle=mu.encode(np.asfortranarray(m.astype(np.uint8))), box=box, level="A", peak_u=1.0, mean_p=1.0)
    cs = s2.refine(Fake(), st, [d], torch.device("cpu"), tta=True)
    if not cs:
        out.append((0.0, box, s2.window(box))); continue
    c = cs[0]; c["_cache"] = {}; c["p95"] = 0
    mm, _, _ = cand_mask(c, dict(DEFAULT, thr=0.5))
    full = _full((c["x"], c["y"], mm))
    out.append((iou(full, m), box, s2.window(box)))
v = np.array([o[0] for o in out])
print("synthetic edge blobs: n", len(v), "IoU mean %.4f min %.4f" % (v.mean(), v.min()))
for o in sorted(out, key=lambda o: o[0])[:4]: print("  worst", o)
