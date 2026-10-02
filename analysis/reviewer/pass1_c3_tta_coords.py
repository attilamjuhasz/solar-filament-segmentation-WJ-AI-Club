import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, cv2
sys.path.insert(0, "src")
import s1
from common import disk_info, radius_map, make_planes, load_meta, CACHE

torch.set_num_threads(2)
dev = torch.device("cpu")

# 1) TTA inverse: a position-dependent "model" that returns its first two input channels as logits.
rng = np.random.default_rng(0)
fake = {s: rng.normal(size=(3, 1024, 1024)).astype(np.float32) for s in ["a", "b", "c"]}
s1.full_planes = lambda s: fake[s]


class Ident(torch.nn.Module):
    def forward(self, x):
        return x[:, :2]


out = s1.predict_probs(Ident(), ["a", "b", "c"], dev, tta=True, bs=2)
for s in out:
    ref = 1 / (1 + np.exp(-fake[s][:2]))
    print("TTA identity max abs err", s, float(np.abs(out[s].astype(np.float32) - ref).max()))


# position-dependent non-equivariant model: must still invert exactly if each view is undone
class Shift(torch.nn.Module):
    def forward(self, x):
        return torch.roll(x[:, :2], 1, 3)  # shift right by 1 in the *view* frame
out = s1.predict_probs(Shift(), ["a"], dev, tta=True)
# expectation: average of D4-conjugated shifts -> each view's inverse maps "right shift" to a different direction
print("(sanity) shift-model differs from identity:", float(np.abs(out["a"].astype(np.float32) - 1 / (1 + np.exp(-fake["a"][:2]))).max()) > 0.01)

# 2) radius_map pixel-center: 1024 grid with step 2 vs. 2x2 average of native r/R
stem = load_meta().file_name.str[:-5].iloc[0]
info = disk_info(stem)
r2048 = radius_map(info, (2048, 2048))
r_avg = r2048.reshape(1024, 2, 1024, 2).mean((1, 3))
r1024 = radius_map(info, (1024, 1024), 0, 0, 2.0)
print("r/R 1024(step2) vs 2x2-avg native: max abs diff", float(np.abs(r1024 - r_avg).max()))
# crop with offset
ox, oy = 300, 123
rc = radius_map(info, (512, 512), 2.0 * ox, 2.0 * oy, 2.0)
print("crop r/R vs full 1024 r/R crop: max abs diff", float(np.abs(rc - r1024[oy:oy + 512, ox:ox + 512]).max()))

# 3) make_planes crop == full planes crop
img = np.load(os.path.join(CACHE, "img1024", stem + ".npy"))
full = make_planes(img, info, 0, 0, 2.0)
crop = make_planes(np.ascontiguousarray(img[oy:oy + 512, ox:ox + 512]), info, 2.0 * ox, 2.0 * oy, 2.0)
print("make_planes crop vs full-crop max abs diff", float(np.abs(full[:, oy:oy + 512, ox:ox + 512] - crop).max()))

# 4) cv2 INTER_AREA 2048->1024 == exact 2x2 mean ; INTER_LINEAR upsample half-pixel convention
a = rng.integers(0, 256, (2048, 2048)).astype(np.float32)
print("INTER_AREA == 2x2 mean:", float(np.abs(cv2.resize(a, (1024, 1024), interpolation=cv2.INTER_AREA) - a.reshape(1024, 2, 1024, 2).mean((1, 3))).max()))
p = np.zeros((1024, 1024), np.float32); p[500, 600] = 1
u = cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR)
ys, xs = np.nonzero(u == u.max())
print("upsampled peak of 1024 px (500,600) lands on native rows", ys, "cols", xs, "(expect 1000-1001, 1200-1201)")
