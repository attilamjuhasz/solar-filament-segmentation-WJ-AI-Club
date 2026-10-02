import os, sys, time, tracemalloc
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, cv2
sys.path.insert(0, "src")
import s1
from common import CACHE, disk_info, disk_mask
from rle import encode, write_submission, validate_submission

tr, va, readings = s1.split(0)
rng = np.random.default_rng(0)
counts = []
for s in va[:4]:
    un = np.load(os.path.join(CACHE, "un1024", s + ".npy")).astype(np.float32) / 255
    fg = np.load(os.path.join(CACHE, "fg1024", readings[s][0] + ".npy")).astype(np.float32) / 255
    # "noisy model": GT + smooth noise
    noise = cv2.GaussianBlur(rng.normal(0, 1, (1024, 1024)).astype(np.float32), (0, 0), 4) * 6
    prob = np.clip(np.stack([fg, un]) * 0.8 + noise, 0, 1).astype(np.float16)
    for gap in (0, 3):
        m = s1.postprocess_s1(prob, s, gap=gap)
        if not m:
            continue
        st = np.stack(m)
        dm = disk_mask(disk_info(s))
        assert st.sum(0).max() <= 1, "overlap"
        assert all(x.any() for x in m), "empty"
        assert not (st.any(0) & ~dm).any(), "outside disk"
        ncc = [0] or [cv2.connectedComponents(x.astype(np.uint8), connectivity=8)[0] - 1 for x in m]
        counts.append((s, gap, len(m), max(ncc)))
print("(stem, gap, n_masks, max CCs per mask):", counts)
print("bytes per returned mask:", m[0].nbytes, "dtype", m[0].dtype)

# uint8 probs (as saved by predict()) fed without /255
p8 = (prob.astype(np.float32) * 255).round().astype(np.uint8)
t = time.time()
m8 = s1.postprocess_s1(p8, s)
mf = s1.postprocess_s1(prob, s)
print("uint8 probs w/o /255 -> n masks", len(m8), "fg px", int(sum(x.sum() for x in m8)), "| float probs -> n masks", len(mf), "fg px", int(sum(x.sum() for x in mf)))

# RLE + validator on these masks
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sub.csv")
write_submission(path, {s: mf})
print(validate_submission(path, [s]))
