import csv, json, os, re, sys, numpy as np
from collections import defaultdict
from pycocotools import mask as mu
path = sys.argv[1]
test = {os.path.splitext(f)[0] for f in os.listdir("data/MAGFiLO_1.0_Kaggle_2026/test/test_images")}
disk = json.load(open("data/cache/disk.json"))
rows = list(csv.reader(open(path, newline="")))
raw = open(path, "rb").read()
print("header", rows[0], "rows", len(rows) - 1, "CRLF" if b"\r\n" in raw else "LF", "trailing-newline", raw.endswith(b"\n"))
ids = [r[0] for r in rows[1:]]
assert len(ids) == len(set(ids))
bad_id = [i for i in ids if not re.fullmatch(r"\d{14}[A-Z][a-z]_\d+", i)]
print("ids not matching <stem>_<n>:", bad_id[:5], len(bad_id))
per = defaultdict(list)
for fid, cnt in rows[1:]:
    st, _, k = fid.rpartition("_")
    assert st in test, st
    per[st].append((int(k), cnt))
print("stems with rows", len(per), "/", len(test), "stems without rows", len(test - set(per)))
areas, outside, out_px, overl, noncontig, ncomp = [], 0, 0, 0, 0, []
yy, xx = np.mgrid[0:2048, 0:2048]
for st, lst in per.items():
    ks = sorted(k for k, _ in lst)
    if ks != list(range(1, len(ks) + 1)): noncontig += 1
    d = disk[st]
    dm = np.hypot(xx - d["cx"], yy - d["cy"]) <= d["r"]
    acc = np.zeros((2048, 2048), np.uint16)
    for k, cnt in lst:
        rle = {"size": [2048, 2048], "counts": cnt.encode()}
        m = mu.decode(rle)
        assert m.shape == (2048, 2048) and m.dtype == np.uint8
        # canonical re-encode
        assert mu.encode(np.asfortranarray(m))["counts"].decode() == cnt
        areas.append(int(m.sum()))
        o = int((m.astype(bool) & ~dm).sum())
        if o: outside += 1; out_px += o
        acc += m
        import cv2
        ncomp.append(cv2.connectedComponents(m, connectivity=8)[0] - 1)
    overl += int((acc > 1).sum())
areas = np.array(areas); ncomp = np.array(ncomp)
print("overlapping pixels total", overl)
print("area min", areas.min(), "p1", np.percentile(areas, 1), "median", np.median(areas), "max", areas.max(), "n<100", (areas < 100).sum(), "zero", (areas == 0).sum())
print("masks with pixels outside disk", outside, "total px", out_px)
print("non-contiguous numbering stems", noncontig)
print("components per mask: 1:", (ncomp == 1).sum(), ">1:", (ncomp > 1).sum(), "max", ncomp.max())
