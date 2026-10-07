"""Guard check between two submissions: kept filaments per image and instance agreement (IoU > .5).

python scripts/exp/compare_subs.py submissions/robust.csv submissions/aggressive.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as mu


def load(p):
    out = defaultdict(list)
    with open(p) as f:
        for fid, counts in list(csv.reader(f))[1:]:
            out[fid.rpartition("_")[0]].append({"size": [2048, 2048], "counts": counts.encode()})
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
na, nb = sum(map(len, a.values())), sum(map(len, b.values()))
match = 0
for s, ra in a.items():
    rb = b.get(s, [])
    if ra and rb:
        match += int((np.asarray(mu.iou(ra, rb, [0] * len(rb))) > 0.5).any(1).sum())
print(f"kept: {na} vs {nb} ({(nb - na) / na:+.1%}); per image {na / 180:.2f} vs {nb / 180:.2f}")
print(f"instances of A matched in B (IoU>.5): {match / na:.1%}")
ok = abs(nb - na) / na <= 0.05 and match / na >= 0.85
print("GUARD", "OK" if ok else "FAIL (consider matching lam to A's kept count)")
