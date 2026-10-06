"""Per-annotator PQ of v2 on fold-0 val, plus per-annotator drawing style on all train readings."""
from decode import *
import pandas as pd
from common import load_meta
m = load_meta(); m["stem"] = m.file_name.str[:-5]
rid = {s: list(m[m.stem == s].image_id) for s in stems}
# FastGT order == meta order filtered; verify by re-deriving areas
acc = {}
for s in stems:
    fin = decode(s, model_score(s, None), 0.225)
    for ri in range(NR[s]):
        a = rid[s][ri].split("-")[0]
        for key in (a[:4], "ALL"):
            acc[key] = acc.get(key, np.zeros(4)) + stats_reading(s, ri, fin)
rows = sorted(((k, pq(v), int(v[1] + v[3]), int(v[1]), int(v[2]), int(v[3])) for k, v in acc.items()), key=lambda r: -r[2])
print("group  PQ    n_gt  tp  fp  fn")
for r in rows:
    print(f"{r[0]:6s} {r[1]:.3f} {r[2]:5d} {r[3]:4d} {r[4]:4d} {r[5]:4d}")
