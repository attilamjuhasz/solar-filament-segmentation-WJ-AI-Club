"""Multi-frame idea: fetch GONG archive JPEGs (public, unlabeled) close in time to each fold-0 val frame.
same-site frames within +-3 min (up to 4) and other-site frames within +-3 min (up to 2)."""
import os, re, sys, pickle, subprocess, time
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solgeo import stem_ts

T = pickle.load(open(os.path.join(HERE, "table.pkl"), "rb"))
stems = T["stems"]
if len(sys.argv) > 1:
    stems = stems[int(sys.argv[1])::int(sys.argv[2])]
OUT = os.path.join(HERE, "gong")
os.makedirs(OUT, exist_ok=True)
listing = {}


def day_list(ts):
    key = ts.strftime("%Y%m%d")
    if key not in listing:
        url = f"https://gong2.nso.edu/HA/hag/{ts.strftime('%Y%m')}/{key}/"
        html = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True).stdout
        listing[key] = sorted(set(re.findall(r'href="(\d{14}[A-Z]h\.jpg)"', html)))
    return listing[key]


plan = {}
for s in stems:
    ts = stem_ts(s)
    files = day_list(ts)
    same, other = [], []
    for f in files:
        st = f[:-4]
        if st == s:
            continue
        dt = abs((stem_ts(st) - ts).total_seconds()) / 60
        if dt <= 3.01:
            (same if st[-2] == s[-2] else other).append((dt, st))
    sel = [z[1] for z in sorted(same)[:4]] + [z[1] for z in sorted(other)[:2]]
    plan[s] = sel
    d = os.path.join(OUT, s); os.makedirs(d, exist_ok=True)
    for st in sel:
        p = os.path.join(d, st + ".jpg")
        if os.path.exists(p) and os.path.getsize(p) > 100000:
            continue
        url = f"https://gong2.nso.edu/HA/hag/{ts.strftime('%Y%m')}/{st[:8]}/{st}.jpg"
        subprocess.run(["curl", "-s", "-m", "120", "-o", p, url])
    print(s, len(same), len(other), sel, flush=True)
pickle.dump(plan, open(os.path.join(HERE, f"mf_plan_{sys.argv[1] if len(sys.argv) > 1 else 'all'}.pkl"), "wb"))
print("done")
