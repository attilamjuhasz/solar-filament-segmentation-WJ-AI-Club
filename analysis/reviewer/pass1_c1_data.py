import json, os, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from common import DATA, CACHE, load_meta

d = json.load(open(os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")))
print("image keys", d["images"][0].keys())
print("ann keys", d["annotations"][0].keys())
print("sample date_captured", [im["date_captured"] for im in d["images"][:3]])
exts = Counter(im["file_name"].rsplit(".", 1)[1] for im in d["images"])
print("exts", exts, "all end .jpeg", all(im["file_name"].endswith(".jpeg") for im in d["images"]))
sizes = Counter((im["height"], im["width"]) for im in d["images"]); print("sizes", sizes)
n_ann = Counter(a["image_id"] for a in d["annotations"])
zero = [im["id"] for im in d["images"] if n_ann[im["id"]] == 0]
print("readings with 0 annotations:", len(zero), zero[:5])
# polygon formats
plen = Counter()
types = Counter()
for a in d["annotations"]:
    s = a["segmentation"]
    types[type(s).__name__ + ("/" + type(s[0]).__name__ if isinstance(s, list) and s else "")] += 1
    if isinstance(s, list):
        plen[min(len(p) for p in s)] += 0
        for p in s:
            if len(p) <= 6:
                plen[len(p)] += 1
        if len(s[0]) == 4:
            print("FIRST POLY HAS 4 COORDS -> frPyObjects treats as bbox!", a["id"])
print("seg types", types, "short polys (<=6 coords):", {k: v for k, v in plen.items() if v})
print("n polys per ann", Counter(len(a["segmentation"]) for a in d["annotations"]))
# image id vs stem consistency
bad = [im for im in d["images"] if im["id"].split("-", 1)[1] != im["file_name"][:-5]]
print("id/stem mismatch", len(bad))
# date vs stem
badd = [im for im in d["images"] if im["date_captured"][:10].replace("-", "") != im["file_name"][:8]]
print("date_captured != stem date", len(badd), [(im["date_captured"], im["file_name"]) for im in badd[:5]])

meta = load_meta()
meta["stem"] = meta.file_name.str[:-5]
print("stems in >1 fold:", (meta.groupby("stem").fold.nunique() > 1).sum())
print("dates in >1 fold:", (meta.groupby("date").fold.nunique() > 1).sum())
print(meta.groupby("fold").agg(stems=("stem", "nunique"), readings=("image_id", "size"), inst=("n_inst", "sum")))
# adjacent-date leakage
dts = meta.drop_duplicates("date")[["date", "fold"]].copy()
dts["dt"] = pd.to_datetime(dts.date)
dts = dts.sort_values("dt").reset_index(drop=True)
gap = dts.dt.diff().dt.days
print("n dates", len(dts), "gap days pct", np.nanpercentile(gap, [0, 10, 25, 50, 75, 90]))
f2d = {f: set(g.dt) for f, g in dts.groupby("fold")}
for f in range(5):
    va = dts[dts.fold == f]
    tr = dts[dts.fold != f]
    trd = np.array(sorted(tr.dt.values))
    near = []
    for t in va.dt.values:
        dd = np.abs((trd - t).astype("timedelta64[h]").astype(float)) / 24
        near.append(dd.min())
    near = np.array(near)
    print(f"fold {f}: val dates {len(va)}; nearest train date <=1d: {(near<=1).mean():.2f}, <=2d: {(near<=2).mean():.2f}, <=3d {(near<=3).mean():.2f}")
# stems near midnight UTC
tm = pd.to_datetime(meta.drop_duplicates("stem").stem.str[:14], format="%Y%m%d%H%M%S")
st = meta.drop_duplicates("stem").assign(t=tm.values).sort_values("t").reset_index(drop=True)
dt_h = st.t.diff().dt.total_seconds() / 3600
cross = ((dt_h < 6) & (st.fold != st.fold.shift())).sum()
print("consecutive stems <6h apart in different folds:", int(cross), "of", int((dt_h < 6).sum()), "pairs <6h")
cross = ((dt_h < 24) & (st.fold != st.fold.shift())).sum()
print("consecutive stems <24h apart in different folds:", int(cross), "of", int((dt_h < 24).sum()), "pairs <24h")
