import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from common import TEST_IMG, load_meta

meta = load_meta()
meta["stem"] = meta.file_name.str[:-5]
st = meta.drop_duplicates("stem").copy()
st["t"] = pd.to_datetime(st.stem.str[:14], format="%Y%m%d%H%M%S")
test = sorted(f[:-5] for f in os.listdir(TEST_IMG))
tt = pd.to_datetime(pd.Series(test).str[:14], format="%Y%m%d%H%M%S")
trt = np.sort(st.t.values)


def nearest_days(ts, ref):
    out = []
    for t in ts:
        out.append(np.abs((ref - t) / np.timedelta64(1, "h")).min() / 24)
    return np.array(out)


nt = nearest_days(tt.values, trt)
print("TEST -> nearest train stem (days): same-date frac",
      np.mean([d in set(st.t.dt.date) for d in tt.dt.date]).round(3),
      "| <=0.25d %.2f <=1d %.2f <=2d %.2f <=3d %.2f" % tuple((nt <= x).mean() for x in (0.25, 1, 2, 3)),
      "median %.2f" % np.median(nt))
for f in range(5):
    va = st[st.fold == f].t.values
    tr = np.sort(st[st.fold != f].t.values)
    nv = nearest_days(va, tr)
    print(f"VAL f{f} -> nearest train stem: <=0.25d %.2f <=1d %.2f <=2d %.2f <=3d %.2f median %.2f" % (
        *(tuple((nv <= x).mean() for x in (0.25, 1, 2, 3))), np.median(nv)))
# cross-fold pairs < 6h
s = st.sort_values("t").reset_index(drop=True)
dh = s.t.diff().dt.total_seconds() / 3600
for i in np.where((dh < 6) & (s.fold != s.fold.shift()))[0]:
    print("cross-fold near pair:", s.stem[i - 1], s.fold[i - 1], s.stem[i], s.fold[i], f"{dh[i]:.1f}h")
print("test year dist", tt.dt.year.value_counts().sort_index().to_dict())
print("train year dist", st.t.dt.year.value_counts().sort_index().to_dict())
