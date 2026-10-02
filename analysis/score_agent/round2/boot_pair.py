"""Paired stem bootstrap of PQ(new) - PQ(v2 on s2_r34). usage: boot_pair.py NEWDIR [mode]  mode: plain|ens"""
import sys
from base import *
from ana import Ctx
import assemble as AS
NEW = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else "plain"
co = Ctx(cdir=os.path.join(ROOT, "runs/s2_r34/cands_val_tta_last"), load_s1=False)
cn = Ctx(cdir=os.path.join(ROOT, NEW), load_s1=False)
if MODE == "ens":
    qo = {(s, c["idx"], c["level"]): c["q"] for s in co.stems for c in co.C[s]}
    for s in cn.stems:
        for c in cn.C[s]:
            c["q"] = 0.5 * (c["q"] + qo.get((s, c["idx"], c["level"]), c["q"]))
a = np.array([co.G.stats(s, assemble(co.C[s], V2)) for s in co.stems], float)
b = np.array([cn.G.stats(s, assemble(cn.C[s], V2)) for s in cn.stems], float)
pqf = lambda t: t[0] / (t[1] + .5 * t[2] + .5 * t[3])
rng = np.random.default_rng(0)
d = []
for _ in range(4000):
    i = rng.integers(0, len(a), len(a))
    d.append(pqf(b[i].sum(0)) - pqf(a[i].sum(0)))
d = np.array(d)
print(f"{NEW} [{MODE}]: PQ new {pqf(b.sum(0)):.4f} vs v2 {pqf(a.sum(0)):.4f}; delta {pqf(b.sum(0)) - pqf(a.sum(0)):+.4f}, paired bootstrap SE {d.std():.4f}, P(delta>0) {np.mean(d > 0):.2f}")
