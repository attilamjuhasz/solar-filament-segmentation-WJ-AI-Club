import os, sys, json, glob, numpy as np
sys.path.insert(0, "src")
from assemble import load_cands, FastGT, assemble, run_pq
from metric import load_gt_rles, evaluate
from rle import _full
from common import DATA
P = json.load(open("configs/assemble_v2.json"))
vd = "runs/s2_r34/cands_val_tta_last"; td = "runs/s2_r34/cands_test_tta_last"
vs = sorted(f[:-4] for f in os.listdir(vd) if f.endswith(".pkl"))
ts = sorted(f[:-4] for f in os.listdir(td) if f.endswith(".pkl"))
vc = load_cands(vd, vs)
gt = FastGT(vs)
finals = {s: assemble(c, P) for s, c in vc.items()}
print("FastGT", gt.pq(finals))
ann = glob.glob(os.path.join(DATA, "**", "*.json"), recursive=True)
print(ann)
g = load_gt_rles(ann[0], set(vs))
print("readings", len(g))
pq, info = evaluate({s: [_full(m) for m in f] for s, f in finals.items()}, g)
print("official replica", pq, info)
# val vs test candidate stats
tc = load_cands(td, ts)
def stats(cc):
    allc = [c for v in cc.values() for c in v]
    lv = {}
    for c in allc: lv[c["level"]] = lv.get(c["level"], 0) + 1
    q = np.array([c["q"] for c in allc]); mp = np.array([c["mean_p"] for c in allc]); a = np.array([c["area50"] for c in allc])
    fin = [assemble(c, P) for c in cc.values()]
    return dict(per_stem=len(allc) / len(cc), levels={k: round(v / len(cc), 2) for k, v in sorted(lv.items())},
                q_mean=q.mean().round(3))
def stats2(cc):
    allc = [c for v in cc.values() for c in v]
    lv = {}
    for c in allc: lv[c["level"]] = lv.get(c["level"], 0) + 1
    q = np.array([c["q"] for c in allc]); mp = np.array([c["mean_p"] for c in allc]); a = np.array([c["area50"] for c in allc])
    fin = [assemble(c, P) for c in cc.values()]
    nf = np.array([len(f) for f in fin]); fa = np.array([m.sum() for f in fin for _, _, m in f])
    return dict(cands_per_stem=round(len(allc) / len(cc), 2), levels={k: round(v / len(cc), 2) for k, v in sorted(lv.items())},
                q_mean=round(q.mean(), 3), qmp_mean=round((q * mp).mean(), 3), area50_med=np.median(a),
                final_per_stem=round(nf.mean(), 2), empty_stems=int((nf == 0).sum()), final_area_med=np.median(fa))
print("VAL ", stats2(vc))
print("TEST", stats2(tc))
