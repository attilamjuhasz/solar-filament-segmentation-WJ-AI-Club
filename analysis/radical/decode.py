"""Idea (a): expected-PQ-optimal decoding over the existing v2 candidates.

Decoders: greedy (= v2 assemble) vs exact max-weight independent set (MWIS) on the overlap graph, and
per-candidate mask-extent choice among thr {.3,.4,.5,.6,.7}.
Scores: model q*mean_p (v2), realized-y oracle, and LOO-annotator oracle (score from OTHER readings, evaluated on the
held-out reading; multi-reading stems only). All evaluation exact (ownership trimming, per-reading PQ counts).
"""
import os, sys, pickle, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "analysis/score_agent"))
from ana import halves, pq, cnt
from assemble import load_cands, FastGT

T = pickle.load(open(os.path.join(HERE, "table.pkl"), "rb"))
A, B = T["A"], T["B"]; stems = T["stems"]; THRS = T["thrs"]
C = load_cands(os.path.join(ROOT, "runs/s2_r34/cands_val_tta_last"), stems)
G = FastGT(stems)
R = {}
for r in T["rows"]:
    R.setdefault(r["stem"], []).append(r)
NR = {s: len(G.by_stem[s]) for s in stems}
MULTI = [s for s in stems if NR[s] > 1]
P = dict(a_min=100, own_frac=0.8, levels="AP")


def yv(r, t=0.5):
    i = r[f"iou_{t}"]
    return i * (i > 0.5)


def stats_reading(stem, ri, final):
    lab, areas = G.by_stem[stem][ri]
    S = TP = FP = 0
    n_gt = int((areas[1:] > 0).sum()); matched = set()
    for x, y, m in final:
        h, w = m.shape
        inter = np.bincount(lab[y:y + h, x:x + w][m], minlength=256); inter[0] = 0
        iou = inter / np.maximum(m.sum() + areas - inter, 1)
        j = int(iou.argmax())
        if iou[j] > 0.5:
            S += iou[j]; TP += 1; matched.add(j)
        else:
            FP += 1
    return np.array([S, TP, FP, n_gt - len(matched)])


def cmask(c, t):
    return c["soft"] > int(t * 255)


def assemble_sel(stem, sel):
    """sel: list of (score, k, thr) already decided; assemble in score order with ownership trimming (v2 rules)."""
    owner = np.zeros((2048, 2048), bool)
    out = []
    for s, k, t in sorted(sel, key=lambda z: -z[0]):
        c = C[stem][k]
        m = cmask(c, t); area = int(m.sum())
        if area < P["a_min"]:
            continue
        x, y = c["x"], c["y"]; h, w = m.shape
        free = m & ~owner[y:y + h, x:x + w]
        if free.sum() < P["own_frac"] * area or free.sum() < P["a_min"]:
            continue
        owner[y:y + h, x:x + w] |= free
        out.append((x, y, free))
    return out


_OV = {}


def overlap(stem, i, ti, j, tj):
    key = (stem, i, ti, j, tj)
    if key not in _OV:
        ci, cj = C[stem][i], C[stem][j]
        mi, mj = cmask(ci, ti), cmask(cj, tj)
        x0, y0 = max(ci["x"], cj["x"]), max(ci["y"], cj["y"])
        x1 = min(ci["x"] + mi.shape[1], cj["x"] + mj.shape[1]); y1 = min(ci["y"] + mi.shape[0], cj["y"] + mj.shape[0])
        n = 0
        if x1 > x0 and y1 > y0:
            n = int((mi[y0 - ci["y"]:y1 - ci["y"], x0 - ci["x"]:x1 - ci["x"]] & mj[y0 - cj["y"]:y1 - cj["y"], x0 - cj["x"]:x1 - cj["x"]]).sum())
        _OV[key] = n
    return _OV[key]


def mwis(stem, items, frac=0.2):
    """items: list of (w, score, k, t, area) with w>0. Exact MWIS on conflict graph (overlap > frac*min area);
    items with same k are always in conflict (alternative extents). Components > 22 nodes -> greedy."""
    n = len(items)
    adj = [0] * n
    for a in range(n):
        for b in range(a + 1, n):
            ka, kb = items[a][2], items[b][2]
            conf = ka == kb
            if not conf:
                ov = overlap(stem, ka, items[a][3], kb, items[b][3]) if ka < kb else overlap(stem, kb, items[b][3], ka, items[a][3])
                conf = ov > frac * min(items[a][4], items[b][4])
            if conf:
                adj[a] |= 1 << b; adj[b] |= 1 << a
    seen = 0; chosen = []
    for st in range(n):
        if seen >> st & 1:
            continue
        comp = [st]; seen |= 1 << st; q = [st]
        while q:
            u = q.pop()
            for v in range(n):
                if adj[u] >> v & 1 and not seen >> v & 1:
                    seen |= 1 << v; comp.append(v); q.append(v)
        if len(comp) > 22:
            comp.sort(key=lambda u: -items[u][1]); taken = 0
            for u in comp:
                if not adj[u] & taken:
                    taken |= 1 << u; chosen.append(u)
            continue
        best = (0.0, [])
        def rec(idx, avail_mask, cur_w, cur):
            nonlocal best
            if cur_w + sum(items[comp[z]][0] for z in range(idx, len(comp)) if avail_mask >> comp[z] & 1) <= best[0]:
                return
            if idx == len(comp):
                if cur_w > best[0]:
                    best = (cur_w, list(cur))
                return
            u = comp[idx]
            if avail_mask >> u & 1:
                rec(idx + 1, avail_mask & ~adj[u] & ~(1 << u), cur_w + items[u][0], cur + [u])
            rec(idx + 1, avail_mask & ~(1 << u), cur_w, cur)
        rec(0, (1 << n) - 1, 0.0, [])
        chosen += best[1]
    return [items[u] for u in chosen]


def decode(stem, score_of, lam, how="greedy", thrs=(0.5,), ext_score=None):
    """score_of(k, t) -> expected y; returns final masks."""
    items = []
    for k, r in enumerate(R[stem]):
        if r["level"] not in P["levels"]:
            continue
        for t in thrs:
            s = score_of(k, t)
            if s >= lam:
                items.append((s - lam, s, k, t, r[f"area_{t}"]))
    if how == "greedy":
        # best extent per candidate, then greedy (v2)
        bestk = {}
        for it in items:
            if it[2] not in bestk or it[1] > bestk[it[2]][1]:
                bestk[it[2]] = it
        sel = [(it[1], it[2], it[3]) for it in bestk.values()]
    else:
        sel = [(it[1], it[2], it[3]) for it in mwis(stem, items)]
    return assemble_sel(stem, sel)


def run(name, mk_score, lam, how="greedy", thrs=(0.5,), stems_=None, loo=False):
    stems_ = stems_ or stems
    tA = np.zeros(4); tB = np.zeros(4)
    for s in stems_:
        if loo:
            for ri in range(NR[s]):
                fin = decode(s, mk_score(s, ri), lam, how, thrs)
                v = stats_reading(s, ri, fin)
                if s in A: tA += v
                else: tB += v
        else:
            fin = decode(s, mk_score(s, None), lam, how, thrs)
            v = sum(stats_reading(s, ri, fin) for ri in range(NR[s]))
            if s in A: tA += v
            else: tB += v
    print(f"{name:66s} lam={lam:.3f} A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} {cnt(tA + tB)}", flush=True)
    return pq(tA + tB)


def model_score(s, ri):
    return lambda k, t: R[s][k]["q"] * R[s][k]["mean_p"]


def real_score(s, ri):
    return lambda k, t: float(yv(R[s][k], t).mean())


def loo_score(s, ri):
    def f(k, t):
        v = yv(R[s][k], t)
        return float(np.delete(v, ri).mean())
    return f


def loo_blend(alpha):
    def mk(s, ri):
        def f(k, t):
            v = yv(R[s][k], t)
            return alpha * float(np.delete(v, ri).mean()) + (1 - alpha) * R[s][k]["q"] * R[s][k]["mean_p"]
        return f
    return mk


def dink(name, mk, how="greedy", thrs=(0.5,), stems_=None, loo=False, lam0=0.228):
    lam = lam0
    for it in range(4):
        v = run(name + f" [it{it}]", mk, lam, how, thrs, stems_, loo)
        if abs(v / 2 - lam) < 0.003:
            break
        lam = v / 2
    return v


if __name__ == "__main__":
    print("=== full val (141 stems), model score q*mean_p ===")
    run("v2 greedy (sanity: should be ~.456)", model_score, 0.225)
    run("MWIS on q*mean_p (raw)", model_score, 0.225, how="mwis")
    print("=== realized-y oracle (full val) ===")
    dink("oracle realized y greedy", real_score)
    dink("oracle realized y MWIS", real_score, how="mwis")
    dink("oracle realized y MWIS + extent {.3..7}", real_score, how="mwis", thrs=THRS)
    print(f"=== multi-reading stems only ({len(MULTI)} stems, {sum(NR[s] for s in MULTI)} readings) ===")
    run("v2 model greedy", model_score, 0.225, stems_=MULTI)
    run("v2 model MWIS", model_score, 0.225, how="mwis", stems_=MULTI)
    dink("realized oracle greedy", real_score, stems_=MULTI)
    dink("LOO-annotator oracle greedy (thr .5)", loo_score, stems_=MULTI, loo=True)
    dink("LOO-annotator oracle MWIS (thr .5)", loo_score, how="mwis", stems_=MULTI, loo=True)
    dink("LOO-annotator oracle greedy + extent", loo_score, thrs=THRS, stems_=MULTI, loo=True)
    dink("LOO-annotator oracle MWIS + extent", loo_score, how="mwis", thrs=THRS, stems_=MULTI, loo=True)
    for a in (0.25, 0.5, 0.75):
        dink(f"blend {a}*LOO + {1 - a}*model greedy", loo_blend(a), stems_=MULTI, loo=True)
