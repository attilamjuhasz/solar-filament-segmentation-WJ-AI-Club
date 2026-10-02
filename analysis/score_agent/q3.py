"""Q3: assembly variants / hybrids with split-half (by date) cross-fitting.

For every method: coordinate ascent on half A -> score half B, and on B -> score A.
  cross-fit PQ = PQ(counts_B(params_A) + counts_A(params_B))   <- the honest number
  plus: params tuned on all val -> PQ on A / B / all (optimistic reference).
python q3.py M0 M1 ...
"""
import pickle
import sys
import time

import cv2
import numpy as np

from ana import BEST, S1BEST, TAG, Ctx, assemble, cand_mask, pq

T0 = time.time()
ctx = Ctx()
rows = pickle.load(open(f"q1_rows_{TAG}.pkl", "rb"))
it = iter(rows)
for s in ctx.stems:  # realized value yv = mean over readings of IoU*1[IoU>.5] (mask thr .5, unclipped)
    for c in ctx.C[s]:
        r = next(it)
        assert r[0] == s and r[2] == c["level"] and abs(r[3] - c["q"]) < 1e-9
        c["yv"] = r[8]
for s in ctx.stems:  # S1 head-0 native prob under each candidate crop (for mask fusion)
    up = cv2.resize(ctx.probs[s][0], (2048, 2048), interpolation=cv2.INTER_LINEAR)
    for c in ctx.C[s]:
        h, w = c["soft"].shape
        c["s1"] = up[c["y"]:c["y"] + h, c["x"]:c["x"] + w].copy()
    del up
S1PRESETS = {"best": S1BEST, "loose": dict(t_hi=0.5, t_lo=0.4, gap=8, min_area=200, head=0)}
S1I = {k: {s: ctx.s1_inst(s, **v) for s in ctx.stems} for k, v in S1PRESETS.items()}
print(f"setup {time.time() - T0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------- helpers
def pav_fit(x, y):
    o = np.argsort(x, kind="mergesort")
    sums, cnts, xs = [], [], []
    for xi, yi in zip(x[o], y[o]):
        sums.append(yi); cnts.append(1); xs.append(xi)
        while len(sums) > 1 and sums[-2] / cnts[-2] > sums[-1] / cnts[-1]:
            s_, c_, x_ = sums.pop(), cnts.pop(), xs.pop()
            sums[-1] += s_; cnts[-1] += c_; xs[-1] += x_
    return np.array(xs) / np.array(cnts), np.array(sums) / np.array(cnts)


def fit_calib(stems):
    model = {}
    for L in "APBC":
        q = np.array([c["q"] for s in stems for c in ctx.C[s] if c["level"] == L])
        y = np.array([c["yv"] for s in stems for c in ctx.C[s] if c["level"] == L])
        model[L] = pav_fit(q, y) if len(q) >= 10 else (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    return model


def fcal(model, c):
    if callable(model):
        return model(c)
    kx, ky = model[c["level"]]
    return float(np.interp(c["q"], kx, ky))


def inter(a, b):
    """a, b = (x, y, m) local masks -> intersection pixel count."""
    ax, ay, am = a
    bx, by, bm = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + am.shape[1], bx + bm.shape[1]), min(ay + am.shape[0], by + bm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).sum())


def grow(final, owner, g):
    if g <= 0:
        return final
    k = np.ones((2 * g + 1,) * 2, np.uint8)
    out = []
    for x, y, m in final:
        x0, y0 = max(x - g, 0), max(y - g, 0)
        x1, y1 = min(x + m.shape[1] + g, 2048), min(y + m.shape[0] + g, 2048)
        big = np.zeros((y1 - y0, x1 - x0), np.uint8)
        big[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] = m
        d = cv2.dilate(big, k).astype(bool) & ~owner[y0:y1, x0:x1]
        d |= big.astype(bool)
        owner[y0:y1, x0:x1] |= d
        out.append((x0, y0, d))
    return out


def greedy(items, P):
    """items: (score, x, y, m, area). Gate, claim unowned pixels, own_frac, then grow."""
    owner = np.zeros((2048, 2048), bool)
    out = []
    for sc, x, y, m, area in sorted(items, key=lambda t: -t[0]):
        if area < P["a_min"] or sc < (P.get("lam_small", P["lam"]) if area < P.get("small", 0) else P["lam"]):
            continue
        h, w = m.shape
        free = m & ~owner[y:y + h, x:x + w]
        fs = int(free.sum())
        if fs < P["own_frac"] * area or fs < P["a_min"]:
            continue
        owner[y:y + h, x:x + w] |= free
        out.append((x, y, free))
    return grow(out, owner, P.get("grow", 0))


def fused_mask(c, thr, w):
    key = ("fuse", thr, w)
    if key not in c["_cache"]:
        f = w * c["soft"].astype(np.float32) + (1 - w) * c["s1"].astype(np.float32)
        m = (f > thr * 255) & (c["soft"] > 0)
        c["_cache"][key] = (m, int(m.sum()))
    return c["_cache"][key]


def cand_items(s, P, model):
    items = []
    for c in ctx.C[s]:
        if c["level"] not in P["levels"]:
            continue
        if P.get("w", 1.0) < 1.0:
            m, area = fused_mask(c, P["thr"], P["w"])
        else:
            m, area, _ = cand_mask(c, dict(thr=P["thr"], ring=0.0, rel=0.0))
        if area == 0:
            continue
        items.append([fcal(model, c), c["x"], c["y"], m, area, c])
    return items


def parts_suppress(items, P):
    """Drop a candidate when >=2 mostly-inside, mutually disjoint, above-threshold candidates
    have a larger summed gain (score - lam) than its own."""
    lam = P["lam"]
    ok = [it for it in items if it[0] >= lam and it[4] >= P["a_min"]]
    keep = []
    for it in items:
        sc, x, y, m, area = it[:5]
        if sc < lam:
            continue
        parts = []
        for o in sorted(ok, key=lambda t: -t[0]):
            if o is it or o[4] >= area:
                continue
            io = inter((x, y, m), (o[1], o[2], o[3]))
            if io < 0.6 * o[4]:
                continue
            if any(inter((o[1], o[2], o[3]), (p[1], p[2], p[3])) > 0.2 * min(o[4], p[4]) for p in parts):
                continue
            parts.append(o)
        if len(parts) >= 2 and sum(p[0] - lam for p in parts) > sc - lam:
            continue
        keep.append(it)
    return keep


# --------------------------------------------------------------------------------------- methods
def M0(s, P, model):  # current src/assemble.py
    return assemble(ctx.C[s], P)


def M1(s, P, model):  # isotonic-calibrated q per level (fit on the tuning half)
    its = cand_items(s, P, model)
    if P.get("parts"):
        its = parts_suppress(its, P)
    return greedy([tuple(i[:5]) for i in its], P)


_S1MATCH = {}


def s1_scores(s, preset, model, levels, match_iou):
    """score of each S1 instance = calibrated score of the best-IoU S2 candidate (mask thr .5)."""
    key = (s, preset, levels, match_iou)
    if key not in _S1MATCH:
        insts = S1I[preset][s]
        res = []
        for x, y, m in insts:
            a = int(m.sum())
            best, bc = 0.0, None
            for c in ctx.C[s]:
                if c["level"] not in levels:
                    continue
                cm, ca, _ = cand_mask(c, dict(thr=0.5, ring=0.0, rel=0.0))
                if ca == 0:
                    continue
                io = inter((x, y, m), (c["x"], c["y"], cm))
                iou = io / max(a + ca - io, 1)
                if iou > best:
                    best, bc = iou, c
            res.append((bc if best >= match_iou else None, x, y, m, a))
        _S1MATCH[key] = res
    return [(None if c is None else fcal(model, c), x, y, m, a) for c, x, y, m, a in _S1MATCH[key]]


def M3(s, P, model):  # S1-only instances gated by the matched S2 candidate's calibrated score
    res = s1_scores(s, P["s1"], model, P["levels"], P["match_iou"])
    items = [((sc if sc is not None else P["d0"]), x, y, m, a) for sc, x, y, m, a in res]
    return greedy(items, P)


def M4(s, P, model):  # mixed pool: calibrated S2 cands + S1 instances (score beta * matched score)
    items = [tuple(i[:5]) for i in cand_items(s, P, model)]
    if P["beta"] > 0:
        for sc, x, y, m, a in s1_scores(s, P["s1"], model, P["levels"], P["match_iou"]):
            if sc is not None:
                items.append((P["beta"] * sc, x, y, m, a))
    return greedy(items, P)


# ---------------------------------------------------------------------------------------- tuning
def evaluate(method, P, model, stems):
    t = np.zeros(4)
    for s in stems:
        t += ctx.G.stats(s, method(s, P, model))
    return t


def ascent(method, fit, P0, grid, stems, rounds=2):
    model = fit(stems) if fit else None
    P = dict(P0)
    best = pq(evaluate(method, P, model, stems))
    for _ in range(rounds):
        imp = False
        for k, vals in grid.items():
            for v in vals:
                if v == P.get(k):
                    continue
                Q = dict(P, **{k: v})
                sc = pq(evaluate(method, Q, model, stems))
                if sc > best + 1e-5:
                    best, P, imp = sc, Q, True
        if not imp:
            break
    return P, model, best


RESULTS = {}


def split_report(name, method, fit, P0, grid, rounds=2):
    t = time.time()
    PA, mA, inA = ascent(method, fit, P0, grid, ctx.A, rounds)
    PB, mB, inB = ascent(method, fit, P0, grid, ctx.B, rounds)
    tB = evaluate(method, PA, mA, ctx.B)
    tA = evaluate(method, PB, mB, ctx.A)
    Pall, mall, _ = ascent(method, fit, P0, grid, ctx.stems, rounds)
    aA, aB = evaluate(method, Pall, mall, ctx.A), evaluate(method, Pall, mall, ctx.B)
    RESULTS[name] = dict(cross=pq(tA + tB), A_from_B=pq(tA), B_from_A=pq(tB), inA=inA, inB=inB,
                         all=pq(aA + aB), all_A=pq(aA), all_B=pq(aB), PA=PA, PB=PB, Pall=Pall)
    print(f"\n### {name}  [{time.time() - t:.0f}s]\n  cross-fit PQ={pq(tA + tB):.4f}  (A scored w/ B-params {pq(tA):.4f}, "
          f"B scored w/ A-params {pq(tB):.4f}) | in-half A {inA:.4f}, B {inB:.4f}\n"
          f"  tuned-on-all: all {pq(aA + aB):.4f}  A {pq(aA):.4f}  B {pq(aB):.4f}  tp/fp/fn={int((aA + aB)[1])}/{int((aA + aB)[2])}/{int((aA + aB)[3])}\n"
          f"  PA={PA}\n  PB={PB}\n  Pall={Pall}", flush=True)
    pickle.dump(RESULTS, open(f"q3_results_{TAG}.pkl", "wb"))


G0 = dict(thr=[0.4, 0.5, 0.55, 0.6, 0.65], lam=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45], a_min=[100, 200, 300, 400, 500],
          score=["q", "qm", "qpu"], lam_small=[0.2, 0.3, 0.4, 1.1], own_frac=[0.4, 0.6, 0.8],
          levels=["APBC", "APB", "AP", "A"], grow=[0, 1])
P1 = dict(thr=0.6, lam=0.22, a_min=400, own_frac=0.8, levels="AP", grow=0, small=0)
G1 = dict(thr=[0.45, 0.5, 0.55, 0.6, 0.65], lam=[0.14, 0.17, 0.2, 0.22, 0.25, 0.28, 0.32],
          a_min=[100, 200, 300, 400, 500], own_frac=[0.4, 0.6, 0.8], levels=["A", "AP", "APB", "APBC"], grow=[0, 1])

if __name__ == "__main__":
    todo = sys.argv[1:] or ["M0", "M1", "M1s", "M2", "M5", "M3", "M4"]
    if "M0" in todo:
        split_report("M0 current assemble (coordinator grid)", M0, None, dict(BEST), G0)
    if "M1" in todo:
        split_report("M1 isotonic per-level calibrated q", M1, fit_calib, P1, G1)
    if "M1s" in todo:
        split_report("M1s M1 + small-mask lam", M1, fit_calib, dict(P1, small=400, lam_small=0.22),
                     dict(G1, lam_small=[0.22, 0.28, 0.35, 1.1], a_min=[100, 200, 300]))
    if "M2" in todo:
        split_report("M2 M1 + merge-vs-parts rule", M1, fit_calib, dict(P1, parts=True), dict(G1, parts=[False, True]))
    if "M5" in todo:
        split_report("M5 M1 + S2/S1 soft fusion", M1, fit_calib, dict(P1, w=1.0),
                     dict(G1, w=[1.0, 0.85, 0.7, 0.6, 0.5]))
    if "M3" in todo:
        split_report("M3 S1-only instances gated by matched S2 score", M3, fit_calib,
                     dict(s1="best", levels="AP", match_iou=0.3, d0=0.0, lam=0.2, a_min=0, own_frac=0.0),
                     dict(s1=["best", "loose"], lam=[0.0, 0.1, 0.14, 0.17, 0.2, 0.22, 0.25, 0.28], match_iou=[0.3, 0.5],
                          d0=[0.0, 1.0], levels=["A", "AP"], a_min=[0, 300, 400, 500]))
    if "M4" in todo:
        split_report("M4 mixed pool S2 cands + S1 instances", M4, fit_calib,
                     dict(P1, s1="best", beta=1.0, match_iou=0.5),
                     dict(G1, beta=[0.0, 0.9, 1.0, 1.1], s1=["best", "loose"]))
