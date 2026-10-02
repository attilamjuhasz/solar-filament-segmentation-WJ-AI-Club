"""Assembly: refined candidates -> final non-overlapping instances; fast PQ for tuning; submission.

Each candidate carries a soft native-res probability crop; its mask is `soft > thr` (optionally
grown into a `ring` band: dilate(soft > thr) & (soft > ring), or a per-candidate relative threshold
rel * p95(soft)). Greedy by score q (= predicted E[IoU * 1(IoU > .5)]): accept if q >= lam (lam_small
for small masks) and area >= a_min, claim only unowned pixels, drop it if it keeps < own_frac of its
pixels. Disjoint by construction.

Tune:    python src/assemble.py tune --cands runs/s2_r34/cands_val_tta_last
Submit:  python src/assemble.py submit --cands runs/s2_r34/cands_test_tta_last --params runs/.../best_params.json
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RUNS, TEST_IMG, load_inst, load_meta  # noqa: E402

DEFAULT = dict(score="q", thr=0.5, ring=0.0, rel=0.0, lam=0.2, lam_small=0.2, small=400, a_min=100,
               own_frac=0.6, levels="APBC", grow=0)
_K3 = np.ones((3, 3), np.uint8)


def load_cands(cdir, stems):
    out = {}
    for s in stems:
        cs = pickle.load(open(os.path.join(cdir, s + ".pkl"), "rb"))
        for c in cs:
            nz = c["soft"][c["soft"] > 0]
            c["p95"] = float(np.percentile(nz, 95)) / 255 if nz.size else 0.0
            c["_cache"] = {}
        out[s] = cs
    return out


def cand_mask(c, P):
    key = (P["thr"], P["ring"], P["rel"])
    if key not in c["_cache"]:
        thr = max(P["thr"], P["rel"] * c["p95"]) if P["rel"] > 0 else P["thr"]
        m = c["soft"] > int(thr * 255)
        if 0 < P["ring"] < thr:
            m = (cv2.dilate(m.astype(np.uint8), _K3) > 0) & (c["soft"] > int(P["ring"] * 255))
        c["_cache"][key] = (m, int(m.sum()), float(c["soft"][m].mean()) / 255 if m.any() else 0.0)
    return c["_cache"][key]


def cand_score(c, mode, mprob):
    if mode == "q":
        return c["q"]
    if mode == "qm":
        return c["q"] * mprob
    if mode == "qpu":
        return c["q"] * (0.5 + 0.5 * c["peak_u"])
    if mode == "qmp":  # S2 q x S1 precise-head mean prob inside the proposal: the two signals are complementary
        return c["q"] * c["mean_p"]
    raise ValueError(mode)


def assemble(cands, P):
    owner = np.zeros((2048, 2048), bool)
    final = []
    scored = []
    for c in cands:
        if c["level"] not in P["levels"]:
            continue
        m, area, mprob = cand_mask(c, P)
        scored.append((cand_score(c, P["score"], mprob), area, m, c))
    scored.sort(key=lambda t: -t[0])
    for s, area, m, c in scored:
        if area < P["a_min"]:
            continue
        if s < (P["lam_small"] if area < P["small"] else P["lam"]):
            continue
        x, y = c["x"], c["y"]
        h, w = m.shape
        free = m & ~owner[y:y + h, x:x + w]
        if free.sum() < P["own_frac"] * area or free.sum() < P["a_min"]:
            continue
        owner[y:y + h, x:x + w] |= free
        final.append((x, y, free))
    if P.get("grow", 0) > 0:  # dilate each mask into unowned pixels (thin GT outlines tend to be wider)
        k = np.ones((2 * P["grow"] + 1,) * 2, np.uint8)
        grown = []
        for x, y, m in final:
            g = P["grow"]
            x0, y0 = max(x - g, 0), max(y - g, 0)
            x1, y1 = min(x + m.shape[1] + g, 2048), min(y + m.shape[0] + g, 2048)
            big = np.zeros((y1 - y0, x1 - x0), np.uint8)
            big[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] = m
            d = cv2.dilate(big, k).astype(bool) & ~owner[y0:y1, x0:x1]
            d |= big.astype(bool)
            owner[y0:y1, x0:x1] |= d
            grown.append((x0, y0, d))
        final = grown
    return final


# ------------------------------------------------------------------------------- fast evaluation
class FastGT:
    """Per-reading GT label maps (uint8) + areas for bincount-based PQ on local masks."""

    def __init__(self, stems):
        meta = load_meta()
        meta = meta[meta.file_name.str[:-5].isin(set(stems))]
        self.by_stem = {}
        for rid, fn in zip(meta.image_id, meta.file_name):
            lab = load_inst(rid)
            assert lab.max() < 255
            self.by_stem.setdefault(fn[:-5], []).append((lab.astype(np.uint8), np.bincount(lab.ravel(), minlength=256)))

    def stats(self, stem, final):
        S = TP = FP = FN = 0
        for lab, areas in self.by_stem.get(stem, []):
            n_gt = int((areas[1:] > 0).sum())
            matched = set()
            for x, y, m in final:
                h, w = m.shape
                inter = np.bincount(lab[y:y + h, x:x + w][m], minlength=256)
                inter[0] = 0
                pa = m.sum()
                iou = inter / np.maximum(pa + areas - inter, 1)
                j = int(iou.argmax())
                if iou[j] > 0.5:
                    S += iou[j]
                    TP += 1
                    matched.add(j)
                else:
                    FP += 1
            FN += n_gt - len(matched)
        return S, TP, FP, FN

    def pq(self, finals):
        t = np.zeros(4)
        for stem in self.by_stem:
            t += self.stats(stem, finals.get(stem, []))
        S, TP, FP, FN = t
        d = TP + 0.5 * FP + 0.5 * FN
        return S / d if d else 1.0, dict(sq=S / max(TP, 1), tp=int(TP), fp=int(FP), fn=int(FN))


def run_pq(cands, gt, P):
    return gt.pq({s: assemble(c, P) for s, c in cands.items()})


def tune(cands, gt, P0=None, grid=None, rounds=2, log=print):
    P = dict(DEFAULT if P0 is None else P0)
    grid = grid or dict(
        thr=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        lam=[0.1, 0.13, 0.16, 0.2, 0.23, 0.26, 0.3, 0.35],
        a_min=[20, 60, 100, 200, 300, 400],
        ring=[0.0, 0.3, 0.35, 0.4],
        rel=[0.0, 0.4, 0.5, 0.6],
        score=["q", "qm", "qpu", "qmp"],
        lam_small=[0.13, 0.2, 0.26, 0.3, 0.4, 1.1],
        small=[200, 400, 800],
        own_frac=[0.4, 0.6, 0.8],
        levels=["APBC", "APB", "AP", "A", "ABC", "PBC"],
        grow=[0, 1],
    )
    best, info = run_pq(cands, gt, P)
    log(f"start {best:.4f} {info} {P}")
    for r in range(rounds):
        for k, vals in grid.items():
            for v in vals:
                if v == P[k]:
                    continue
                Q = dict(P, **{k: v})
                s, inf = run_pq(cands, gt, Q)
                if s > best + 1e-5:
                    best, P, info = s, Q, inf
                    log(f"  {k}={v} -> {best:.4f} {info}")
    return P, best, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "eval", "submit"])
    ap.add_argument("--cands", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-fold", type=int, default=0)
    a = ap.parse_args()
    stems = sorted(f[:-4] for f in os.listdir(a.cands) if f.endswith(".pkl"))
    P = json.load(open(a.params)) if a.params else dict(DEFAULT)
    if a.mode in ("tune", "eval"):
        meta = load_meta()
        val = set(meta[meta.fold == a.val_fold].file_name.str[:-5])
        assert set(stems) == val, f"candidates cover {len(set(stems) & val)}/{len(val)} val stems"
        cands = load_cands(a.cands, stems)
        gt = FastGT(stems)
        if a.mode == "eval":
            print(run_pq(cands, gt, P))
            return
        P, best, info = tune(cands, gt, P)
        print("BEST", best, info, P)
        json.dump(dict(P, val_pq=best, **info), open(os.path.join(a.cands, "best_params.json"), "w"), indent=1)
    else:
        from rle import validate_submission, write_submission
        test_stems = sorted(f.rsplit(".", 1)[0] for f in os.listdir(TEST_IMG) if f.endswith(".jpeg"))
        assert set(stems) == set(test_stems), "candidates must cover every test image"
        cands = load_cands(a.cands, stems)
        preds = {}
        for s in stems:
            masks = assemble(cands[s], P)
            if masks:
                preds[s] = masks
        out = a.out or os.path.join(os.path.dirname(os.path.dirname(__file__)), "submissions", "submission.csv")
        n = write_submission(out, preds)
        print("wrote", n, "rows ->", out, validate_submission(out, test_stems))


if __name__ == "__main__":
    main()
