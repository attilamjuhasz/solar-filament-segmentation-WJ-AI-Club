"""Shared helpers for the stage-2 candidate analysis (fold-0 val, split-half by date)."""
import json
import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
from assemble import FastGT, assemble, cand_mask, load_cands  # noqa: E402,F401
from common import load_meta  # noqa: E402
from s1 import postprocess_s1  # noqa: E402

CDIR = os.path.join(ROOT, os.environ.get("CANDS", "runs/s2_r34/cands_val_tta_ep4"))
TAG = os.environ.get("TAG", "ep4")
PDIR = os.path.join(ROOT, "runs/s1_r34_f0/probs_tta")
KEYS = ["score", "thr", "ring", "rel", "lam", "lam_small", "small", "a_min", "own_frac", "levels", "grow"]
BEST = {k: v for k, v in json.load(open(os.path.join(ROOT, "runs/s2_r34/cands_val_tta_ep4/best_params.json"))).items() if k in KEYS}
S1BEST = dict(t_hi=0.6, t_lo=0.45, gap=8, min_area=400, head=0)
SIZE_BINS = [0, 200, 400, 1000, 3000, 8000, 10**9]


def halves():
    """Fold-0 stems split by observation date (alternate sorted dates) -> (A, B)."""
    m = load_meta()
    m = m[m.fold == 0].assign(stem=m.file_name.str[:-5]).drop_duplicates("stem")
    dates = sorted(m.date.unique())
    da = set(dates[0::2])
    A = sorted(m[m.date.isin(da)].stem)
    B = sorted(m[~m.date.isin(da)].stem)
    return A, B


class Ctx:
    def __init__(self, cdir=CDIR, load_s1=True):
        self.A, self.B = halves()
        self.stems = sorted(self.A + self.B)
        self.C = load_cands(cdir, self.stems)
        self.G = FastGT(self.stems)
        self.probs = {}
        if load_s1:
            for s in self.stems:
                self.probs[s] = np.load(os.path.join(PDIR, s + ".npy"))

    # ---- per-object per-reading IoUs
    def obj_ious(self, stem, x, y, m):
        """-> list over readings of (best_iou, best_label, inter_vector) for one local mask."""
        out = []
        for lab, areas in self.G.by_stem[stem]:
            h, w = m.shape
            inter = np.bincount(lab[y:y + h, x:x + w][m], minlength=256)
            inter[0] = 0
            iou = inter / np.maximum(m.sum() + areas - inter, 1)
            j = int(iou.argmax())
            out.append((float(iou[j]), j, inter))
        return out

    def pq_counts(self, finals, stems):
        t = np.zeros(4)
        for s in stems:
            t += self.G.stats(s, finals.get(s, []))
        return t

    def report(self, finals_fn, name=""):
        """finals_fn(stem) -> [(x, y, m)]. Prints PQ on A, B and all."""
        fin = {s: finals_fn(s) for s in self.stems}
        tA, tB = self.pq_counts(fin, self.A), self.pq_counts(fin, self.B)
        r = dict(A=pq(tA), B=pq(tB), all=pq(tA + tB))
        if name:
            print(f"{name:55s} A={r['A']:.4f} B={r['B']:.4f} all={r['all']:.4f}  {cnt(tA + tB)}", flush=True)
        return r, fin

    def s1_inst(self, stem, **P):
        return postprocess_s1(self.probs[stem], stem, **P)


def pq(t):
    S, TP, FP, FN = t
    d = TP + 0.5 * FP + 0.5 * FN
    return S / d if d else 1.0


def cnt(t):
    return f"tp={int(t[1])} fp={int(t[2])} fn={int(t[3])} sq={t[0] / max(t[1], 1):.3f}"


def assemble_prov(cands, P, score_fn=None, extra_gate=None):
    """assemble() clone that also returns the source candidate of each final mask (no grow)."""
    owner = np.zeros((2048, 2048), bool)
    scored = []
    for c in cands:
        if c["level"] not in P["levels"]:
            continue
        m, area, mprob = cand_mask(c, P)
        s = score_fn(c, m, area, mprob) if score_fn else c["q"]
        scored.append((s, area, m, c))
    scored.sort(key=lambda t: -t[0])
    final, prov = [], []
    for s, area, m, c in scored:
        if area < P["a_min"]:
            continue
        if s < (P["lam_small"] if area < P["small"] else P["lam"]):
            continue
        if extra_gate is not None and not extra_gate(c, m, area, s):
            continue
        x, y = c["x"], c["y"]
        h, w = m.shape
        free = m & ~owner[y:y + h, x:x + w]
        if free.sum() < P["own_frac"] * area or free.sum() < P["a_min"]:
            continue
        owner[y:y + h, x:x + w] |= free
        final.append((x, y, free))
        prov.append((c, s))
    return final, prov
