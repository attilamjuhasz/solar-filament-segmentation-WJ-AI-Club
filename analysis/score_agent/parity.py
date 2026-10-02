"""Metric parity: official notebook port vs src/metric.py evaluate vs src/assemble.py FastGT.pq.

python parity.py <scenario> [n_stems]   scenarios: first union perturb overlap empty nogt csv
"""
import gc
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

torch.set_num_threads(2)
ROOT = "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official as O  # noqa: E402
from assemble import FastGT, to_local  # noqa: E402
from common import DATA, TEST_IMG, load_meta  # noqa: E402
from metric import evaluate, load_gt_rles  # noqa: E402
from rle import validate_submission, write_submission  # noqa: E402

ANN = os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
OUT = os.path.dirname(os.path.abspath(__file__))


def gt_dataframe(stems):
    """Hidden-GT stand-in in the official format: filament_id = '<reading>_<k>', pycocotools RLE counts."""
    d = json.load(open(ANN))
    keep = {im["id"] for im in d["images"] if im["file_name"].rsplit(".", 1)[0] in stems}
    rows, k = [], defaultdict(int)
    for a in d["annotations"]:
        if a["image_id"] not in keep:
            continue
        k[a["image_id"]] += 1
        rle = mu.merge(mu.frPyObjects(a["segmentation"], 2048, 2048))
        rows.append((f"{a['image_id']}_{k[a['image_id']]}", rle["counts"].decode("ascii")))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def enc(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))


# ----------------------------------------------------------------------------------- scenarios
def readings_of(gt):
    by = defaultdict(list)
    for rid, (stem, rles) in sorted(gt.items()):
        by[stem].append((rid, rles))
    return by


def sc_first(by, rng):
    return {s: list(v[0][1]) for s, v in by.items()}


def sc_union(by, rng):
    out = {}
    for s, v in by.items():
        u = np.zeros((2048, 2048), np.uint8)
        for _, rles in v:
            if rles:
                u |= mu.decode(mu.merge(rles, intersect=False))
        n, lab = cv2.connectedComponents(u, connectivity=8)
        out[s] = [enc(lab == i) for i in range(1, n)]
    return out


def _perturb_one(m, rng):
    m = m.astype(np.uint8)
    r = rng.random()
    if r < 0.15:
        return []  # drop
    if r < 0.35:
        k = int(rng.integers(1, 3))
        m2 = cv2.erode(m, np.ones((2 * k + 1,) * 2, np.uint8))
        return [m2] if m2.any() else [m]
    if r < 0.55:
        k = int(rng.integers(1, 4))
        return [cv2.dilate(m, np.ones((2 * k + 1,) * 2, np.uint8))]
    if r < 0.65:  # split by a line through the centroid into two pieces
        ys, xs = np.nonzero(m)
        a = rng.uniform(0, np.pi)
        proj = (xs - xs.mean()) * np.cos(a) + (ys - ys.mean()) * np.sin(a)
        t = np.quantile(proj, rng.uniform(0.2, 0.8))
        p1, p2 = np.zeros_like(m), np.zeros_like(m)
        p1[ys[proj <= t], xs[proj <= t]] = 1
        p2[ys[proj > t], xs[proj > t]] = 1
        return [p for p in (p1, p2) if p.any()]
    if r < 0.75:  # shift
        dx, dy = rng.integers(-4, 5, 2)
        return [np.roll(np.roll(m, int(dy), 0), int(dx), 1)]
    return [m]


def sc_perturb(by, rng, disjoint=True, dup=False):
    out = {}
    for s, v in by.items():
        rid, rles = v[int(rng.integers(len(v)))]
        masks = []
        for r in rles:
            masks += _perturb_one(mu.decode(r), rng)
        # merge some pairs (union of two instances, = merged prediction)
        if len(masks) >= 2 and rng.random() < 0.5:
            i, j = rng.choice(len(masks), 2, replace=False)
            merged = masks[i] | masks[j]
            masks = [m for t, m in enumerate(masks) if t not in (i, j)] + [merged]
        for _ in range(int(rng.integers(0, 4))):  # false blobs inside the frame
            fb = np.zeros((2048, 2048), np.uint8)
            cx, cy = rng.integers(300, 1750, 2)
            cv2.ellipse(fb, (int(cx), int(cy)), (int(rng.integers(5, 60)), int(rng.integers(2, 6))),
                        float(rng.uniform(0, 180)), 0, 360, 1, -1)
            masks.append(fb)
        rng.shuffle(masks)
        if dup and masks:  # deliberately overlapping duplicates (Kaggle would reject; tests counting rules)
            for m in list(masks)[: max(1, len(masks) // 3)]:
                masks.append(cv2.dilate(m, np.ones((3, 3), np.uint8)))
        if disjoint:
            owner = np.zeros((2048, 2048), bool)
            fin = []
            for m in masks:
                f = m.astype(bool) & ~owner
                if f.sum() >= 1:
                    owner |= f
                    fin.append(f)
            masks = fin
        out[s] = [enc(m) for m in masks if m.any()]
    return out


def sc_random_blobs(stems, rng):
    out = {}
    for s in stems:
        ms = []
        for _ in range(5):
            fb = np.zeros((2048, 2048), np.uint8)
            cx, cy = rng.integers(300, 1750, 2)
            cv2.ellipse(fb, (int(cx), int(cy)), (40, 4), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            ms.append(fb)
        owner = np.zeros((2048, 2048), bool)
        fin = []
        for m in ms:
            f = m.astype(bool) & ~owner
            owner |= f
            if f.any():
                fin.append(enc(f))
        out[s] = fin
    return out


# ----------------------------------------------------------------------------------- scoring
def pred_dataframe(preds):
    rows = [(f"{s}_{j}", r["counts"].decode("ascii")) for s in sorted(preds) for j, r in enumerate(preds[s], 1)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def score_official(gt_df, pred_df):
    ov = O.get_overlap_df(gt_df, pred_df)
    pq, c = O.get_pq_score_counts(ov)
    return pq, c


def score_metric(preds, gt):
    pq, info = evaluate(preds, gt)
    return pq, dict(S=info["sq"] * info["tp"], tp=info["tp"], fp=info["fp"], fn=info["fn"])


def score_fast(preds, stems, include_missing=True, chunk=25):
    t = np.zeros(4)
    stems = sorted(stems)
    for i in range(0, len(stems), chunk):
        ch = stems[i:i + chunk]
        g = FastGT(ch)
        finals = {}
        for s in ch:
            if s in preds:
                finals[s] = [to_local(r) for r in preds[s]]
            elif include_missing:
                finals[s] = []
        for s, f in finals.items():
            t += g.stats(s, f)
        del g
        gc.collect()
    S, TP, FP, FN = t
    d = TP + 0.5 * FP + 0.5 * FN
    return (S / d if d else 1.0), dict(S=S, tp=int(TP), fp=int(FP), fn=int(FN))


def fmt(name, pq, c):
    return f"{name:9s} PQ={pq:.6f} S={c['S']:.4f} TP={c['tp']} FP={c['fp']} FN={c['fn']}"


def main():
    sc = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    meta = load_meta()
    stems = sorted(meta[meta.fold == 0].file_name.str[:-5].unique())
    if n:
        stems = stems[:n]
    stems = set(stems)
    t0 = time.time()
    gt = load_gt_rles(ANN, stems)
    gt_df = gt_dataframe(stems)
    by = readings_of(gt)
    rng = np.random.default_rng(123)
    test_stems = sorted(f.rsplit(".", 1)[0] for f in os.listdir(TEST_IMG))
    extra_note = ""
    if sc == "first":
        preds = sc_first(by, rng)
    elif sc == "union":
        preds = sc_union(by, rng)
    elif sc == "perturb":
        preds = sc_perturb(by, rng)
    elif sc == "overlap":
        preds = sc_perturb(by, rng, disjoint=False, dup=True)
    elif sc == "empty":
        preds = {}
    elif sc == "nogt":
        half = sorted(stems)[::2]
        preds = sc_perturb({s: by[s] for s in half}, rng)
        preds.update(sc_random_blobs(test_stems[:20], rng))
        extra_note = f"(preds for {len(half)}/{len(stems)} val stems + 20 test stems w/o GT)"
    elif sc == "csv":
        preds = sc_perturb(by, rng)
    else:
        raise SystemExit(sc)
    print(f"[{sc}] stems={len(stems)} readings={len(gt)} gt_rows={len(gt_df)} "
          f"pred_rows={sum(len(v) for v in preds.values())} {extra_note} build {time.time() - t0:.0f}s", flush=True)

    if sc == "csv":
        path = os.path.join(OUT, "parity_sub.csv")
        class Lazy(dict):  # decode one mask at a time (bounded memory)
            def __getitem__(self, k):
                return (mu.decode(r).astype(bool) for r in dict.__getitem__(self, k))
        write_submission(path, Lazy({s: v for s, v in preds.items() if v}))
        print("validator:", validate_submission(path, stems))
        pred_df = pd.read_csv(path)
        mem_df = pred_dataframe(preds)
        print("csv rows == in-memory rows:", pred_df.equals(mem_df))
    else:
        pred_df = pred_dataframe(preds) if preds else pd.DataFrame(columns=["filament_id", "segmentation_rle"])

    t1 = time.time()
    r_off = score_official(gt_df, pred_df)
    print(fmt("official", *r_off), f"{time.time() - t1:.0f}s", flush=True)
    r_met = score_metric(preds, gt)
    print(fmt("metric", *r_met), flush=True)
    r_fast = score_fast(preds, stems)
    print(fmt("FastGT", *r_fast), flush=True)
    if sc == "nogt":
        r_fast2 = score_fast(preds, stems, include_missing=False)
        print(fmt("FastGT*", *r_fast2), " <- val stems absent from the finals dict", flush=True)
    for name, r in (("metric", r_met), ("FastGT", r_fast)):
        same = (r[1]["tp"], r[1]["fp"], r[1]["fn"]) == (r_off[1]["tp"], r_off[1]["fp"], r_off[1]["fn"])
        print(f"  {name} vs official: counts {'IDENTICAL' if same else 'DIFFER'}  |dPQ|={abs(r[0] - r_off[0]):.2e} "
              f"|dS|={abs(r[1]['S'] - r_off[1]['S']):.2e}")


if __name__ == "__main__":
    main()
