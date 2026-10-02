"""Replica of the competition PQ.

For every GT reading ("<annotator>-<stem>") the predictions of that stem are scored independently;
counts are pooled (micro) over all readings. A pair matches when IoU > 0.5 (no assignment step --
non-overlapping masks make matches unique anyway).
"""
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mu

from rle import H, W


def _rles(masks_or_rles):
    out = []
    for m in masks_or_rles:
        if isinstance(m, dict):
            out.append(m)
        else:
            out.append(mu.encode(np.asfortranarray((np.asarray(m) > 0).astype(np.uint8))))
    return out


def load_gt_rles(ann_path, stems=None):
    """{reading_id: (stem, [rle])} rasterized with pycocotools (polygon -> RLE)."""
    d = json.load(open(ann_path))
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    out = {}
    for im in d["images"]:
        stem = im["file_name"].rsplit(".", 1)[0]
        if stems is not None and stem not in stems:
            continue
        rles = [mu.merge(mu.frPyObjects(a["segmentation"], H, W)) for a in by_img[im["id"]]]
        out[im["id"]] = (stem, rles)
    return out


def reading_stats(pred_rles, gt_rles, thr=0.5):
    """(sum_iou, tp, fp, fn) for one reading."""
    n_p, n_g = len(pred_rles), len(gt_rles)
    if n_p == 0 or n_g == 0:
        return 0.0, 0, n_p, n_g
    iou = np.asarray(mu.iou(pred_rles, gt_rles, [0] * n_g))  # (n_pred, n_gt)
    hit = iou > thr
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(1) == 0).sum()), int((hit.sum(0) == 0).sum())


def evaluate(preds, gt, thr=0.5, per_size=False):
    """preds: {stem: [mask or rle]}; gt: output of load_gt_rles (restricted to the eval stems).

    Returns (pq, info) with SQ, RQ, tp/fp/fn and optional recall per GT size bin.
    """
    pred_rles = {s: _rles(v) for s, v in preds.items()}
    S = TP = FP = FN = 0
    bins = [0, 400, 1000, 3000, 8000, 1e9]
    size_hits = np.zeros(len(bins) - 1)
    size_tot = np.zeros(len(bins) - 1)
    for rid, (stem, g) in gt.items():
        p = pred_rles.get(stem, [])
        s, tp, fp, fn = reading_stats(p, g, thr)
        S, TP, FP, FN = S + s, TP + tp, FP + fp, FN + fn
        if per_size and g:
            area = np.array([mu.area(r) for r in g])
            matched = (np.asarray(mu.iou(p, g, [0] * len(g))) > thr).any(0) if p else np.zeros(len(g), bool)
            b = np.digitize(area, bins) - 1
            np.add.at(size_tot, b, 1)
            np.add.at(size_hits, b, matched)
    denom = TP + 0.5 * FP + 0.5 * FN
    pq = S / denom if denom else 1.0
    info = dict(pq=pq, sq=S / max(TP, 1), rq=TP / denom if denom else 1.0, tp=TP, fp=FP, fn=FN,
                prec=TP / max(TP + FP, 1), rec=TP / max(TP + FN, 1))
    if per_size:
        info["recall_by_size"] = {f"<{int(bins[i + 1])}" if i < len(bins) - 2 else f">={int(bins[i])}":
                                  round(size_hits[i] / max(size_tot[i], 1), 3) for i in range(len(bins) - 1)}
    return pq, info
