from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
from pycocotools import mask as coco_mask

from src.contracts import IOU_HIT_THRESHOLD, NATIVE_IMAGE_SHAPE, PQStats


def compute_iou_matrix(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> np.ndarray:
    """Compute pairwise IoU matrix of shape [len(gt_masks), len(pred_masks)].
    
    Uses pycocotools fast C-level RLE IoU calculation.
    """
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)

    # Encode all GT masks to Fortran uint8 RLEs
    gt_rles = [
        coco_mask.encode(np.asfortranarray((m > 0).astype(np.uint8)))
        for m in gt_masks
    ]
    pred_rles = [
        coco_mask.encode(np.asfortranarray((m > 0).astype(np.uint8)))
        for m in pred_masks
    ]

    # pycocotools.mask.iou returns [len(pred_rles), len(gt_rles)] when iscrowd is 0/list
    # Note: coco_mask.iou(dt, gt, iscrowd)
    iou_matrix = coco_mask.iou(gt_rles, pred_rles, [0] * len(pred_rles))
    return np.asarray(iou_matrix, dtype=np.float64)


def evaluate_entry_pq(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    iou_threshold: float = IOU_HIT_THRESHOLD
) -> Tuple[float, int, int, int, int, int]:
    """Evaluate a single annotator-image entry against predicted instance masks.
    
    Returns:
        (total_iou, tp, fp, fn, one_to_many, many_to_one)
    """
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)

    if num_gt == 0 and num_pred == 0:
        return 0.0, 0, 0, 0, 0, 0
    if num_gt == 0:
        return 0.0, 0, num_pred, 0, 0, 0
    if num_pred == 0:
        return 0.0, 0, 0, num_gt, 0, 0

    iou_mat = compute_iou_matrix(gt_masks, pred_masks)  # [num_gt, num_pred]

    # Under strict > 0.5 threshold, at most one element in any row or column can be > 0.5
    hits = iou_mat > iou_threshold

    tp = int(hits.sum())
    total_iou = float(iou_mat[hits].sum())

    # An instance without any qualifying hit (> 0.5) is unmatched
    gt_matched = hits.any(axis=1)    # [num_gt]
    pred_matched = hits.any(axis=0)  # [num_pred]

    fn = int((~gt_matched).sum())
    fp = int((~pred_matched).sum())

    # Split/Merge diagnostics (partial overlap IoU > 0.05)
    partial_overlaps = iou_mat > 0.05
    one_to_many = int((partial_overlaps.sum(axis=1) > 1).sum())  # 1 GT overlapping multiple predictions
    many_to_one = int((partial_overlaps.sum(axis=0) > 1).sum())  # 1 pred overlapping multiple GTs

    return total_iou, tp, fp, fn, one_to_many, many_to_one


def evaluate_dataset_pq(
    entries: Sequence[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    iou_threshold: float = IOU_HIT_THRESHOLD
) -> PQStats:
    """Compute dataset-wide Panoptic Quality matching the Kaggle 2026 self-evaluation protocol.
    
    Each entry is a tuple of (gt_masks_for_annotator, pred_masks_for_observation).
    """
    total_iou = 0.0
    global_tp = 0
    global_fp = 0
    global_fn = 0
    total_one_to_many = 0
    total_many_to_one = 0

    for gt_masks, pred_masks in entries:
        iou_sum, tp, fp, fn, o2m, m2o = evaluate_entry_pq(gt_masks, pred_masks, iou_threshold)
        total_iou += iou_sum
        global_tp += tp
        global_fp += fp
        global_fn += fn
        total_one_to_many += o2m
        total_many_to_one += m2o

    denominator = global_tp + 0.5 * (global_fp + global_fn)
    pq = (total_iou / denominator) if denominator > 0 else 0.0
    sq = (total_iou / global_tp) if global_tp > 0 else 0.0
    rq = (global_tp / denominator) if denominator > 0 else 0.0

    return PQStats(
        pq=pq,
        sq=sq,
        rq=rq,
        tp=global_tp,
        fp=global_fp,
        fn=global_fn,
        total_iou=total_iou,
        entry_count=len(entries),
        one_to_many_count=total_one_to_many,
        many_to_one_count=total_many_to_one
    )
