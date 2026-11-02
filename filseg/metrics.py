"""Local re-implementation of the competition metrics.

* Panoptic Quality (Kirillov et al., CVPR 2019): a prediction and a ground-truth
  filament are a true positive when IoU > 0.5 (such matches are unique), and
      PQ = sum(IoU of TPs) / (|TP| + 0.5 |FP| + 0.5 |FN|)
* Dice / IoU of the union masks per image.
* Fragmentation (one GT covered by several predictions) and over-merging
  (one prediction covering several GTs).

The organizers' self-evaluation notebook is the reference; use this to iterate
quickly and cross-check it against their numbers once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils

MATCH_IOU = 0.5
# A prediction "belongs" to a GT filament when this fraction of it lies inside that
# GT (and vice versa for merges); used only for the fragmentation statistics.
COVER_FRAC = 0.5


@dataclass
class ImageScore:
    image_id: str
    tp: int
    fp: int
    fn: int
    iou_sum: float
    dice: float
    iou: float
    fragmented_gt: int  # GT filaments split over >= 2 predictions
    merged_pred: int  # predictions spanning >= 2 GT filaments

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denom if denom else 1.0


def _iou_matrix(pred: list[dict], gt: list[dict]) -> np.ndarray:
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)))
    return np.asarray(mask_utils.iou(pred, gt, [0] * len(gt)))


def _union_iou(pred: list[dict], gt: list[dict]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    p = mask_utils.merge(pred, intersect=False)
    g = mask_utils.merge(gt, intersect=False)
    return float(mask_utils.iou([p], [g], [0])[0, 0])


def score_image(image_id: str, gt: list[dict], pred: list[dict]) -> ImageScore:
    ious = _iou_matrix(pred, gt)
    matched = ious > MATCH_IOU
    tp = int(matched.sum())

    # Intersection from IoU and areas: |A n B| = IoU * (|A| + |B|) / (1 + IoU)
    a_pred = mask_utils.area(pred).astype(float) if pred else np.zeros(0)
    a_gt = mask_utils.area(gt).astype(float) if gt else np.zeros(0)
    inter = ious * (a_pred[:, None] + a_gt[None, :]) / (1.0 + ious)
    pred_in_gt = inter / np.maximum(a_pred[:, None], 1)  # fraction of pred inside gt
    gt_in_pred = inter / np.maximum(a_gt[None, :], 1)  # fraction of gt inside pred

    union_iou = _union_iou(pred, gt)
    return ImageScore(
        image_id=image_id,
        tp=tp,
        fp=len(pred) - tp,
        fn=len(gt) - tp,
        iou_sum=float(ious[matched].sum()),
        dice=2 * union_iou / (1 + union_iou),
        iou=union_iou,
        fragmented_gt=int(((pred_in_gt > COVER_FRAC).sum(axis=0) >= 2).sum()),
        merged_pred=int(((gt_in_pred > COVER_FRAC).sum(axis=1) >= 2).sum()),
    )


def summarize(scores: list[ImageScore]) -> dict[str, float]:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    iou_sum = sum(s.iou_sum for s in scores)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": iou_sum / denom if denom else 1.0,  # pooled over the dataset
        "pq_mean_per_image": float(np.mean([s.pq for s in scores])),
        "sq": iou_sum / tp if tp else 0.0,  # segmentation quality (mean IoU of matches)
        "rq": tp / denom if denom else 1.0,  # recognition quality (F1-like)
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(np.mean([s.dice for s in scores])),
        "iou": float(np.mean([s.iou for s in scores])),
        "fragmented_gt": sum(s.fragmented_gt for s in scores),
        "merged_pred": sum(s.merged_pred for s in scores),
        "n_images": len(scores),
    }


def format_summary(summary: dict[str, float]) -> str:
    return (
        f"PQ {summary['pq']:.4f} (per-image mean {summary['pq_mean_per_image']:.4f}) | "
        f"SQ {summary['sq']:.3f} RQ {summary['rq']:.3f} | "
        f"TP {summary['tp']} FP {summary['fp']} FN {summary['fn']} | "
        f"Dice {summary['dice']:.4f} IoU {summary['iou']:.4f} | "
        f"fragmented GT {summary['fragmented_gt']} merged preds {summary['merged_pred']} | "
        f"{summary['n_images']} images"
    )
