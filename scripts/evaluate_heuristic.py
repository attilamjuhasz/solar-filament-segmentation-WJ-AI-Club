"""
Gate C: Evaluate the deterministic heuristic baseline on the exact same
fixed tuning and confirmation partitions as neural models.
Computes strict competition PQ, SQ, RQ, TP, FP, FN, and instance error diagnostics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np

from evaluate import compute_instance_diagnostics
from generate_submission import extract_filaments_from_image
from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.annotations import load_coco_annotations
from src.data.folds import get_deterministic_fold_partitions, load_frozen_folds_manifest
from src.data.manifest import canonical_observation_id
from src.evaluation.competition_adapter import evaluate_entry_pq
from src.inference.rle import decode_instance


def evaluate_heuristic_on_partition(
    split_name: str,
    obs_ids: List[str],
    annotations_index: Any,
    images_dir: Path,
    bth_kernel_size: int = 41,
    disk_erode_size: int = 15,
    seed_thresh: int = 50,
    low_thresh: int = 24,
    min_area: int = 500,
    min_eccentricity: float = 0.90,
    min_major_axis: float = 90.0,
    max_candidates: int = 10,
) -> Dict[str, Any]:
    """Evaluate heuristic baseline on a specific partition of observations."""
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_erode_size, disk_erode_size))
    bth_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bth_kernel_size, bth_kernel_size))

    print(f"\n[Gate C] Evaluating heuristic baseline on '{split_name}' ({len(obs_ids)} observations)...", flush=True)

    entries_for_pq: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
    total_iou_sum = 0.0
    global_tp = 0
    global_fp = 0
    global_fn = 0
    total_one_to_many = 0
    total_many_to_one = 0

    diag_agg = {
        "n_gt": 0,
        "n_pred": 0,
        "fragmented_gt_count": 0,
        "over_merged_pred_count": 0,
        "missed_gt_count": 0,
        "spurious_pred_count": 0,
        "strict_matched_count": 0,
        "strict_unmatched_fn": 0,
        "strict_unmatched_fp": 0,
    }

    dice_scores: List[float] = []
    obs_reports: List[Dict[str, Any]] = []

    for obs_id in obs_ids:
        canon_id = canonical_observation_id(obs_id)
        variants = annotations_index.get_observation_variants(canon_id)
        if not variants:
            variants = annotations_index.get_observation_variants(obs_id)
        if not variants:
            print(f"  [Warning] No annotations found for {obs_id} (canonical {canon_id}), skipping.", flush=True)
            continue

        # Locate image file
        img_name = variants[0].file_name
        img_path = images_dir / img_name
        if not img_path.is_file():
            # Search by stem
            candidates = list(images_dir.glob(f"*{canon_id}*"))
            if not candidates:
                raise FileNotFoundError(f"Could not find image for {obs_id} in {images_dir}")
            img_path = candidates[0]

        # Run heuristic extraction
        rle_rows = extract_filaments_from_image(
            image_path=str(img_path),
            obs_id=canon_id,
            disk_kernel=disk_kernel,
            bth_kernel=bth_kernel,
            seed_thresh=seed_thresh,
            low_thresh=low_thresh,
            min_area=min_area,
            min_eccentricity=min_eccentricity,
            min_major_axis=min_major_axis,
            max_candidates_per_image=max_candidates,
        )

        pred_masks = [decode_instance(rle, shape=NATIVE_IMAGE_SHAPE) for _, rle in rle_rows]
        n_preds = len(pred_masks)

        # Union of predictions for semantic Dice
        pred_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
        for pm in pred_masks:
            pred_union |= (pm > 0)

        obs_entry_reports = []
        for v in variants:
            gt_masks = [inst.get_mask(NATIVE_IMAGE_SHAPE) for inst in v.instances if inst.area > 0]
            # Filter empty GT masks
            gt_masks = [m for m in gt_masks if (m > 0).any()]
            n_gts = len(gt_masks)

            # Evaluate entry PQ
            iou_sum, tp, fp, fn, o2m, m2o = evaluate_entry_pq(gt_masks, pred_masks)
            total_iou_sum += iou_sum
            global_tp += tp
            global_fp += fp
            global_fn += fn
            total_one_to_many += o2m
            total_many_to_one += m2o

            # Compute instance diagnostics
            inst_diag = compute_instance_diagnostics(gt_masks, pred_masks)
            for k in diag_agg:
                diag_agg[k] += inst_diag.get(k, 0)

            # Compute semantic Dice
            gt_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
            for gm in gt_masks:
                gt_union |= (gm > 0)
            inter = int(np.logical_and(gt_union, pred_union).sum())
            union_sum = int(gt_union.sum()) + int(pred_union.sum())
            entry_dice = (2.0 * inter / union_sum) if union_sum > 0 else (1.0 if int(gt_union.sum()) == 0 else 0.0)
            dice_scores.append(entry_dice)

            obs_entry_reports.append({
                "variant_id": v.annotator_image_id,
                "n_gt": n_gts,
                "n_pred": n_preds,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou_sum": round(iou_sum, 4),
                "dice": round(entry_dice, 4),
                "diagnostics": inst_diag,
            })

        obs_reports.append({
            "observation_id": canon_id,
            "raw_observation_id": obs_id,
            "image_file": img_path.name,
            "pred_instance_count": n_preds,
            "variants": obs_entry_reports,
        })
        print(f"  Obs {canon_id}: {n_preds} preds across {len(variants)} variants | Preds area: {int(pred_union.sum())}", flush=True)

    denom = global_tp + 0.5 * (global_fp + global_fn)
    overall_pq = (total_iou_sum / denom) if denom > 0 else 0.0
    overall_sq = (total_iou_sum / global_tp) if global_tp > 0 else 0.0
    overall_rq = (global_tp / denom) if denom > 0 else 0.0
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

    print(f"\n[Gate C] '{split_name}' Summary:")
    print(f"  PQ: {overall_pq:.4f} | SQ: {overall_sq:.4f} | RQ: {overall_rq:.4f} | Mean Dice: {mean_dice:.4f}")
    print(f"  TP: {global_tp} | FP: {global_fp} | FN: {global_fn} (Total entries: {len(dice_scores)})")
    print(f"  Diagnostics: Missed GT: {diag_agg['missed_gt_count']}, Spurious Pred: {diag_agg['spurious_pred_count']}, Frag: {diag_agg['fragmented_gt_count']}, OverMerge: {diag_agg['over_merged_pred_count']}")

    return {
        "split": split_name,
        "observations_count": len(obs_ids),
        "annotator_entries_count": len(dice_scores),
        "overall": {
            "pq": round(overall_pq, 5),
            "sq": round(overall_sq, 5),
            "rq": round(overall_rq, 5),
            "mean_dice": round(mean_dice, 5),
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn,
            "total_iou": round(total_iou_sum, 5),
            "one_to_many": total_one_to_many,
            "many_to_one": total_many_to_one,
        },
        "diagnostics": diag_agg,
        "parameters": {
            "bth_kernel_size": bth_kernel_size,
            "disk_erode_size": disk_erode_size,
            "seed_thresh": seed_thresh,
            "low_thresh": low_thresh,
            "min_area": min_area,
            "min_eccentricity": min_eccentricity,
            "min_major_axis": min_major_axis,
            "max_candidates": max_candidates,
        },
        "observations": obs_reports,
    }


def main():
    manifest_path = root_dir / "artifacts" / "folds_manifest.json"
    assignments, manifest_sha256 = load_frozen_folds_manifest(manifest_path)
    data_dir = root_dir / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_images = data_dir / "train" / "train_images"

    ann_index = load_coco_annotations(str(train_json))
    partitions = get_deterministic_fold_partitions(assignments, target_fold=0)

    tuning_obs = partitions["tuning"]
    confirmation_obs = partitions["confirmation"]

    # 1. Evaluate on Tuning partition
    tuning_report = evaluate_heuristic_on_partition(
        split_name="tuning",
        obs_ids=tuning_obs,
        annotations_index=ann_index,
        images_dir=train_images,
    )
    tuning_report["folds_manifest_sha256"] = manifest_sha256

    report_dir = root_dir / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    tuning_report_path = report_dir / "heuristic_baseline_tuning.json"
    with open(tuning_report_path, "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2)
    print(f"[Gate C] Tuning report saved to: {tuning_report_path}")

    # 2. Evaluate on Confirmation partition
    confirmation_report = evaluate_heuristic_on_partition(
        split_name="confirmation",
        obs_ids=confirmation_obs,
        annotations_index=ann_index,
        images_dir=train_images,
    )
    confirmation_report["folds_manifest_sha256"] = manifest_sha256

    confirm_report_path = report_dir / "heuristic_baseline_confirmation.json"
    with open(confirm_report_path, "w", encoding="utf-8") as f:
        json.dump(confirmation_report, f, indent=2)
    print(f"[Gate C] Confirmation report saved to: {confirm_report_path}")


if __name__ == "__main__":
    main()
