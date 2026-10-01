from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import yaml

from src.contracts import NATIVE_IMAGE_SHAPE, PQStats
from src.data.annotations import load_coco_annotations
from src.data.dataset import SolarFilamentDataset, load_solar_image
from src.data.folds import (
    ObservationRecord,
    assign_stratified_group_folds,
    consolidate_canonical_records,
    freeze_folds_manifest,
    get_deterministic_fold_partitions,
    load_frozen_folds_manifest,
    verify_fold_isolation,
)
from src.data.manifest import canonical_observation_id
from src.evaluation.competition_adapter import evaluate_entry_pq
from src.inference.config import UNSET, resolve_inference_config
from src.inference.engine import (
    compute_bytes_sha256,
    compute_file_sha256,
    compute_state_dict_sha256,
    load_cached_prediction,
    predict_full_observation,
    save_cached_prediction,
)
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import decode_instance
from src.models import build_model
from train import build_fold_records


def compute_instance_diagnostics(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Analyze instance error modes: fragmentation, over-merging, and complete misses using true bipartite matrices."""
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    if n_gt == 0 and n_pred == 0:
        return {
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

    gt_areas = [int((gm > 0).sum()) for gm in gt_masks]
    pred_areas = [int((pm > 0).sum()) for pm in pred_masks]

    # Bipartite overlap matrices
    intersection = np.zeros((n_gt, n_pred), dtype=np.int64)
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    gt_coverage = np.zeros((n_gt, n_pred), dtype=np.float64)
    pred_coverage = np.zeros((n_gt, n_pred), dtype=np.float64)

    for g_idx in range(n_gt):
        g_bool = gt_masks[g_idx] > 0
        g_area = gt_areas[g_idx]
        if g_area == 0:
            continue
        for p_idx in range(n_pred):
            p_bool = pred_masks[p_idx] > 0
            p_area = pred_areas[p_idx]
            if p_area == 0:
                continue
            inter = int(np.logical_and(g_bool, p_bool).sum())
            if inter > 0:
                intersection[g_idx, p_idx] = inter
                union = g_area + p_area - inter
                iou[g_idx, p_idx] = inter / union
                gt_coverage[g_idx, p_idx] = inter / g_area
                pred_coverage[g_idx, p_idx] = inter / p_area

    # Strict PQ matching (greedy bipartite by IoU >= iou_thresh)
    strict_matched = 0
    matched_gt = set()
    matched_pred = set()
    if n_gt > 0 and n_pred > 0:
        pairs = []
        for g in range(n_gt):
            for p in range(n_pred):
                if iou[g, p] > iou_thresh:
                    pairs.append((iou[g, p], g, p))
        pairs.sort(key=lambda x: x[0], reverse=True)
        for _, g, p in pairs:
            if g not in matched_gt and p not in matched_pred:
                matched_gt.add(g)
                matched_pred.add(p)
                strict_matched += 1

    strict_fn = n_gt - strict_matched
    strict_fp = n_pred - strict_matched

    # Physical error diagnostics:
    # 1. Missed GT: ground truth with zero predictions having meaningful coverage (IoU >= 0.05 or (gt_coverage >= 0.10 and pred_coverage >= 0.05))
    missed_gt = 0
    fragmented_gt = 0
    for g_idx in range(n_gt):
        if gt_areas[g_idx] == 0:
            missed_gt += 1
            continue
        inter_min = min(16, int(gt_areas[g_idx]))
        meaningful_preds = [
            p for p in range(n_pred)
            if (iou[g_idx, p] >= 0.05 or (gt_coverage[g_idx, p] >= 0.10 and pred_coverage[g_idx, p] >= 0.05))
            and intersection[g_idx, p] >= inter_min
        ]
        if len(meaningful_preds) == 0:
            missed_gt += 1
        elif len(meaningful_preds) >= 2:
            fragmented_gt += 1

    # 2. Spurious and Over-merged predictions:
    spurious_preds = 0
    over_merged_preds = 0
    for p_idx in range(n_pred):
        if pred_areas[p_idx] == 0:
            spurious_preds += 1
            continue
        inter_min = min(16, int(pred_areas[p_idx]))
        meaningful_gts = [
            g for g in range(n_gt)
            if (iou[g, p_idx] >= 0.05 or (pred_coverage[g, p_idx] >= 0.10 and gt_coverage[g, p_idx] >= 0.05))
            and intersection[g, p_idx] >= inter_min
        ]
        if len(meaningful_gts) == 0:
            spurious_preds += 1
        elif len(meaningful_gts) >= 2:
            over_merged_preds += 1

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "fragmented_gt_count": fragmented_gt,
        "over_merged_pred_count": over_merged_preds,
        "missed_gt_count": missed_gt,
        "spurious_pred_count": spurious_preds,
        "strict_matched_count": strict_matched,
        "strict_unmatched_fn": strict_fn,
        "strict_unmatched_fp": strict_fp,
    }


def evaluate_oof(
    checkpoint_path: str,
    fold: Optional[int] = None,
    split: str = "tuning",
    config_path: str = "configs/b0_resnet34.yaml",
    tile_size: Optional[int] = None,
    stride: Optional[int] = None,
    limit: Optional[int] = None,
    allow_unverified_provenance: bool = False,
    cache_dir: str = "artifacts/cache",
    method: Optional[str] = None,
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
    center_threshold: Optional[float] = None,
    boundary_weight: Optional[float] = None,
    marker_min_distance: Optional[int] = None,
    marker_cap_per_component: Optional[int] = UNSET,
    max_peaks: Optional[int] = None,
    max_instances: Optional[int] = UNSET,
    min_area: Optional[int] = None,
    tile_batch_size: Optional[int] = None,
    calibration_ratio: float = 0.50,
    device_str: Optional[str] = None,
    inference_config_overrides: Optional[Dict[str, Any]] = None,
    strict_cache: Optional[bool] = None,
) -> Dict[str, Any]:
    """Evaluate Out-Of-Fold (OOF) model performance on full 2048x2048 validation observations.
    
    Guarantees:
    - Rejects mismatched checkpoint/requested folds and train/validation observation overlaps.
    - Uses unified prediction engine with cached probability/offset maps.
    - Evaluates against individual annotator variants (without unioning or multi-annotator mixing).
    - Computes competition-aligned non-Hungarian PQ, SQ, RQ, Dice, and error breakdown.
    - Partitions validation set into calibration vs confirmation sets to guard against overfitting.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_sha256 = compute_file_sha256(ckpt_path)
    is_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_name = device_str if device_str else ("cuda" if is_cuda else "cpu")
    device = torch.device(device_name)

    # 1. Load and verify checkpoint provenance
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ckpt_fold = ckpt.get("fold")
    ckpt_manifest_sha = ckpt.get("folds_manifest_sha256")
    train_obs_ckpt = ckpt.get("train_observations")
    val_obs_ckpt = ckpt.get("val_observations")
    metric_name = ckpt.get("metric_name")

    has_full_provenance = (
        ckpt_fold is not None
        and ckpt_manifest_sha is not None
        and train_obs_ckpt is not None
        and val_obs_ckpt is not None
        and metric_name is not None
    )

    if not has_full_provenance:
        if not allow_unverified_provenance:
            missing_fields = []
            if ckpt_fold is None: missing_fields.append("fold")
            if ckpt_manifest_sha is None: missing_fields.append("folds_manifest_sha256")
            if train_obs_ckpt is None: missing_fields.append("train_observations")
            if val_obs_ckpt is None: missing_fields.append("val_observations")
            if metric_name is None: missing_fields.append("metric_name")
            raise ValueError(
                f"Checkpoint {ckpt_path.name} lacks complete authenticated provenance metadata: "
                f"missing {missing_fields}. Specify --allow-unverified-provenance to inspect legacy/unverified checkpoints."
            )
        provenance = "unverified_legacy"
        target_fold = 0 if fold is None else fold
    else:
        if fold is not None and fold != ckpt_fold:
            raise ValueError(
                f"Fold mismatch! Checkpoint was trained on fold {ckpt_fold}, but evaluation requested fold {fold}. "
                "Refusing evaluation to prevent training observation leakage."
            )
        target_fold = ckpt_fold
        provenance = "verified"

    config = ckpt.get("config")
    if config is None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Resolve unified immutable inference configuration
    overrides: Dict[str, Any] = {}
    if method is not None: overrides["method"] = method
    if high_threshold is not None: overrides["high_threshold"] = high_threshold
    if low_threshold is not None: overrides["low_threshold"] = low_threshold
    if center_threshold is not None: overrides["center_threshold"] = center_threshold
    if boundary_weight is not None: overrides["boundary_weight"] = boundary_weight
    if marker_min_distance is not None: overrides["marker_min_distance"] = marker_min_distance
    if marker_cap_per_component is not UNSET: overrides["marker_cap_per_component"] = marker_cap_per_component
    if max_peaks is not None: overrides["max_peaks"] = max_peaks
    if max_instances is not UNSET: overrides["max_instances"] = max_instances
    if min_area is not None: overrides["min_area"] = min_area
    if tile_size is not None: overrides["tile_size"] = tile_size
    if stride is not None: overrides["stride"] = stride
    if tile_batch_size is not None:
        if tile_batch_size <= 0:
            raise ValueError(f"Require positive tile_batch_size (> 0), got {tile_batch_size}")
        overrides["tile_batch_size"] = tile_batch_size
    if inference_config_overrides is not None:
        overrides.update(inference_config_overrides)

    resolved_inf_cfg = resolve_inference_config(config, overrides=overrides)
    act_tile_size = resolved_inf_cfg["tile_size"]
    act_stride = resolved_inf_cfg["stride"]
    act_norm_mode = resolved_inf_cfg["norm_mode"]
    act_tile_batch_size = resolved_inf_cfg.get("tile_batch_size", 16)
    include_aux = (resolved_inf_cfg["method"] == "watershed")

    # 2. Build model and load weights
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load dataset & validated fold assignments
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    train_images = data_dir / "train" / "train_images"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    annotation_index = load_coco_annotations(str(train_json))
    records = build_fold_records(annotation_index)

    # Strictly require frozen folds manifest for verified evaluation
    manifest_file = root_dir / "artifacts" / "folds_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Folds manifest not found at {manifest_file}. Refusing to generate arbitrary split during evaluation."
        )

    fold_assignments, actual_manifest_sha = load_frozen_folds_manifest(manifest_file, verify_annotations_path=train_json)

    if provenance == "verified":
        if ckpt_manifest_sha != actual_manifest_sha:
            raise ValueError(
                f"Checkpoint folds_manifest_sha256 ({ckpt_manifest_sha}) does not match "
                f"active frozen folds manifest SHA256 ({actual_manifest_sha})!"
            )
        # Verify train_observations membership
        for obs in train_obs_ckpt:
            c_obs = canonical_observation_id(obs)
            f = fold_assignments.get(c_obs)
            if f is None or f == target_fold:
                raise ValueError(
                    f"Provenance audit violation: recorded train observation '{obs}' (fold {f}) "
                    f"belongs to target evaluation fold {target_fold} or is missing from assignments!"
                )

    val_ds = SolarFilamentDataset(
        images_dir=train_images,
        annotation_index=annotation_index,
        fold_assignments=fold_assignments,
        target_fold=target_fold,
        is_train=False,
        patch_size=(2048, 2048),
    )

    # 4. Strict leakage audit: verify zero overlap with training observations
    if train_obs_ckpt is not None:
        train_set = {canonical_observation_id(o) for o in train_obs_ckpt}
        val_set = {canonical_observation_id(o) for o in val_ds.observations}
        overlap = train_set.intersection(val_set)
        if overlap:
            raise ValueError(f"Strict OOF violation! Train and validation observations overlap: {sorted(list(overlap))}")

    # Deterministic validation partitions (tuning vs confirmation vs explored)
    partitions = get_deterministic_fold_partitions(fold_assignments, target_fold=target_fold)
    if split == "tuning":
        target_pool = partitions["tuning"]
    elif split == "confirmation":
        target_pool = partitions["confirmation"]
    elif split == "explored":
        target_pool = partitions["explored"]
    elif split == "all":
        target_pool = partitions["all_validation"]
    else:
        raise ValueError(f"Unknown split '{split}'; must be 'tuning', 'confirmation', 'explored', or 'all'")

    num_eval = len(target_pool) if limit is None else min(limit, len(target_pool))
    eval_obs_list = target_pool[:num_eval]
    is_limited = (limit is not None and limit < len(target_pool))

    print(f"[Eval] Checkpoint: {ckpt_path.name} (SHA256: {ckpt_sha256[:12]}..., Provenance: {provenance})")
    print(f"[Eval] Evaluating Fold {target_fold} [{split.upper()} partition]: {len(eval_obs_list)} observations")

    # Metrics accumulators
    results_by_split = {
        "overall": {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "dice_scores": [], "frag": 0, "merge": 0, "miss": 0, "spurious": 0, "n_gt": 0, "n_pred": 0},
        "tuning": {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "dice_scores": [], "frag": 0, "merge": 0, "miss": 0, "spurious": 0, "n_gt": 0, "n_pred": 0},
        "confirmation": {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "dice_scores": [], "frag": 0, "merge": 0, "miss": 0, "spurious": 0, "n_gt": 0, "n_pred": 0},
        "explored": {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "dice_scores": [], "frag": 0, "merge": 0, "miss": 0, "spurious": 0, "n_gt": 0, "n_pred": 0},
    }

    per_obs_reports = []

    for idx, obs_id in enumerate(eval_obs_list):
        img_path = val_ds._resolve_image_path(obs_id)
        img_rgb = load_solar_image(img_path)
        variants = val_ds.get_observation_annotations(obs_id)

        # 5. Load cached prediction maps or compute and cache
        img_sha256 = compute_bytes_sha256(img_rgb.tobytes())
        model_state_sha = compute_state_dict_sha256(ckpt["model_state_dict"])
        act_precision = resolved_inf_cfg.get("precision", "float32")
        use_strict_cache = (provenance == "verified") if strict_cache is None else bool(strict_cache)
        cached = load_cached_prediction(
            cache_dir=cache_dir,
            ckpt_hash=ckpt_sha256,
            obs_id=obs_id,
            expected_image_sha256=img_sha256,
            expected_model_state_sha256=model_state_sha,
            expected_spatial_shape=NATIVE_IMAGE_SHAPE,
            expected_tile_size=act_tile_size,
            expected_stride=act_stride,
            expected_norm_mode=act_norm_mode,
            expected_precision=act_precision,
            expected_preprocessing_version="v3",
            expected_inference_policy="identity",
            requires_aux=include_aux,
            strict=use_strict_cache,
        )
        if cached is not None:
            full_fg, full_ctr, full_bnd, full_off = cached
        else:
            full_fg, full_ctr, full_bnd, full_off = predict_full_observation(
                model=model,
                image_rgb=img_rgb,
                device=device,
                tile_size=act_tile_size,
                stride=act_stride,
                tile_batch_size=act_tile_batch_size,
                norm_mode=act_norm_mode,
                include_aux=include_aux,
            )
            if cache_dir is not None:
                save_cached_prediction(
                    cache_dir=cache_dir,
                    ckpt_hash=ckpt_sha256,
                    obs_id=obs_id,
                    fg=full_fg,
                    ctr=full_ctr,
                    bnd=full_bnd,
                    off=full_off,
                    image_sha256=img_sha256,
                    model_state_sha256=model_state_sha,
                    tile_size=act_tile_size,
                    stride=act_stride,
                    norm_mode=act_norm_mode,
                    precision=act_precision,
                    preprocessing_version="v3",
                    inference_policy="identity",
                )

        # 6. Extract predicted instances using resolved configuration
        pred_instances = extract_instances_from_maps(
            foreground_prob=full_fg,
            center_prob=full_ctr if include_aux else None,
            boundary_prob=full_bnd if include_aux else None,
            offset_field=full_off if include_aux else None,
            obs_id=obs_id,
            high_threshold=resolved_inf_cfg["high_threshold"],
            low_threshold=resolved_inf_cfg["low_threshold"],
            min_area=resolved_inf_cfg["min_area"],
            method=resolved_inf_cfg["method"],
            center_threshold=resolved_inf_cfg["center_threshold"],
            boundary_weight=resolved_inf_cfg["boundary_weight"],
            marker_min_distance=resolved_inf_cfg["marker_min_distance"],
            marker_cap_per_component=resolved_inf_cfg["marker_cap_per_component"],
            max_peaks=resolved_inf_cfg["max_peaks"],
            max_instances=resolved_inf_cfg["max_instances"],
            shape=NATIVE_IMAGE_SHAPE,
        )
        pred_masks = [decode_instance(p.rle_counts, shape=NATIVE_IMAGE_SHAPE) for p in pred_instances]

        # Determine split tag
        tuning_set = set(partitions.get("tuning", []))
        confirmation_set = set(partitions.get("confirmation", []))
        if obs_id in tuning_set:
            split_tag = "tuning"
        elif obs_id in confirmation_set:
            split_tag = "confirmation"
        else:
            split_tag = "explored"

        # Evaluate against all annotator variants
        obs_tp, obs_fp, obs_fn, obs_iou = 0, 0, 0, 0.0
        obs_dice_list = []

        for var in variants:
            gt_masks = [inst.get_mask(NATIVE_IMAGE_SHAPE) for inst in var.instances if inst.area > 0]
            gt_masks = [m for m in gt_masks if (m > 0).any()]

            iou_sum, tp, fp, fn, _, _ = evaluate_entry_pq(gt_masks, pred_masks)
            obs_tp += tp
            obs_fp += fp
            obs_fn += fn
            obs_iou += iou_sum

            # Error mode diagnosis
            diag = compute_instance_diagnostics(gt_masks, pred_masks)

            # Dice score
            gt_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
            for gm in gt_masks:
                gt_union |= (gm > 0)
            pred_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
            for pm in pred_masks:
                pred_union |= (pm > 0)

            inter = 2.0 * np.logical_and(gt_union, pred_union).sum()
            union = gt_union.sum() + pred_union.sum()
            d_val = float((inter + 1e-6) / (union + 1e-6))
            obs_dice_list.append(d_val)

            # Accumulate into overall and split-specific accumulators
            for target_split in ("overall", split_tag):
                acc = results_by_split[target_split]
                acc["tp"] += tp
                acc["fp"] += fp
                acc["fn"] += fn
                acc["iou_sum"] += iou_sum
                acc["dice_scores"].append(d_val)
                acc["frag"] += diag["fragmented_gt_count"]
                acc["merge"] += diag["over_merged_pred_count"]
                acc["miss"] += diag["missed_gt_count"]
                acc["spurious"] += diag["spurious_pred_count"]
                acc["n_gt"] += diag["n_gt"]
                acc["n_pred"] += diag["n_pred"]

        per_obs_reports.append({
            "observation_id": obs_id,
            "split": split_tag,
            "annotator_variants_evaluated": len(variants),
            "predicted_instances": len(pred_instances),
            "tp": obs_tp,
            "fp": obs_fp,
            "fn": obs_fn,
            "mean_dice": float(np.mean(obs_dice_list)) if obs_dice_list else 0.0,
        })

    def format_pq_stats(acc: Dict[str, Any]) -> Dict[str, Any]:
        tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
        sq = float(acc["iou_sum"] / tp) if tp > 0 else 0.0
        denom = tp + 0.5 * fp + 0.5 * fn
        rq = float(tp / denom) if denom > 0 else 0.0
        pq = float(sq * rq)
        mean_dice = float(np.mean(acc["dice_scores"])) if acc["dice_scores"] else 0.0
        n_gt = max(1, acc["n_gt"])
        n_pred = max(1, acc["n_pred"])

        return {
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_dice": mean_dice,
            "total_gt_instances": acc["n_gt"],
            "total_pred_instances": acc["n_pred"],
            "fragmented_gt_count": acc["frag"],
            "over_merged_pred_count": acc["merge"],
            "missed_gt_count": acc["miss"],
            "spurious_pred_count": acc["spurious"],
            "fragmentation_rate": float(acc["frag"] / n_gt),
            "over_merge_rate": float(acc["merge"] / n_pred),
            "miss_rate": float(acc["miss"] / n_gt),
            "spurious_rate": float(acc["spurious"] / n_pred),
        }

    overall_stats = format_pq_stats(results_by_split["overall"])

    print("\n" + "=" * 65)
    print(f"OOF Evaluation Results on Fold {target_fold} [{split.upper()} partition] ({len(eval_obs_list)} observations):")
    print(f"  Overall PQ:       {overall_stats['pq']:.4f} (SQ: {overall_stats['sq']:.4f}, RQ: {overall_stats['rq']:.4f})")
    print(f"  True Positives:   {overall_stats['tp']}")
    print(f"  False Positives:  {overall_stats['fp']}")
    print(f"  False Negatives:  {overall_stats['fn']}")
    print(f"  Mean Dice:        {overall_stats['mean_dice']:.4f}")
    print(f"  Diagnostics:      Miss Rate: {overall_stats['miss_rate']:.1%}, Frag Rate: {overall_stats['fragmentation_rate']:.1%}, Merge Rate: {overall_stats['over_merge_rate']:.1%}")
    print("=" * 65 + "\n")

    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "provenance": provenance,
        "is_limited_diagnostic": is_limited,
        "fold": target_fold,
        "split": split,
        "total_evaluated_observations": len(eval_obs_list),
        "postprocess_params": resolved_inf_cfg,
        "resolved_inference_config": resolved_inf_cfg,
        "overall": overall_stats,
        "per_observation": per_obs_reports,
    }

    # Persist report
    reports_dir = root_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"eval_{ckpt_sha256[:8]}_fold{target_fold}_{split}_{'diag' if is_limited else 'full'}.json"
    report_path = reports_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Eval] Detailed evaluation report saved to {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Out-Of-Fold model evaluation with PQ, SQ, RQ, Dice, and error breakdown")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--fold", type=int, default=None, help="Validation fold ID (defaults to checkpoint fold)")
    parser.add_argument("--split", type=str, default="tuning", choices=["tuning", "confirmation", "explored", "all"], help="Validation partition")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation observations (for quick diagnostics)")
    parser.add_argument("--allow-unverified-provenance", action="store_true", help="Permit evaluating legacy checkpoint without provenance metadata")
    parser.add_argument("--method", type=str, default=None, choices=["watershed", "connected_components"], help="Instance extraction method")
    parser.add_argument("--high-threshold", type=float, default=None, help="High confidence hysteresis threshold")
    parser.add_argument("--low-threshold", type=float, default=None, help="Low confidence hysteresis threshold")
    parser.add_argument("--center-threshold", type=float, default=None, help="Center proposal peak threshold")
    parser.add_argument("--boundary-weight", type=float, default=None, help="Boundary suppression weight in watershed energy")
    parser.add_argument("--min-distance", type=int, default=None, help="Marker NMS minimum distance")
    parser.add_argument("--max-peaks", type=int, default=None, help="Maximum center peaks")
    parser.add_argument("--max-instances", type=int, default=None, help="Maximum instances per observation")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum pixel area for instance")
    parser.add_argument("--tile-size", type=int, default=None, help="Tiling window size")
    parser.add_argument("--stride", type=int, default=None, help="Tiling stride step")
    parser.add_argument("--tile-batch-size", type=int, default=None, help="Tiling batch size (e.g. 4 for GTX 1650 4GB)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu')")
    args = parser.parse_args()

    evaluate_oof(
        checkpoint_path=args.checkpoint,
        fold=args.fold,
        split=args.split,
        limit=args.limit,
        allow_unverified_provenance=args.allow_unverified_provenance,
        method=args.method,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        center_threshold=args.center_threshold,
        boundary_weight=args.boundary_weight,
        marker_min_distance=args.min_distance,
        max_peaks=args.max_peaks,
        max_instances=args.max_instances,
        min_area=args.min_area,
        tile_size=args.tile_size,
        stride=args.stride,
        tile_batch_size=args.tile_batch_size,
        device_str=args.device,
    )
