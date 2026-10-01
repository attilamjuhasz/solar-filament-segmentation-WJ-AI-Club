#!/usr/bin/env python3
"""Local calibration and gated candidate generation for Arm B Epoch 1.

Stages:
  0. preflight: Cryptographically verify inputs, checkpoint metadata, annotations,
     disjoint fold 0 membership, and dataset image resolution without model or inference.
  1. cache-tuning: Precompute and persist strict FP32 foreground probability maps
     for the 10 physical tuning observations using tile_batch_size=4.
  2. calibrate: Run the complete 48-setting Cartesian grid search over cached maps,
     computing strict competition PQ, TP/FP/FN, SQ/RQ, Dice, and error modes.
     Deterministically rank and freeze the winning setting before comparison.
  3. evaluate-frozen: If Gate 1 is satisfied (PQ >= 0.31764891564223335), precompute
     comparison maps (9 physical groups / 15 entries) and evaluate ONLY the frozen winner.
  4. generate-frozen: If both Gate 1 and Gate 2 (PQ >= 0.30880793475475555) pass,
     generate the candidate submission CSV, selection config, and manifest for all 180 test IDs.
  5. all: Run stages sequentially with strict gate enforcement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import compute_instance_diagnostics
from inference import run_inference
from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.annotations import load_coco_annotations
from src.data.dataset import SolarFilamentDataset, load_solar_image
from src.data.folds import load_frozen_folds_manifest
from src.evaluation.competition_adapter import evaluate_entry_pq
from src.inference.config import resolve_inference_config
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

# Constants & Immutable Provenance Expectations
EXPECTED_CHECKPOINT_SHA256 = "8ae7cb726b35d17949a58e9dcd26dd09bf4ecdc7c6f3c10d18d11210e013cf0a"
EXPECTED_PARENT_FILE_SHA256 = "9580632d5de717999bb1a60ee940e3f14ee716e853d87d1220be456db62d344a"
EXPECTED_FOLDS_MANIFEST_SHA256 = "2495791553a6107873c9962d05a0ba40d25873420937d483641121e7e9c989bd"
EXPECTED_PARTITIONS_SHA256 = "81022d8462bec5031c9b1f3760fe310a9cb15008f00770ab57a62a4f39bff467"
EXPECTED_ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"

DEFAULT_CHECKPOINT_PATH = Path(
    r"C:\Users\Xxran\.codex\monitoring\filament-cloud-export-20261001\artifacts\runs\run_20261001_171453_ede396\checkpoints\epoch_001.pt"
)

# Quality Gates
GATE1_TUNING_TARGET = 0.31764891564223335
GATE2_CONF_FLOOR = 0.30880793475475555

# Memory & Hardware Execution Policy
BATCH_SIZE_POLICY = 4
TILE_SIZE = 512
STRIDE = 256
NORM_MODE = "imagenet"
PRECISION = "float32"
INFERENCE_POLICY = "identity"

# Calibration Grid: 3 * 4 * 2 * 2 = 48 settings
GRID_HIGH_THRESHOLDS = [0.80, 0.85, 0.90]
GRID_LOW_THRESHOLDS = [0.55, 0.65, 0.70, 0.75]
GRID_MIN_AREAS = [400, 800]
GRID_MAX_INSTANCES = [12, 20]
ANCHOR_SETTING = (0.85, 0.70, 400, 20)


def check_deadline(deadline_ts: Optional[float], stage_name: str) -> None:
    """Enforce finite absolute deadline between operations."""
    if deadline_ts is not None:
        if isinstance(deadline_ts, bool) or not isinstance(deadline_ts, (int, float)):
            raise TypeError(f"deadline must be a numeric timestamp, got {type(deadline_ts).__name__}: {deadline_ts!r}")
        if not math.isfinite(float(deadline_ts)):
            raise ValueError(f"deadline must be finite, got: {deadline_ts!r}")
        if time.time() > float(deadline_ts):
            raise TimeoutError(
                f"Absolute deadline exceeded during {stage_name} (now={time.time():.1f}, deadline={float(deadline_ts):.1f}). "
                "Stopping cleanly with preserved caches."
            )


def verify_inputs(
    checkpoint_path: Path,
    expected_ckpt_sha: str = EXPECTED_CHECKPOINT_SHA256,
    partitions_path: Path = REPO_ROOT / "artifacts" / "partitions_migrated_v1.json",
    expected_partitions_sha: str = EXPECTED_PARTITIONS_SHA256,
    manifest_path: Path = REPO_ROOT / "artifacts" / "folds_manifest.json",
    expected_manifest_sha: str = EXPECTED_FOLDS_MANIFEST_SHA256,
    annotations_path: Path = REPO_ROOT / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    expected_annotations_sha: str = EXPECTED_ANNOTATIONS_SHA256,
) -> Dict[str, Any]:
    """Verify cryptographic provenance, disjoint partition integrity, and metadata of immutable inputs."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    actual_ckpt_sha = compute_file_sha256(checkpoint_path)
    if expected_ckpt_sha and actual_ckpt_sha != expected_ckpt_sha:
        raise ValueError(
            f"Checkpoint SHA256 mismatch!\n  Expected: {expected_ckpt_sha}\n  Actual:   {actual_ckpt_sha}"
        )

    if not partitions_path.is_file():
        raise FileNotFoundError(f"Partitions file not found at: {partitions_path}")
    actual_part_sha = compute_file_sha256(partitions_path)
    if expected_partitions_sha and actual_part_sha != expected_partitions_sha:
        raise ValueError(
            f"Partitions SHA256 mismatch!\n  Expected: {expected_partitions_sha}\n  Actual:   {actual_part_sha}"
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Folds manifest not found at: {manifest_path}")
    actual_man_sha = compute_file_sha256(manifest_path)
    if expected_manifest_sha and actual_man_sha != expected_manifest_sha:
        raise ValueError(
            f"Folds manifest SHA256 mismatch!\n  Expected: {expected_manifest_sha}\n  Actual:   {actual_man_sha}"
        )

    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found at: {annotations_path}")
    actual_ann_sha = compute_file_sha256(annotations_path)
    if expected_annotations_sha and actual_ann_sha != expected_annotations_sha:
        raise ValueError(
            f"Annotations SHA256 mismatch!\n  Expected: {expected_annotations_sha}\n  Actual:   {actual_ann_sha}"
        )

    # Load and inspect checkpoint metadata
    ckpt_data = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if ckpt_data.get("epoch") != 1:
        raise ValueError(f"Selected checkpoint must be epoch 1, got epoch {ckpt_data.get('epoch')}")
    if ckpt_data.get("skipped_updates", 0) > 0:
        raise ValueError(f"Checkpoint has {ckpt_data.get('skipped_updates')} skipped updates")
    if ckpt_data.get("eligible_for_promotion") is not True:
        raise ValueError("Checkpoint is not eligible for promotion")
    if ckpt_data.get("fold") != 0:
        raise ValueError(f"Checkpoint fold mismatch: expected fold 0, got {ckpt_data.get('fold')}")

    rc = ckpt_data.get("runtime_config", {})
    if rc.get("augment_flips") is not True:
        raise ValueError(f"Arm B checkpoint must have augment_flips=True, got {rc.get('augment_flips')}")

    model_state_sha = compute_state_dict_sha256(ckpt_data["model_state_dict"])

    # Load partition observation IDs
    with open(partitions_path, "r", encoding="utf-8") as f:
        part_data = json.load(f)
    tuning_obs = list(part_data["tuning"]["canonical_observation_ids"])
    conf_obs = list(part_data["confirmation"]["canonical_observation_ids"])

    if len(tuning_obs) != 10 or len(set(tuning_obs)) != 10:
        raise ValueError(f"Expected exactly 10 unique tuning observations, got {len(set(tuning_obs))}")
    if len(conf_obs) != 9 or len(set(conf_obs)) != 9:
        raise ValueError(f"Expected exactly 9 unique confirmation observations, got {len(set(conf_obs))}")
    if len(set(tuning_obs) & set(conf_obs)) > 0:
        raise ValueError("Tuning and confirmation observations must be completely disjoint")

    # Load fold assignments from manifest and audit disjoint membership
    fold_assignments, actual_m_sha = load_frozen_folds_manifest(manifest_path, verify_annotations_path=annotations_path)
    for obs in tuning_obs + conf_obs:
        f = fold_assignments.get(obs)
        if f is None or f != 0:
            raise ValueError(f"Partition observation {obs} does not belong to fold 0 in manifest (fold={f})")

    # Checkpoint training observations must be strictly disjoint from fold 0 / partitions
    train_obs = ckpt_data.get("train_observations", [])
    if train_obs:
        overlap = set(train_obs) & set(tuning_obs + conf_obs)
        if overlap:
            raise ValueError(f"Strict leakage violation: checkpoint train observations overlap with validation partition: {overlap}")
        for tobs in train_obs:
            tf = fold_assignments.get(tobs)
            if tf is not None and tf == 0:
                raise ValueError(f"Strict leakage violation: train observation {tobs} belongs to target fold 0!")

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_ckpt_sha,
        "model_state_sha256": model_state_sha,
        "partitions_path": str(partitions_path),
        "partitions_sha256": actual_part_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_man_sha,
        "annotations_path": str(annotations_path),
        "annotations_sha256": actual_ann_sha,
        "tuning_observations": tuning_obs,
        "confirmation_observations": conf_obs,
        "ckpt_data": ckpt_data,
        "fold_assignments": fold_assignments,
    }


def cache_observation_maps(
    val_ds: SolarFilamentDataset,
    obs_ids: List[str],
    checkpoint_info: Dict[str, Any],
    cache_dir: Path,
    device_str: Optional[str] = None,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    deadline_ts: Optional[float] = None,
) -> Dict[str, Path]:
    """Compute and persist strict FP32 foreground prediction maps for target observations."""
    if tile_batch_size <= 0:
        raise ValueError(f"tile_batch_size must be positive, got {tile_batch_size}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_name = device_str if device_str else ("cuda" if is_cuda else "cpu")
    device = torch.device(device_name)
    print(f"[Cache] Using device: {device}, tile_batch_size: {tile_batch_size}")

    ckpt_data = checkpoint_info["ckpt_data"]
    ckpt_hash = checkpoint_info["checkpoint_sha256"]
    model_state_sha = checkpoint_info["model_state_sha256"]

    config = ckpt_data.get("config", {"model": {"name": "resnet34_unet"}})
    model = build_model(config).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    cached_map_paths: Dict[str, Path] = {}

    for idx, obs_id in enumerate(obs_ids, start=1):
        check_deadline(deadline_ts, f"cache map for observation {obs_id}")
        print(f"[Cache] Processing {idx}/{len(obs_ids)}: {obs_id}...", flush=True)

        # Locate image file using canonical dataset mapping
        img_path = val_ds._resolve_image_path(obs_id)
        if not img_path.is_file():
            raise FileNotFoundError(f"Resolved image path does not exist for observation: {obs_id} -> {img_path}")

        img_rgb = load_solar_image(img_path)
        img_sha = compute_bytes_sha256(img_rgb.tobytes())

        # Check existing strict cache
        cached = load_cached_prediction(
            cache_dir=cache_dir,
            ckpt_hash=ckpt_hash,
            obs_id=obs_id,
            expected_image_sha256=img_sha,
            expected_model_state_sha256=model_state_sha,
            expected_spatial_shape=NATIVE_IMAGE_SHAPE,
            expected_tile_size=TILE_SIZE,
            expected_stride=STRIDE,
            expected_norm_mode=NORM_MODE,
            expected_precision=PRECISION,
            expected_preprocessing_version="v3",
            expected_inference_policy=INFERENCE_POLICY,
            strict=True,
        )

        if cached is not None and cached[0] is not None:
            full_fg = cached[0]
            print(f"  [Cache hit] Loaded valid cached map for {obs_id}")
        else:
            try:
                full_fg, _, _, _ = predict_full_observation(
                    model=model,
                    image_rgb=img_rgb,
                    device=device,
                    tile_size=TILE_SIZE,
                    stride=STRIDE,
                    tile_batch_size=tile_batch_size,
                    norm_mode=NORM_MODE,
                    include_aux=False,
                )
            except torch.cuda.OutOfMemoryError as oom_err:
                raise RuntimeError(
                    f"GPU OOM while predicting observation {obs_id} with tile_batch_size={tile_batch_size}: {oom_err}"
                ) from oom_err

            if not np.isfinite(full_fg).all():
                raise ValueError(f"Non-finite prediction values detected in map for {obs_id}")
            if full_fg.shape != NATIVE_IMAGE_SHAPE:
                raise ValueError(f"Invalid prediction shape {full_fg.shape} for {obs_id}, expected {NATIVE_IMAGE_SHAPE}")

            save_cached_prediction(
                cache_dir=cache_dir,
                ckpt_hash=ckpt_hash,
                obs_id=obs_id,
                fg=full_fg,
                ctr=None,
                bnd=None,
                off=None,
                image_sha256=img_sha,
                model_state_sha256=model_state_sha,
                tile_size=TILE_SIZE,
                stride=STRIDE,
                norm_mode=NORM_MODE,
                precision=PRECISION,
                preprocessing_version="v3",
                inference_policy=INFERENCE_POLICY,
            )
            print(f"  [Cache write] Saved verified map for {obs_id}")

        cache_file = cache_dir / f"{ckpt_hash[:16]}_{obs_id}.npz"
        cached_map_paths[obs_id] = cache_file

    return cached_map_paths


def load_verified_maps(
    val_ds: SolarFilamentDataset,
    obs_ids: List[str],
    checkpoint_info: Dict[str, Any],
    cache_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load and verify all prediction maps for target observations from strict cache."""
    maps: Dict[str, np.ndarray] = {}

    for obs_id in obs_ids:
        img_path = val_ds._resolve_image_path(obs_id)
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image for observation {obs_id}: {img_path}")
        img_rgb = load_solar_image(img_path)
        img_sha = compute_bytes_sha256(img_rgb.tobytes())

        cached = load_cached_prediction(
            cache_dir=cache_dir,
            ckpt_hash=checkpoint_info["checkpoint_sha256"],
            obs_id=obs_id,
            expected_image_sha256=img_sha,
            expected_model_state_sha256=checkpoint_info["model_state_sha256"],
            expected_spatial_shape=NATIVE_IMAGE_SHAPE,
            expected_tile_size=TILE_SIZE,
            expected_stride=STRIDE,
            expected_norm_mode=NORM_MODE,
            expected_precision=PRECISION,
            expected_preprocessing_version="v3",
            expected_inference_policy=INFERENCE_POLICY,
            strict=True,
        )

        if cached is None or cached[0] is None:
            raise FileNotFoundError(f"Strict cache miss or invalid provenance for observation {obs_id} in {cache_dir}")

        fg_map = cached[0]
        if not np.isfinite(fg_map).all():
            raise ValueError(f"Non-finite values in cached map for {obs_id}")
        if fg_map.shape != NATIVE_IMAGE_SHAPE:
            raise ValueError(f"Map shape mismatch for {obs_id}: {fg_map.shape} vs {NATIVE_IMAGE_SHAPE}")

        maps[obs_id] = fg_map

    return maps


def evaluate_setting_on_maps(
    fg_maps: Dict[str, np.ndarray],
    annotator_variants_by_obs: Dict[str, List[Any]],
    high_threshold: float,
    low_threshold: float,
    min_area: int,
    max_instances: int,
) -> Dict[str, Any]:
    """Evaluate a single postprocessing configuration across all observations and annotator variants."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    total_dice_list: List[float] = []
    total_frag = 0
    total_merge = 0
    total_miss = 0
    total_spurious = 0
    total_gt = 0
    total_pred = 0
    entry_count = 0

    for obs_id, fg_map in fg_maps.items():
        variants = annotator_variants_by_obs.get(obs_id, [])
        if not variants:
            raise ValueError(f"No annotator variants found for observation {obs_id}")

        # Extract predictions
        instances = extract_instances_from_maps(
            foreground_prob=fg_map,
            obs_id=obs_id,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            min_area=min_area,
            method="connected_components",
            max_instances=max_instances,
            shape=NATIVE_IMAGE_SHAPE,
        )
        pred_masks = [decode_instance(inst.rle_counts, shape=NATIVE_IMAGE_SHAPE) for inst in instances]
        pred_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
        for pm in pred_masks:
            pred_union |= (pm > 0)

        for var in variants:
            entry_count += 1
            gt_masks = [inst.get_mask(NATIVE_IMAGE_SHAPE) for inst in var.instances if inst.area > 0]
            gt_masks = [m for m in gt_masks if (m > 0).any()]

            iou_sum, tp, fp, fn, _, _ = evaluate_entry_pq(gt_masks, pred_masks)
            diag = compute_instance_diagnostics(gt_masks, pred_masks)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_iou += iou_sum

            total_frag += diag["fragmented_gt_count"]
            total_merge += diag["over_merged_pred_count"]
            total_miss += diag["missed_gt_count"]
            total_spurious += diag["spurious_pred_count"]
            total_gt += diag["n_gt"]
            total_pred += diag["n_pred"]

            gt_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
            for gm in gt_masks:
                gt_union |= (gm > 0)

            inter = 2.0 * np.logical_and(gt_union, pred_union).sum()
            union = gt_union.sum() + pred_union.sum()
            d_val = float((inter + 1e-6) / (union + 1e-6))
            total_dice_list.append(d_val)

    sq = float(total_iou / total_tp) if total_tp > 0 else 0.0
    denom = total_tp + 0.5 * total_fp + 0.5 * total_fn
    rq = float(total_tp / denom) if denom > 0 else 0.0
    pq = float(sq * rq)
    mean_dice = float(np.mean(total_dice_list)) if total_dice_list else 0.0

    n_gt_norm = max(1, total_gt)
    n_pred_norm = max(1, total_pred)

    # Compute knobs changed from anchor
    changed_knobs = (
        (1 if abs(high_threshold - ANCHOR_SETTING[0]) > 1e-6 else 0)
        + (1 if abs(low_threshold - ANCHOR_SETTING[1]) > 1e-6 else 0)
        + (1 if min_area != ANCHOR_SETTING[2] else 0)
        + (1 if max_instances != ANCHOR_SETTING[3] else 0)
    )

    return {
        "config": {
            "method": "connected_components",
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "min_area": min_area,
            "max_instances": max_instances,
        },
        "changed_knobs": changed_knobs,
        "metrics": {
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "mean_dice": mean_dice,
            "total_iou": total_iou,
            "total_gt_instances": total_gt,
            "total_pred_instances": total_pred,
            "fragmented_gt_count": total_frag,
            "over_merged_pred_count": total_merge,
            "missed_gt_count": total_miss,
            "spurious_pred_count": total_spurious,
            "fragmentation_rate": float(total_frag / n_gt_norm),
            "over_merge_rate": float(total_merge / n_pred_norm),
            "miss_rate": float(total_miss / n_gt_norm),
            "spurious_rate": float(total_spurious / n_pred_norm),
            "evaluated_observations": len(fg_maps),
            "evaluated_entries": entry_count,
        },
    }


def run_calibration_grid(
    tuning_maps: Dict[str, np.ndarray],
    annotator_variants_by_obs: Dict[str, List[Any]],
    checkpoint_info: Dict[str, Any],
    reports_dir: Path,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    deadline_ts: Optional[float] = None,
) -> Tuple[Dict[str, Any], Path]:
    """Execute complete 48-setting grid search and determine frozen winning configuration."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    grid_results: List[Dict[str, Any]] = []

    print(f"\n[Calibrate] Evaluating complete 48-setting Cartesian grid...")
    setting_idx = 0
    for high, low, area, max_inst in product(
        GRID_HIGH_THRESHOLDS, GRID_LOW_THRESHOLDS, GRID_MIN_AREAS, GRID_MAX_INSTANCES
    ):
        setting_idx += 1
        check_deadline(deadline_ts, f"calibration grid setting {setting_idx}")
        res = evaluate_setting_on_maps(
            fg_maps=tuning_maps,
            annotator_variants_by_obs=annotator_variants_by_obs,
            high_threshold=high,
            low_threshold=low,
            min_area=area,
            max_instances=max_inst,
        )
        m = res["metrics"]
        print(
            f"  [{setting_idx:02d}/48] high={high:.2f}, low={low:.2f}, area={area}, cap={max_inst} -> "
            f"PQ={m['pq']:.6f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})"
        )
        grid_results.append(res)

    if len(grid_results) != 48:
        raise ValueError(f"Grid search evaluated {len(grid_results)} settings, expected exactly 48")

    # Deterministic ranking:
    # 1. Max PQ
    # 2. Min FP
    # 3. Smallest changed knobs from anchor
    # 4. Deterministic lexicographic: (-high, low, area, cap)
    def rank_key(item: Dict[str, Any]):
        cfg = item["config"]
        met = item["metrics"]
        return (
            -met["pq"],
            met["fp"],
            item["changed_knobs"],
            -cfg["high_threshold"],
            cfg["low_threshold"],
            cfg["min_area"],
            cfg["max_instances"],
        )

    grid_results.sort(key=rank_key)
    winner = grid_results[0]

    # Check for conflicting existing frozen winner to protect against unintended overwrite
    frozen_path = reports_dir / "b_epoch1_frozen_winner.json"
    if frozen_path.is_file():
        with open(frozen_path, "r", encoding="utf-8") as f:
            existing_f = json.load(f)
        if (
            existing_f.get("checkpoint_sha256") != checkpoint_info["checkpoint_sha256"]
            or existing_f.get("partitions_sha256") != checkpoint_info["partitions_sha256"]
            or existing_f.get("manifest_sha256") != checkpoint_info["manifest_sha256"]
        ):
            raise FileExistsError(
                f"Conflicting existing frozen winner report at {frozen_path}. "
                "Refusing to overwrite divergent run results."
            )

    # Save complete grid report
    grid_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_info["checkpoint_path"],
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "total_settings_evaluated": len(grid_results),
        "anchor_setting": {
            "high_threshold": ANCHOR_SETTING[0],
            "low_threshold": ANCHOR_SETTING[1],
            "min_area": ANCHOR_SETTING[2],
            "max_instances": ANCHOR_SETTING[3],
        },
        "winning_setting": winner,
        "grid_results": grid_results,
    }
    grid_path = reports_dir / "b_epoch1_calibration_grid.json"
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump(grid_report, f, indent=2)

    # Freeze selected winner configuration BEFORE comparison
    frozen_selection = {
        "frozen_timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": checkpoint_info["checkpoint_path"],
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "model_state_sha256": checkpoint_info["model_state_sha256"],
        "partitions_sha256": checkpoint_info["partitions_sha256"],
        "manifest_sha256": checkpoint_info["manifest_sha256"],
        "annotations_sha256": checkpoint_info["annotations_sha256"],
        "tile_batch_size": tile_batch_size,
        "selected_config": winner["config"],
        "tuning_metrics": winner["metrics"],
        "changed_knobs_from_anchor": winner["changed_knobs"],
        "gate1_target": GATE1_TUNING_TARGET,
        "gate1_passed": bool(winner["metrics"]["pq"] >= GATE1_TUNING_TARGET),
    }
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen_selection, f, indent=2)

    print(f"\n[Calibrate] Winning setting selected and frozen:")
    print(f"  Config:     {winner['config']}")
    print(f"  Tuning PQ:  {winner['metrics']['pq']:.6f} (Gate 1 Target: {GATE1_TUNING_TARGET:.6f})")
    print(f"  Gate 1:     {'PASSED' if frozen_selection['gate1_passed'] else 'FAILED'}")
    print(f"  Frozen at:  {frozen_path}")

    return frozen_selection, frozen_path


def evaluate_frozen_winner_on_confirmation(
    frozen_selection: Dict[str, Any],
    conf_maps: Dict[str, np.ndarray],
    annotator_variants_by_obs: Dict[str, List[Any]],
    checkpoint_info: Dict[str, Any],
    reports_dir: Path,
    frozen_winner_path: Path,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    deadline_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate exclusively the frozen winner on the 9 comparison physical groups / 15 annotator entries."""
    check_deadline(deadline_ts, "comparison evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)
    cfg = frozen_selection["selected_config"]
    print(f"\n[Comparison] Evaluating frozen winner on 9 confirmation groups...")

    res = evaluate_setting_on_maps(
        fg_maps=conf_maps,
        annotator_variants_by_obs=annotator_variants_by_obs,
        high_threshold=cfg["high_threshold"],
        low_threshold=cfg["low_threshold"],
        min_area=cfg["min_area"],
        max_instances=cfg["max_instances"],
    )

    conf_pq = float(res["metrics"]["pq"])
    gate2_passed = bool(conf_pq >= GATE2_CONF_FLOOR)
    frozen_selection_sha256 = compute_file_sha256(frozen_winner_path)

    comp_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": checkpoint_info["checkpoint_path"],
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "model_state_sha256": checkpoint_info["model_state_sha256"],
        "frozen_selection_sha256": frozen_selection_sha256,
        "frozen_config": cfg,
        "tile_batch_size": tile_batch_size,
        "partitions_sha256": checkpoint_info["partitions_sha256"],
        "manifest_sha256": checkpoint_info["manifest_sha256"],
        "annotations_sha256": checkpoint_info["annotations_sha256"],
        "evaluated_physical_ids": sorted(list(conf_maps.keys())),
        "total_entries": sum(len(v) for v in annotator_variants_by_obs.values()),
        "confirmation_metrics": res["metrics"],
        "gate2_floor": GATE2_CONF_FLOOR,
        "gate2_passed": gate2_passed,
    }

    comp_path = reports_dir / "b_epoch1_comparison_evaluation.json"
    if comp_path.is_file():
        with open(comp_path, "r", encoding="utf-8") as f:
            existing_comp = json.load(f)
        if (
            existing_comp.get("checkpoint_sha256") != checkpoint_info["checkpoint_sha256"]
            or existing_comp.get("frozen_selection_sha256") != frozen_selection_sha256
        ):
            raise FileExistsError(
                f"Conflicting existing comparison evaluation report at {comp_path}. "
                "Refusing to overwrite divergent run results."
            )

    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_report, f, indent=2)

    print(f"[Comparison] Results:")
    print(f"  Confirmation PQ: {conf_pq:.6f} (Floor: {GATE2_CONF_FLOOR:.6f})")
    print(f"  Gate 2:          {'PASSED' if gate2_passed else 'FAILED'}")
    print(f"  Report saved:    {comp_path}")

    return comp_report


def generate_candidate_submission(
    frozen_selection: Dict[str, Any],
    checkpoint_path: Path,
    output_csv: Path,
    output_manifest: Path,
    output_cfg: Path,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    device_str: Optional[str] = None,
    deadline_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate audited candidate submission files using the frozen winning postprocessing settings."""
    check_deadline(deadline_ts, "generate candidate submission")
    cfg = frozen_selection["selected_config"]
    test_dir = REPO_ROOT / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"

    if output_csv.is_file():
        raise FileExistsError(f"Submission CSV already exists at {output_csv}. Refusing to overwrite.")

    selection_cfg_data = {
        "candidate_id": "b_epoch1_calibrated",
        "description": "Local postprocessing calibration of preserved Arm B Epoch 1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": frozen_selection["checkpoint_sha256"],
        "inference_config": {
            "method": cfg["method"],
            "high_threshold": cfg["high_threshold"],
            "low_threshold": cfg["low_threshold"],
            "min_area": cfg["min_area"],
            "max_instances": cfg["max_instances"],
            "tile_size": TILE_SIZE,
            "stride": STRIDE,
            "tile_batch_size": tile_batch_size,
            "norm_mode": NORM_MODE,
            "precision": PRECISION,
            "inference_policy": INFERENCE_POLICY,
        },
        "tuning_metrics": frozen_selection["tuning_metrics"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_cfg, "w", encoding="utf-8") as f:
        json.dump(selection_cfg_data, f, indent=2)

    print(f"\n[Generate] Running test inference with tile_batch_size={tile_batch_size}...")
    audit_rep = run_inference(
        checkpoint_path=str(checkpoint_path),
        test_images_dir=str(test_dir),
        output_csv=str(output_csv),
        output_manifest=str(output_manifest),
        method=cfg["method"],
        high_threshold=cfg["high_threshold"],
        low_threshold=cfg["low_threshold"],
        min_area=cfg["min_area"],
        max_instances=cfg["max_instances"],
        tile_size=TILE_SIZE,
        stride=STRIDE,
        tile_batch_size=tile_batch_size,
        device_str=device_str,
    )

    if not audit_rep["is_valid"]:
        raise ValueError(f"Candidate package audit failed: {audit_rep['errors']}")

    csv_sha = compute_file_sha256(output_csv)
    man_sha = compute_file_sha256(output_manifest)
    cfg_sha = compute_file_sha256(output_cfg)

    print(f"[Generate] Candidate package created and audited successfully:")
    print(f"  CSV:      {output_csv} (SHA256: {csv_sha})")
    print(f"  Manifest: {output_manifest} (SHA256: {man_sha})")
    print(f"  Config:   {output_cfg} (SHA256: {cfg_sha})")

    return {
        "csv_path": str(output_csv),
        "csv_sha256": csv_sha,
        "manifest_path": str(output_manifest),
        "manifest_sha256": man_sha,
        "selection_config_path": str(output_cfg),
        "selection_config_sha256": cfg_sha,
        "audit_report": audit_rep,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate Arm B Epoch 1 and Generate Local Candidate")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["preflight", "cache-tuning", "calibrate", "evaluate-frozen", "generate-frozen", "all"],
                        help="Execution stage")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH),
                        help="Path to Arm B epoch 1 checkpoint")
    parser.add_argument("--checkpoint-sha256", type=str, default=EXPECTED_CHECKPOINT_SHA256,
                        help="Expected SHA256 of checkpoint")
    parser.add_argument("--partitions", type=str, default=str(REPO_ROOT / "artifacts" / "partitions_migrated_v1.json"),
                        help="Path to partitions artifact")
    parser.add_argument("--partitions-sha256", type=str, default=EXPECTED_PARTITIONS_SHA256,
                        help="Expected SHA256 of partitions artifact")
    parser.add_argument("--manifest", type=str, default=str(REPO_ROOT / "artifacts" / "folds_manifest.json"),
                        help="Path to folds manifest")
    parser.add_argument("--manifest-sha256", type=str, default=EXPECTED_FOLDS_MANIFEST_SHA256,
                        help="Expected SHA256 of folds manifest")
    parser.add_argument("--annotations", type=str,
                        default=str(REPO_ROOT / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
                        help="Path to train annotations JSON")
    parser.add_argument("--annotations-sha256", type=str, default=EXPECTED_ANNOTATIONS_SHA256,
                        help="Expected SHA256 of train annotations")
    parser.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "artifacts" / "cache"),
                        help="Cache directory for predicted maps")
    parser.add_argument("--reports-dir", type=str, default=str(REPO_ROOT / "artifacts" / "reports"),
                        help="Reports directory")
    parser.add_argument("--tile-batch-size", type=int, default=BATCH_SIZE_POLICY,
                        help="Tiling batch size (memory control, default 4)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--deadline", type=float, default=None,
                        help="Absolute deadline timestamp (seconds since epoch)")
    args = parser.parse_args()

    if args.tile_batch_size <= 0:
        raise ValueError(f"Require positive tile_batch_size (> 0), got {args.tile_batch_size}")

    ckpt_path = Path(args.checkpoint)
    cache_dir = Path(args.cache_dir)
    reports_dir = Path(args.reports_dir)
    partitions_path = Path(args.partitions)
    manifest_path = Path(args.manifest)
    annotations_path = Path(args.annotations)
    deadline_ts = args.deadline

    check_deadline(deadline_ts, "initialization")

    # 1. Cryptographic preflight audit
    info = verify_inputs(
        checkpoint_path=ckpt_path,
        expected_ckpt_sha=args.checkpoint_sha256,
        partitions_path=partitions_path,
        expected_partitions_sha=args.partitions_sha256,
        manifest_path=manifest_path,
        expected_manifest_sha=args.manifest_sha256,
        annotations_path=annotations_path,
        expected_annotations_sha=args.annotations_sha256,
    )

    tuning_obs = info["tuning_observations"]
    conf_obs = info["confirmation_observations"]

    # Load annotations & dataset for canonical image resolution
    annotation_index = load_coco_annotations(str(annotations_path))
    val_ds = SolarFilamentDataset(
        images_dir=REPO_ROOT / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images",
        annotation_index=annotation_index,
        fold_assignments=info["fold_assignments"],
        target_fold=0,
        is_train=False,
        patch_size=(2048, 2048),
    )

    variants_tuning = {obs: val_ds.get_observation_annotations(obs) for obs in tuning_obs}
    variants_conf = {obs: val_ds.get_observation_annotations(obs) for obs in conf_obs}

    tune_entry_count = sum(len(v) for v in variants_tuning.values())
    conf_entry_count = sum(len(v) for v in variants_conf.values())

    if tune_entry_count != 24:
        raise ValueError(f"Expected exactly 24 tuning annotator entries, got {tune_entry_count}")
    if conf_entry_count != 15:
        raise ValueError(f"Expected exactly 15 confirmation annotator entries, got {conf_entry_count}")

    # Stage: preflight
    if args.stage == "preflight":
        print("\n========================================================")
        print("=== STAGE 0: REAL-INPUT PREFLIGHT AUDIT PASSED       ===")
        print("========================================================")
        print(f"  Checkpoint:   {info['checkpoint_path']} (SHA: {info['checkpoint_sha256'][:16]}...)")
        print(f"  Partitions:   {info['partitions_path']} (SHA: {info['partitions_sha256'][:16]}...)")
        print(f"  Manifest:     {info['manifest_path']} (SHA: {info['manifest_sha256'][:16]}...)")
        print(f"  Annotations:  {info['annotations_path']} (SHA: {info['annotations_sha256'][:16]}...)")
        print(f"  Tuning Obs:   {len(tuning_obs)} unique IDs ({tune_entry_count} entries)")
        print(f"  Confirm Obs:  {len(conf_obs)} unique IDs ({conf_entry_count} entries)")
        print(f"  Tile Batch:   {args.tile_batch_size}")
        print("Preflight verification completely satisfied with zero inference calls.")
        return

    # Stage: cache-tuning
    if args.stage in ("cache-tuning", "all"):
        check_deadline(deadline_ts, "Stage 1: cache-tuning")
        print("\n========================================================")
        print("=== STAGE 1: CACHING TUNING MAPS (10 physical groups) ===")
        print("========================================================")
        cache_observation_maps(
            val_ds=val_ds,
            obs_ids=tuning_obs,
            checkpoint_info=info,
            cache_dir=cache_dir,
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

    # Stage: calibrate
    frozen_selection: Optional[Dict[str, Any]] = None
    frozen_path = reports_dir / "b_epoch1_frozen_winner.json"
    if args.stage in ("calibrate", "all"):
        check_deadline(deadline_ts, "Stage 2: calibrate")
        print("\n========================================================")
        print("=== STAGE 2: 48-SETTING CALIBRATION GRID SEARCH       ===")
        print("========================================================")
        tuning_maps = load_verified_maps(val_ds, tuning_obs, info, cache_dir)
        frozen_selection, _ = run_calibration_grid(
            tuning_maps=tuning_maps,
            annotator_variants_by_obs=variants_tuning,
            checkpoint_info=info,
            reports_dir=reports_dir,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

    # Load frozen selection if running downstream stages directly
    if frozen_selection is None and args.stage in ("evaluate-frozen", "generate-frozen"):
        if not frozen_path.is_file():
            raise FileNotFoundError(f"Frozen winner report not found at {frozen_path}. Run calibrate stage first.")
        with open(frozen_path, "r", encoding="utf-8") as f:
            frozen_selection = json.load(f)

    # Stage: evaluate-frozen
    t_pq = float(frozen_selection["tuning_metrics"]["pq"]) if frozen_selection else 0.0
    gate1_passed = bool(t_pq >= GATE1_TUNING_TARGET)
    comp_report: Optional[Dict[str, Any]] = None

    if args.stage in ("evaluate-frozen", "all"):
        check_deadline(deadline_ts, "Stage 3: evaluate-frozen")
        print("\n========================================================")
        print("=== STAGE 3: EVALUATE FROZEN WINNER ON COMPARISON    ===")
        print("========================================================")
        if not gate1_passed:
            print(f"[Gate 1 Failed] Tuning PQ {t_pq:.6f} < {GATE1_TUNING_TARGET:.6f}.")
            print("Refusing to evaluate comparison set to prevent leakage.")
            return

        cache_observation_maps(
            val_ds=val_ds,
            obs_ids=conf_obs,
            checkpoint_info=info,
            cache_dir=cache_dir,
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )
        conf_maps = load_verified_maps(val_ds, conf_obs, info, cache_dir)
        comp_report = evaluate_frozen_winner_on_confirmation(
            frozen_selection=frozen_selection,
            conf_maps=conf_maps,
            annotator_variants_by_obs=variants_conf,
            checkpoint_info=info,
            reports_dir=reports_dir,
            frozen_winner_path=frozen_path,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

    # Stage: generate-frozen
    if args.stage in ("generate-frozen", "all"):
        check_deadline(deadline_ts, "Stage 4: generate-frozen")
        print("\n========================================================")
        print("=== STAGE 4: GENERATE CANDIDATE SUBMISSION PACKAGE   ===")
        print("========================================================")
        if comp_report is None:
            comp_path = reports_dir / "b_epoch1_comparison_evaluation.json"
            if not comp_path.is_file():
                raise FileNotFoundError(
                    f"Saved comparison report not found at {comp_path}. "
                    "Stage 'evaluate-frozen' must be completed before 'generate-frozen'."
                )
            with open(comp_path, "r", encoding="utf-8") as f:
                comp_report = json.load(f)

        # Cryptographically validate binding between comparison report, frozen winner, and inputs
        frozen_winner_sha256 = compute_file_sha256(frozen_path)
        if comp_report.get("checkpoint_sha256") != info["checkpoint_sha256"]:
            raise ValueError(
                f"Comparison report checkpoint SHA mismatch: expected {info['checkpoint_sha256']}, "
                f"got {comp_report.get('checkpoint_sha256')}"
            )
        if comp_report.get("model_state_sha256") != info["model_state_sha256"]:
            raise ValueError("Comparison report model state SHA mismatch")
        if comp_report.get("frozen_selection_sha256") != frozen_winner_sha256:
            raise ValueError(
                f"Comparison report frozen selection SHA mismatch: expected {frozen_winner_sha256}, "
                f"got {comp_report.get('frozen_selection_sha256')}"
            )
        if comp_report.get("frozen_config") != frozen_selection["selected_config"]:
            raise ValueError("Comparison report config mismatch with frozen winner config")
        if comp_report.get("partitions_sha256") != info["partitions_sha256"]:
            raise ValueError("Comparison report partitions SHA mismatch")
        if comp_report.get("manifest_sha256") != info["manifest_sha256"]:
            raise ValueError("Comparison report manifest SHA mismatch")
        if comp_report.get("annotations_sha256") != info["annotations_sha256"]:
            raise ValueError("Comparison report annotations SHA mismatch")
        if sorted(comp_report.get("evaluated_physical_ids", [])) != sorted(conf_obs):
            raise ValueError("Comparison report evaluated physical IDs mismatch")
        if comp_report.get("total_entries") != 15:
            raise ValueError(f"Comparison report entry count mismatch: expected 15, got {comp_report.get('total_entries')}")

        # Recompute quality gates from finite numeric measurements
        t_pq = float(frozen_selection["tuning_metrics"]["pq"])
        c_pq = float(comp_report["confirmation_metrics"]["pq"])
        if not (math.isfinite(t_pq) and math.isfinite(c_pq)):
            raise ValueError(f"Non-finite PQ metrics detected: tuning={t_pq}, confirmation={c_pq}")

        gate1_passed = bool(t_pq >= GATE1_TUNING_TARGET)
        gate2_passed = bool(c_pq >= GATE2_CONF_FLOOR)

        if not (gate1_passed and gate2_passed):
            print("[Gate Audit] Candidate generation inhibited because quality gates were not fully satisfied:")
            print(f"  Gate 1 (Tuning PQ {t_pq:.6f} >= {GATE1_TUNING_TARGET:.6f}):        {'PASSED' if gate1_passed else 'FAILED'}")
            print(f"  Gate 2 (Confirmation PQ {c_pq:.6f} >= {GATE2_CONF_FLOOR:.6f}):  {'PASSED' if gate2_passed else 'FAILED'}")
            print("Candidate submission files NOT generated. Preserving existing submissions.")
            return

        out_csv = REPO_ROOT / "artifacts" / "submission_candidate_3_b_epoch1_calibrated.csv"
        out_man = REPO_ROOT / "artifacts" / "submission_candidate_3_b_epoch1_calibrated.manifest.json"
        out_cfg = REPO_ROOT / "artifacts" / "submission_candidate_3_b_epoch1_calibrated.selection_config.json"

        package = generate_candidate_submission(
            frozen_selection=frozen_selection,
            checkpoint_path=ckpt_path,
            output_csv=out_csv,
            output_manifest=out_man,
            output_cfg=out_cfg,
            tile_batch_size=args.tile_batch_size,
            device_str=args.device,
            deadline_ts=deadline_ts,
        )

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidate_id": "candidate_3_b_epoch1_calibrated",
            "checkpoint_sha256": info["checkpoint_sha256"],
            "frozen_config": frozen_selection["selected_config"],
            "tile_batch_size": args.tile_batch_size,
            "tuning_pq": t_pq,
            "confirmation_pq": c_pq,
            "package": package,
        }
        with open(reports_dir / "b_epoch1_candidate_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
