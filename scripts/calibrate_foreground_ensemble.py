#!/usr/bin/env python3
"""Local calibration and gated candidate generation for Two-Component Foreground Probability Ensemble.

Ensemble: p = (1 - alpha) * parent + alpha * B1
  - Parent: epoch 006 (artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt)
  - B1: Arm B epoch 1 (artifacts/runs/run_20261001_171453_ede396/checkpoints/epoch_001.pt)

Stages:
  0. preflight: Cryptographically verify inputs, checkpoint metadata, annotations,
     disjoint fold 0 membership, and dataset image resolution without model or inference.
  1. cache-tuning: Precompute and persist strict FP32 foreground probability maps
     for the 10 physical tuning observations for parent and B1 sequentially.
  2. calibrate: Run the 6-setting Cartesian grid search (alpha in [0.25, 0.50, 0.75],
     low in [0.60, 0.65]), compute strict competition metrics, check C3 anchor reproducibility,
     rank deterministically, and freeze the winning setting before comparison.
  3. evaluate-frozen: If Gate 1 (PQ >= 0.32202621736050074) passes, evaluate ONLY the frozen
     winner on the 9 comparison physical groups / 15 annotator entries.
  4. cache-test: Precompute and persist strict FP32 foreground maps for all 180 test observations
     for parent and B1 sequentially.
  5. generate-frozen: If both Gate 1 and Gate 2 (PQ >= 0.31018747359291294) pass, generate
     candidate 4 submission CSV, selection config, and manifest for all 180 test IDs.
  6. all: Run stages sequentially with strict gate enforcement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import compute_instance_diagnostics
from scripts.calibrate_b_epoch1 import check_deadline, evaluate_setting_on_maps
from src.contracts import InstancePrediction, NATIVE_IMAGE_SHAPE
from src.data.annotations import load_coco_annotations
from src.data.dataset import SolarFilamentDataset, load_solar_image
from src.data.folds import load_frozen_folds_manifest
from src.data.manifest import canonical_observation_id
from src.inference.engine import (
    compute_bytes_sha256,
    compute_file_sha256,
    compute_state_dict_sha256,
    load_cached_prediction,
    predict_full_observation,
    save_cached_prediction,
)
from src.inference.ensemble import (
    compute_ensemble_definition_sha256,
    compute_ensemble_foreground_prob,
)
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import audit_submission_and_manifest
from src.models import build_model

# Constants & Immutable Provenance Expectations
EXPECTED_PARENT_FILE_SHA256 = "9580632d5de717999bb1a60ee940e3f14ee716e853d87d1220be456db62d344a"
EXPECTED_PARENT_STATE_SHA256 = "f7d19980cd2ed382ca4683b6b0932ae404e98359976b5bd9eaebbc44bb4f202c"
DEFAULT_PARENT_PATH = REPO_ROOT / "artifacts" / "runs" / "run_20260930_111126_15cb60" / "checkpoints" / "epoch_006.pt"

EXPECTED_B1_FILE_SHA256 = "8ae7cb726b35d17949a58e9dcd26dd09bf4ecdc7c6f3c10d18d11210e013cf0a"
EXPECTED_B1_STATE_SHA256 = "0824c4972126ca4cd65347c2b13ba58413932c5ffea53675e4355f453b7eadb5"
DEFAULT_B1_PATH = Path(
    r"C:\Users\Xxran\.codex\monitoring\filament-cloud-export-20261001\artifacts\runs\run_20261001_171453_ede396\checkpoints\epoch_001.pt"
)

EXPECTED_FOLDS_MANIFEST_SHA256 = "2495791553a6107873c9962d05a0ba40d25873420937d483641121e7e9c989bd"
EXPECTED_PARTITIONS_SHA256 = "81022d8462bec5031c9b1f3760fe310a9cb15008f00770ab57a62a4f39bff467"
EXPECTED_ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"

# Quality Gates (predeclared iteration 4 gates)
GATE1_TUNING_TARGET = 0.32202621736050074  # candidate-three tuning + 0.002
GATE2_CONF_FLOOR = 0.31018747359291294     # candidate-three comparison - 0.005

# Candidate 3 anchor metrics for reproduction check
C3_ANCHOR_PQ = 0.32002621736050074
C3_ANCHOR_TP = 110
C3_ANCHOR_FP = 159
C3_ANCHOR_FN = 81

# Execution & Memory Control Policy
BATCH_SIZE_POLICY = 4
TILE_SIZE = 512
STRIDE = 256
NORM_MODE = "imagenet"
PRECISION = "float32"
INFERENCE_POLICY = "identity"

# Predeclared 6-setting search grid
GRID_ALPHAS = [0.25, 0.50, 0.75]
GRID_LOW_THRESHOLDS = [0.60, 0.65]
FIXED_HIGH_THRESHOLD = 0.85
FIXED_MIN_AREA = 400
FIXED_MAX_INSTANCES = 20
FIXED_METHOD = "connected_components"


def verify_ensemble_inputs(
    parent_path: Path,
    b1_path: Path,
    expected_parent_sha: str = EXPECTED_PARENT_FILE_SHA256,
    expected_b1_sha: str = EXPECTED_B1_FILE_SHA256,
    partitions_path: Path = REPO_ROOT / "artifacts" / "partitions_migrated_v1.json",
    expected_partitions_sha: str = EXPECTED_PARTITIONS_SHA256,
    manifest_path: Path = REPO_ROOT / "artifacts" / "folds_manifest.json",
    expected_manifest_sha: str = EXPECTED_FOLDS_MANIFEST_SHA256,
    annotations_path: Path = REPO_ROOT / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    expected_annotations_sha: str = EXPECTED_ANNOTATIONS_SHA256,
) -> Dict[str, Any]:
    """Verify cryptographic provenance, exact 565/142 disjoint membership, and metadata of immutable inputs."""
    # Enforce non-empty expected hashes (fail-closed)
    for hname, hval in [
        ("expected_parent_sha", expected_parent_sha),
        ("expected_b1_sha", expected_b1_sha),
        ("expected_partitions_sha", expected_partitions_sha),
        ("expected_manifest_sha", expected_manifest_sha),
        ("expected_annotations_sha", expected_annotations_sha),
    ]:
        if not hval or not isinstance(hval, str) or len(hval) != 64:
            raise ValueError(f"{hname} must be a valid non-empty 64-hex SHA256 string, got {hval!r}")

    # 1. Parent checkpoint verification
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent checkpoint not found at: {parent_path}")
    actual_parent_sha = compute_file_sha256(parent_path)
    if actual_parent_sha != expected_parent_sha:
        raise ValueError(f"Parent checkpoint SHA mismatch!\n  Expected: {expected_parent_sha}\n  Actual:   {actual_parent_sha}")

    parent_ckpt = torch.load(str(parent_path), map_location="cpu", weights_only=False)
    parent_state_sha = compute_state_dict_sha256(parent_ckpt["model_state_dict"])
    if expected_parent_sha == EXPECTED_PARENT_FILE_SHA256 and parent_state_sha != EXPECTED_PARENT_STATE_SHA256:
        raise ValueError(f"Parent model-state SHA mismatch: expected {EXPECTED_PARENT_STATE_SHA256}, got {parent_state_sha}")

    # Check parent weights finiteness
    for k, v in parent_ckpt["model_state_dict"].items():
        if not torch.isfinite(v).all():
            raise ValueError(f"Parent checkpoint contains non-finite weights in parameter '{k}'")

    if parent_ckpt.get("fold") != 0:
        raise ValueError(f"Parent checkpoint fold mismatch: expected fold 0, got {parent_ckpt.get('fold')}")

    # 2. B1 checkpoint verification
    if not b1_path.is_file():
        raise FileNotFoundError(f"B1 checkpoint not found at: {b1_path}")
    actual_b1_sha = compute_file_sha256(b1_path)
    if actual_b1_sha != expected_b1_sha:
        raise ValueError(f"B1 checkpoint SHA mismatch!\n  Expected: {expected_b1_sha}\n  Actual:   {actual_b1_sha}")

    b1_ckpt = torch.load(str(b1_path), map_location="cpu", weights_only=False)
    b1_state_sha = compute_state_dict_sha256(b1_ckpt["model_state_dict"])
    if expected_b1_sha == EXPECTED_B1_FILE_SHA256 and b1_state_sha != EXPECTED_B1_STATE_SHA256:
        raise ValueError(f"B1 model-state SHA mismatch: expected {EXPECTED_B1_STATE_SHA256}, got {b1_state_sha}")

    for k, v in b1_ckpt["model_state_dict"].items():
        if not torch.isfinite(v).all():
            raise ValueError(f"B1 checkpoint contains non-finite weights in parameter '{k}'")

    if b1_ckpt.get("epoch") != 1:
        raise ValueError(f"B1 checkpoint must be epoch 1, got epoch {b1_ckpt.get('epoch')}")
    if b1_ckpt.get("skipped_updates", 0) > 0:
        raise ValueError(f"B1 checkpoint has {b1_ckpt.get('skipped_updates')} skipped updates")
    if b1_ckpt.get("eligible_for_promotion") is not True:
        raise ValueError("B1 checkpoint is not eligible for promotion")
    if b1_ckpt.get("fold") != 0:
        raise ValueError(f"B1 checkpoint fold mismatch: expected fold 0, got {b1_ckpt.get('fold')}")
    if b1_ckpt.get("runtime_config", {}).get("augment_flips") is not True:
        raise ValueError("B1 checkpoint runtime_config must have augment_flips=True")

    # 3. Partitions verification
    if not partitions_path.is_file():
        raise FileNotFoundError(f"Partitions file not found at: {partitions_path}")
    actual_part_sha = compute_file_sha256(partitions_path)
    if actual_part_sha != expected_partitions_sha:
        raise ValueError(f"Partitions SHA mismatch!\n  Expected: {expected_partitions_sha}\n  Actual:   {actual_part_sha}")

    # 4. Manifest verification
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Folds manifest not found at: {manifest_path}")
    actual_man_sha = compute_file_sha256(manifest_path)
    if actual_man_sha != expected_manifest_sha:
        raise ValueError(f"Folds manifest SHA mismatch!\n  Expected: {expected_manifest_sha}\n  Actual:   {actual_man_sha}")

    # 5. Annotations verification
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found at: {annotations_path}")
    actual_ann_sha = compute_file_sha256(annotations_path)
    if actual_ann_sha != expected_annotations_sha:
        raise ValueError(f"Annotations SHA mismatch!\n  Expected: {expected_annotations_sha}\n  Actual:   {actual_ann_sha}")

    # 6. Load partitions and audit observations
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

    # Load fold assignments from manifest and audit full expected canonical sets
    fold_assignments, actual_m_sha_from_helper = load_frozen_folds_manifest(manifest_path, verify_annotations_path=annotations_path)
    if actual_m_sha_from_helper != actual_man_sha:
        raise ValueError(f"Manifest helper SHA mismatch: {actual_m_sha_from_helper} vs {actual_man_sha}")

    expected_val_142 = {canonical_observation_id(k) for k, f in fold_assignments.items() if f == 0}
    expected_train_565 = {canonical_observation_id(k) for k, f in fold_assignments.items() if f != 0}

    if len(expected_val_142) != 142:
        raise ValueError(f"Folds manifest has {len(expected_val_142)} canonical fold 0 validation observations; expected 142")
    if len(expected_train_565) != 565:
        raise ValueError(f"Folds manifest has {len(expected_train_565)} canonical non-fold-0 training observations; expected 565")
    if expected_val_142 & expected_train_565:
        raise ValueError("Folds manifest train and validation sets have non-zero overlap")

    # Verify partition observations belong strictly to fold 0 validation set
    for obs in tuning_obs:
        c_obs = canonical_observation_id(obs)
        if c_obs not in expected_val_142:
            raise ValueError(f"Tuning observation '{obs}' is not in the 142 fold 0 validation set")
    for obs in conf_obs:
        c_obs = canonical_observation_id(obs)
        if c_obs not in expected_val_142:
            raise ValueError(f"Confirmation observation '{obs}' is not in the 142 fold 0 validation set")

    # Verify both checkpoints record exact 565 train and 142 val canonical physical IDs, matching manifest SHA
    for name, ckpt in [("Parent", parent_ckpt), ("B1", b1_ckpt)]:
        ckpt_m_sha = ckpt.get("folds_manifest_sha256")
        if not ckpt_m_sha or ckpt_m_sha != actual_man_sha:
            raise ValueError(
                f"{name} checkpoint recorded folds_manifest_sha256 mismatch: expected {actual_man_sha}, got {ckpt_m_sha}"
            )

        # Check train observations: exact 565 unique canonical IDs
        raw_train = ckpt.get("train_observations")
        if not isinstance(raw_train, list):
            raise ValueError(f"{name} checkpoint missing train_observations list")
        if len(raw_train) != 565:
            raise ValueError(f"{name} checkpoint has {len(raw_train)} train_observations; expected exactly 565")
        if len(set(raw_train)) != 565:
            raise ValueError(f"{name} checkpoint contains duplicate train_observations (unique={len(set(raw_train))})")
        canon_train = {canonical_observation_id(x) for x in raw_train}
        if canon_train != expected_train_565:
            diff = canon_train.symmetric_difference(expected_train_565)
            raise ValueError(f"{name} checkpoint train_observations do not match expected 565 physical training IDs; symmetric diff: {len(diff)}")

        # Check val observations: exact 142 unique canonical IDs
        raw_val = ckpt.get("val_observations")
        if not isinstance(raw_val, list):
            raise ValueError(f"{name} checkpoint missing val_observations list")
        if len(raw_val) != 142:
            raise ValueError(f"{name} checkpoint has {len(raw_val)} val_observations; expected exactly 142")
        if len(set(raw_val)) != 142:
            raise ValueError(f"{name} checkpoint contains duplicate val_observations (unique={len(set(raw_val))})")
        canon_val = {canonical_observation_id(x) for x in raw_val}
        if canon_val != expected_val_142:
            diff = canon_val.symmetric_difference(expected_val_142)
            raise ValueError(f"{name} checkpoint val_observations do not match expected 142 physical validation IDs; symmetric diff: {len(diff)}")

        # Checkpoint internal disjointness
        if canon_train & canon_val:
            raise ValueError(f"{name} checkpoint train and val observation sets overlap: {canon_train & canon_val}")

        # Strict leakage check against partition observations
        overlap = set(raw_train) & set(tuning_obs + conf_obs)
        if overlap:
            raise ValueError(f"Strict leakage violation: {name} train observations overlap with validation: {overlap}")

    return {
        "parent_path": str(parent_path),
        "parent_checkpoint_sha256": actual_parent_sha,
        "parent_model_state_sha256": parent_state_sha,
        "parent_ckpt_data": parent_ckpt,
        "b1_path": str(b1_path),
        "b1_checkpoint_sha256": actual_b1_sha,
        "b1_model_state_sha256": b1_state_sha,
        "b1_ckpt_data": b1_ckpt,
        "partitions_path": str(partitions_path),
        "partitions_sha256": actual_part_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_man_sha,
        "annotations_path": str(annotations_path),
        "annotations_sha256": actual_ann_sha,
        "tuning_observations": tuning_obs,
        "confirmation_observations": conf_obs,
        "fold_assignments": fold_assignments,
    }


def cache_component_maps_sequentially(
    component_name: str,
    ckpt_path: Path,
    ckpt_hash: str,
    model_state_sha: str,
    ckpt_data: Dict[str, Any],
    obs_ids: List[str],
    img_path_resolver: Any,
    cache_dir: Path,
    fallback_cache_dirs: Optional[List[Path]] = None,
    device_str: Optional[str] = None,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    deadline_ts: Optional[float] = None,
) -> None:
    """Precompute and persist FP32 foreground prediction maps for one component using sequential model loading."""
    if tile_batch_size <= 0:
        raise ValueError(f"tile_batch_size must be positive, got {tile_batch_size}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    fallbacks = fallback_cache_dirs or []

    # First pass: identify missing observations
    missing_obs = []
    for obs_id in obs_ids:
        img_path = img_path_resolver(obs_id)
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image for observation {obs_id}: {img_path}")
        img_rgb = load_solar_image(img_path)
        img_sha = compute_bytes_sha256(img_rgb.tobytes())

        cached = None
        for cdir in [cache_dir] + fallbacks:
            if cdir.is_dir():
                cached = load_cached_prediction(
                    cache_dir=cdir,
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
                    # If found in fallback, copy to primary cache_dir for complete provenance
                    if cdir != cache_dir:
                        save_cached_prediction(
                            cache_dir=cache_dir,
                            ckpt_hash=ckpt_hash,
                            obs_id=obs_id,
                            fg=cached[0],
                            ctr=None, bnd=None, off=None,
                            image_sha256=img_sha,
                            model_state_sha256=model_state_sha,
                            tile_size=TILE_SIZE,
                            stride=STRIDE,
                            norm_mode=NORM_MODE,
                            precision=PRECISION,
                            preprocessing_version="v3",
                            inference_policy=INFERENCE_POLICY,
                        )
                    break
        if cached is None or cached[0] is None:
            missing_obs.append(obs_id)

    if not missing_obs:
        print(f"[{component_name}] All {len(obs_ids)} maps already strictly cached.")
        return

    print(f"[{component_name}] Computing {len(missing_obs)} missing maps (loading model sequentially)...")
    is_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_name = device_str if device_str else ("cuda" if is_cuda else "cpu")
    device = torch.device(device_name)

    config = ckpt_data.get("config", {"model": {"name": "resnet34_unet"}})
    model = build_model(config).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    try:
        for idx, obs_id in enumerate(missing_obs, start=1):
            check_deadline(deadline_ts, f"cache map for {component_name} ({obs_id})")
            print(f"  [{component_name}] Computing {idx}/{len(missing_obs)}: {obs_id}...", flush=True)

            img_path = img_path_resolver(obs_id)
            img_rgb = load_solar_image(img_path)
            img_sha = compute_bytes_sha256(img_rgb.tobytes())

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
                    f"GPU OOM while predicting {obs_id} for {component_name} with batch {tile_batch_size}: {oom_err}"
                ) from oom_err

            if not np.isfinite(full_fg).all():
                raise ValueError(f"Non-finite prediction values detected in map for {obs_id}")
            if full_fg.shape != NATIVE_IMAGE_SHAPE:
                raise ValueError(f"Invalid prediction shape {full_fg.shape} for {obs_id}")

            save_cached_prediction(
                cache_dir=cache_dir,
                ckpt_hash=ckpt_hash,
                obs_id=obs_id,
                fg=full_fg,
                ctr=None, bnd=None, off=None,
                image_sha256=img_sha,
                model_state_sha256=model_state_sha,
                tile_size=TILE_SIZE,
                stride=STRIDE,
                norm_mode=NORM_MODE,
                precision=PRECISION,
                preprocessing_version="v3",
                inference_policy=INFERENCE_POLICY,
            )
    finally:
        # Strictly unload model from memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_component_verified_maps(
    ckpt_hash: str,
    model_state_sha: str,
    obs_ids: List[str],
    img_path_resolver: Any,
    cache_dir: Path,
    fallback_cache_dirs: Optional[List[Path]] = None,
) -> Dict[str, np.ndarray]:
    """Load and verify prediction maps for target observations from strict cache."""
    maps: Dict[str, np.ndarray] = {}
    fallbacks = fallback_cache_dirs or []

    for obs_id in obs_ids:
        img_path = img_path_resolver(obs_id)
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image for observation {obs_id}: {img_path}")
        img_rgb = load_solar_image(img_path)
        img_sha = compute_bytes_sha256(img_rgb.tobytes())

        cached = None
        for cdir in [cache_dir] + fallbacks:
            if cdir.is_dir():
                cached = load_cached_prediction(
                    cache_dir=cdir,
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
                    break

        if cached is None or cached[0] is None:
            raise FileNotFoundError(
                f"Strict cache miss or invalid provenance for observation {obs_id} under ckpt {ckpt_hash[:16]}"
            )

        fg_map = cached[0]
        if not np.isfinite(fg_map).all():
            raise ValueError(f"Non-finite values in cached map for {obs_id}")
        if fg_map.shape != NATIVE_IMAGE_SHAPE:
            raise ValueError(f"Map shape mismatch for {obs_id}: {fg_map.shape} vs {NATIVE_IMAGE_SHAPE}")

        maps[obs_id] = fg_map

    return maps


def extract_ensemble_instances(
    fg_parent: np.ndarray,
    fg_b1: np.ndarray,
    alpha: float,
    obs_id: str,
    high_threshold: float,
    low_threshold: float,
    min_area: int,
    max_instances: int,
    method: str = FIXED_METHOD,
) -> Tuple[np.ndarray, List[InstancePrediction]]:
    """Shared single helper to combine foreground maps and extract instances across validation and test."""
    ens_fg = compute_ensemble_foreground_prob(
        fg_parent=fg_parent,
        fg_b1=fg_b1,
        alpha=alpha,
    )
    instances = extract_instances_from_maps(
        foreground_prob=ens_fg,
        obs_id=obs_id,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        min_area=min_area,
        method=method,
        max_instances=max_instances,
        shape=NATIVE_IMAGE_SHAPE,
    )
    return ens_fg, instances


def evaluate_setting_on_maps_with_evidence(
    parent_maps: Dict[str, np.ndarray],
    b1_maps: Dict[str, np.ndarray],
    annotator_variants_by_obs: Dict[str, List[Any]],
    alpha: float,
    high_threshold: float,
    low_threshold: float,
    min_area: int,
    max_instances: int,
    method: str = FIXED_METHOD,
) -> Dict[str, Any]:
    """Evaluate an ensemble configuration across observations, producing aggregate and per-entry evidence."""
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
    per_entry_evidence: List[Dict[str, Any]] = []

    for obs_id in sorted(parent_maps.keys()):
        variants = annotator_variants_by_obs.get(obs_id, [])
        if not variants:
            raise ValueError(f"No annotator variants found for observation {obs_id}")

        ens_fg, instances = extract_ensemble_instances(
            fg_parent=parent_maps[obs_id],
            fg_b1=b1_maps[obs_id],
            alpha=alpha,
            obs_id=obs_id,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            min_area=min_area,
            max_instances=max_instances,
            method=method,
        )

        from src.inference.rle import decode_instance
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

            # Per-entry metrics
            entry_sq = float(iou_sum / tp) if tp > 0 else 0.0
            entry_denom = tp + 0.5 * fp + 0.5 * fn
            entry_rq = float(tp / entry_denom) if entry_denom > 0 else 0.0
            entry_pq = float(entry_sq * entry_rq)

            var_img_id = getattr(var, "annotator_image_id", None) or getattr(var, "file_name", f"{obs_id}_entry")
            per_entry_evidence.append({
                "observation_id": obs_id,
                "annotator_image_id": var_img_id,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "sq": entry_sq,
                "rq": entry_rq,
                "pq": entry_pq,
                "dice": d_val,
                "fragmented_gt_count": int(diag["fragmented_gt_count"]),
                "over_merged_pred_count": int(diag["over_merged_pred_count"]),
                "missed_gt_count": int(diag["missed_gt_count"]),
                "spurious_pred_count": int(diag["spurious_pred_count"]),
                "n_gt": int(diag["n_gt"]),
                "n_pred": int(diag["n_pred"]),
            })

    sq = float(total_iou / total_tp) if total_tp > 0 else 0.0
    denom = total_tp + 0.5 * total_fp + 0.5 * total_fn
    rq = float(total_tp / denom) if denom > 0 else 0.0
    pq = float(sq * rq)
    mean_dice = float(np.mean(total_dice_list)) if total_dice_list else 0.0

    return {
        "config": {
            "alpha": alpha,
            "method": method,
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "min_area": min_area,
            "max_instances": max_instances,
        },
        "metrics": {
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "mean_dice": mean_dice,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_iou": total_iou,
            "entry_count": entry_count,
            "fragmented_gt_count": total_frag,
            "over_merged_pred_count": total_merge,
            "missed_gt_count": total_miss,
            "spurious_pred_count": total_spurious,
            "total_gt": total_gt,
            "total_pred": total_pred,
        },
        "per_entry_evidence": per_entry_evidence,
    }


def validate_frozen_winner_binding(frozen_selection: Union[Dict[str, Any], Path, str], info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that frozen winner artifact binds strictly to current verified inputs and quality gates."""
    if isinstance(frozen_selection, (str, Path)):
        p = Path(frozen_selection)
        if not p.is_file():
            raise FileNotFoundError(f"Frozen winner report not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            frozen_selection = json.load(f)
    elif not isinstance(frozen_selection, dict):
        raise TypeError(f"frozen_selection must be a dict or Path, got {type(frozen_selection).__name__}")

    # Check input cryptographic hashes
    hash_fields = [
        ("parent_checkpoint_sha256", info["parent_checkpoint_sha256"]),
        ("parent_model_state_sha256", info["parent_model_state_sha256"]),
        ("b1_checkpoint_sha256", info["b1_checkpoint_sha256"]),
        ("b1_model_state_sha256", info["b1_model_state_sha256"]),
        ("partitions_sha256", info["partitions_sha256"]),
        ("manifest_sha256", info["manifest_sha256"]),
        ("annotations_sha256", info["annotations_sha256"]),
    ]
    for key, expected_val in hash_fields:
        actual_val = frozen_selection.get(key)
        if actual_val != expected_val:
            raise ValueError(f"Frozen winner report {key} mismatch: expected {expected_val}, got {actual_val}")

    # Validate configuration belongs strictly to the declared 6-setting grid
    cfg = frozen_selection.get("selected_config", {})
    alpha = cfg.get("alpha")
    low_th = cfg.get("low_threshold")
    if alpha not in GRID_ALPHAS:
        raise ValueError(f"Frozen winner alpha {alpha} not in declared grid: {GRID_ALPHAS}")
    if low_th not in GRID_LOW_THRESHOLDS:
        raise ValueError(f"Frozen winner low_threshold {low_th} not in declared grid: {GRID_LOW_THRESHOLDS}")
    if cfg.get("high_threshold") != FIXED_HIGH_THRESHOLD:
        raise ValueError(f"Frozen winner high_threshold mismatch: expected {FIXED_HIGH_THRESHOLD}, got {cfg.get('high_threshold')}")
    if cfg.get("min_area") != FIXED_MIN_AREA:
        raise ValueError(f"Frozen winner min_area mismatch: expected {FIXED_MIN_AREA}, got {cfg.get('min_area')}")
    if cfg.get("max_instances") != FIXED_MAX_INSTANCES:
        raise ValueError(f"Frozen winner max_instances mismatch: expected {FIXED_MAX_INSTANCES}, got {cfg.get('max_instances')}")
    if cfg.get("method") != FIXED_METHOD:
        raise ValueError(f"Frozen winner method mismatch: expected {FIXED_METHOD}, got {cfg.get('method')}")

    # Validate tuning metrics: finite numeric and internal consistency
    t_metrics = frozen_selection.get("tuning_metrics", {})
    t_pq = t_metrics.get("pq")
    if isinstance(t_pq, bool) or not isinstance(t_pq, (int, float)) or not math.isfinite(float(t_pq)):
        raise ValueError(f"Frozen winner tuning PQ must be a finite float, got {t_pq!r}")
    t_pq_float = float(t_pq)

    # Check metric consistency if tp/fp/fn/sq/rq present
    tp = t_metrics.get("tp")
    fp = t_metrics.get("fp")
    fn = t_metrics.get("fn")
    sq = t_metrics.get("sq")
    rq = t_metrics.get("rq")
    if all(x is not None for x in [tp, fp, fn, sq, rq]):
        denom = float(tp) + 0.5 * float(fp) + 0.5 * float(fn)
        calc_rq = (float(tp) / denom) if denom > 0 else 0.0
        if abs(float(rq) - calc_rq) > 1e-4:
            raise ValueError(f"Frozen winner tuning metric inconsistency: recorded RQ={rq}, calculated RQ={calc_rq}")
        calc_pq = float(sq) * float(rq)
        if abs(t_pq_float - calc_pq) > 1e-4:
            raise ValueError(f"Frozen winner tuning metric inconsistency: recorded PQ={t_pq_float}, calculated PQ={calc_pq}")

    # Enforce Gate 1 numerically
    gate1_passed = bool(t_pq_float >= GATE1_TUNING_TARGET)
    if not gate1_passed:
        raise ValueError(f"Frozen winner failed Gate 1: tuning PQ {t_pq_float:.6f} < target {GATE1_TUNING_TARGET:.6f}")

    return frozen_selection


def validate_comparison_report_binding(
    comp_report: Union[Dict[str, Any], Path, str],
    frozen_selection: Union[Dict[str, Any], Path, str],
    info: Dict[str, Any],
    frozen_selection_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate that comparison evaluation report binds strictly to verified inputs, frozen winner, and quality gates."""
    if isinstance(comp_report, (str, Path)):
        p = Path(comp_report)
        if not p.is_file():
            raise FileNotFoundError(f"Comparison report not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            comp_report = json.load(f)
    elif not isinstance(comp_report, dict):
        raise TypeError(f"comp_report must be a dict or Path, got {type(comp_report).__name__}")

    if isinstance(frozen_selection, (str, Path)):
        f_p = Path(frozen_selection)
        if frozen_selection_sha256 is None and f_p.is_file():
            frozen_selection_sha256 = compute_file_sha256(f_p)
        if not f_p.is_file():
            raise FileNotFoundError(f"Frozen winner report not found at: {f_p}")
        with open(f_p, "r", encoding="utf-8") as f:
            frozen_selection = json.load(f)
    # Check hashes against info
    hash_fields = [
        ("parent_checkpoint_sha256", info["parent_checkpoint_sha256"]),
        ("parent_model_state_sha256", info["parent_model_state_sha256"]),
        ("b1_checkpoint_sha256", info["b1_checkpoint_sha256"]),
        ("b1_model_state_sha256", info["b1_model_state_sha256"]),
        ("partitions_sha256", info["partitions_sha256"]),
        ("manifest_sha256", info["manifest_sha256"]),
        ("annotations_sha256", info["annotations_sha256"]),
    ]
    for key, expected_val in hash_fields:
        actual_val = comp_report.get(key)
        if actual_val != expected_val:
            raise ValueError(f"Comparison report {key} mismatch: expected {expected_val}, got {actual_val}")

    # Check frozen selection SHA binding if provided
    if frozen_selection_sha256 is not None:
        rep_f_sha = comp_report.get("frozen_selection_sha256")
        if rep_f_sha != frozen_selection_sha256:
            raise ValueError(
                f"Comparison report frozen selection SHA mismatch: expected {frozen_selection_sha256}, got {rep_f_sha}"
            )

    # Check config binding against frozen winner
    if comp_report.get("frozen_config") != frozen_selection.get("selected_config"):
        raise ValueError("Comparison report frozen_config does not match frozen winner selected_config")

    # Check evaluated physical IDs and entry counts
    eval_ids = comp_report.get("evaluated_physical_ids", [])
    expected_ids = sorted(info["confirmation_observations"])
    if sorted(eval_ids) != expected_ids:
        raise ValueError(f"Comparison report evaluated physical IDs mismatch: expected {expected_ids}, got {sorted(eval_ids)}")
    if comp_report.get("total_entries") != 15:
        raise ValueError(f"Comparison report total entries mismatch: expected 15, got {comp_report.get('total_entries')}")

    # Check confirmation metrics: finite numeric and internal consistency
    c_metrics = comp_report.get("confirmation_metrics", {})
    c_pq = c_metrics.get("pq")
    if isinstance(c_pq, bool) or not isinstance(c_pq, (int, float)) or not math.isfinite(float(c_pq)):
        raise ValueError(f"Comparison report confirmation PQ must be a finite float, got {c_pq!r}")
    c_pq_float = float(c_pq)

    tp = c_metrics.get("tp")
    fp = c_metrics.get("fp")
    fn = c_metrics.get("fn")
    sq = c_metrics.get("sq")
    rq = c_metrics.get("rq")
    if all(x is not None for x in [tp, fp, fn, sq, rq]):
        denom = float(tp) + 0.5 * float(fp) + 0.5 * float(fn)
        calc_rq = (float(tp) / denom) if denom > 0 else 0.0
        if abs(float(rq) - calc_rq) > 1e-4:
            raise ValueError(f"Comparison report metric inconsistency: recorded RQ={rq}, calculated RQ={calc_rq}")
        calc_pq = float(sq) * float(rq)
        if abs(c_pq_float - calc_pq) > 1e-4:
            raise ValueError(f"Comparison report metric inconsistency: recorded PQ={c_pq_float}, calculated PQ={calc_pq}")

    # Enforce Gate 2 numerically
    gate2_passed = bool(c_pq_float >= GATE2_CONF_FLOOR)
    if not gate2_passed:
        raise ValueError(f"Comparison report failed Gate 2: confirmation PQ {c_pq_float:.6f} < floor {GATE2_CONF_FLOOR:.6f}")

    return comp_report


def run_ensemble_calibration_grid(
    parent_maps: Dict[str, np.ndarray],
    b1_maps: Dict[str, np.ndarray],
    annotator_variants_by_obs: Dict[str, List[Any]],
    info: Dict[str, Any],
    reports_dir: Path,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    deadline_ts: Optional[float] = None,
) -> Tuple[Dict[str, Any], Path]:
    """Evaluate C3 anchor reproduction and all 6 ensemble settings, ranking deterministically."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reproduce C3 Anchor Check (pure B1, alpha=1.0, low=0.65)
    print("\n[Anchor Check] Evaluating Candidate 3 baseline reproduction on 10 tuning observations...")
    c3_res = evaluate_setting_on_maps(
        fg_maps=b1_maps,
        annotator_variants_by_obs=annotator_variants_by_obs,
        high_threshold=FIXED_HIGH_THRESHOLD,
        low_threshold=0.65,
        min_area=FIXED_MIN_AREA,
        max_instances=FIXED_MAX_INSTANCES,
    )
    c3_m = c3_res["metrics"]
    print(f"  Reproduced C3 Tuning PQ: {c3_m['pq']:.6f} (Expected: {C3_ANCHOR_PQ:.6f})")
    print(f"  TP: {c3_m['tp']} (Exp: {C3_ANCHOR_TP}), FP: {c3_m['fp']} (Exp: {C3_ANCHOR_FP}), FN: {c3_m['fn']} (Exp: {C3_ANCHOR_FN})")

    # Check floating point reproduction tolerance (< 1e-4)
    if abs(c3_m["pq"] - C3_ANCHOR_PQ) > 1e-4:
        raise ValueError(
            f"Candidate 3 anchor reproduction discrepancy: got PQ {c3_m['pq']:.6f}, expected {C3_ANCHOR_PQ:.6f}"
        )

    # 2. Evaluate exactly 6 ensemble configurations
    print("\n[Calibrate] Evaluating complete 6-setting ensemble grid...")
    grid_results: List[Dict[str, Any]] = []
    setting_idx = 0

    for alpha, low_th in product(GRID_ALPHAS, GRID_LOW_THRESHOLDS):
        setting_idx += 1
        check_deadline(deadline_ts, f"ensemble grid setting {setting_idx}")

        res = evaluate_setting_on_maps_with_evidence(
            parent_maps=parent_maps,
            b1_maps=b1_maps,
            annotator_variants_by_obs=annotator_variants_by_obs,
            alpha=alpha,
            high_threshold=FIXED_HIGH_THRESHOLD,
            low_threshold=low_th,
            min_area=FIXED_MIN_AREA,
            max_instances=FIXED_MAX_INSTANCES,
            method=FIXED_METHOD,
        )
        m = res["metrics"]
        print(
            f"  [{setting_idx}/6] alpha={alpha:.2f} (parent={1.0-alpha:.2f}, b1={alpha:.2f}), low={low_th:.2f} -> "
            f"PQ={m['pq']:.6f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})"
        )
        grid_results.append(res)

    if len(grid_results) != 6:
        raise ValueError(f"Ensemble grid evaluated {len(grid_results)} settings, expected exactly 6")

    # Deterministic ranking:
    # 1. Max PQ (-pq)
    # 2. Min FP (+fp)
    # 3. Weight closest to 0.50 (abs(alpha - 0.50))
    # 4. Low closest to 0.65 (abs(low - 0.65))
    # 5. Numeric alpha ascending, then low ascending
    def rank_key(item: Dict[str, Any]):
        cfg = item["config"]
        met = item["metrics"]
        alpha = cfg["alpha"]
        low = cfg["low_threshold"]
        return (
            -met["pq"],
            met["fp"],
            abs(alpha - 0.50),
            abs(low - 0.65),
            alpha,
            low,
        )

    grid_results.sort(key=rank_key)
    winner = grid_results[0]

    # Save grid report
    grid_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": info["parent_path"],
        "parent_checkpoint_sha256": info["parent_checkpoint_sha256"],
        "b1_checkpoint": info["b1_path"],
        "b1_checkpoint_sha256": info["b1_checkpoint_sha256"],
        "anchor_c3_reproduction": c3_res,
        "evaluated_physical_ids": sorted(list(parent_maps.keys())),
        "total_entries": sum(len(v) for v in annotator_variants_by_obs.values()),
        "winning_setting": winner,
        "grid_results": grid_results,
    }
    grid_path = reports_dir / "foreground_ensemble_calibration_grid.json"
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump(grid_report, f, indent=2)

    # Freeze selected winner configuration BEFORE comparison
    frozen_path = reports_dir / "foreground_ensemble_frozen_winner.json"
    if frozen_path.is_file():
        with open(frozen_path, "r", encoding="utf-8") as f:
            existing_f = json.load(f)
        if (
            existing_f.get("parent_checkpoint_sha256") != info["parent_checkpoint_sha256"]
            or existing_f.get("b1_checkpoint_sha256") != info["b1_checkpoint_sha256"]
            or existing_f.get("partitions_sha256") != info["partitions_sha256"]
        ):
            raise FileExistsError(f"Conflicting existing frozen winner report at {frozen_path}. Refusing to overwrite.")

    frozen_selection = {
        "frozen_timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint_path": info["parent_path"],
        "parent_checkpoint_sha256": info["parent_checkpoint_sha256"],
        "parent_model_state_sha256": info["parent_model_state_sha256"],
        "b1_checkpoint_path": info["b1_path"],
        "b1_checkpoint_sha256": info["b1_checkpoint_sha256"],
        "b1_model_state_sha256": info["b1_model_state_sha256"],
        "partitions_sha256": info["partitions_sha256"],
        "manifest_sha256": info["manifest_sha256"],
        "annotations_sha256": info["annotations_sha256"],
        "tile_batch_size": tile_batch_size,
        "selected_config": winner["config"],
        "tuning_metrics": winner["metrics"],
        "per_entry_evidence": winner["per_entry_evidence"],
        "evaluated_physical_ids": sorted(list(parent_maps.keys())),
        "total_entries": sum(len(v) for v in annotator_variants_by_obs.values()),
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


def evaluate_frozen_ensemble_on_confirmation(
    frozen_selection: Dict[str, Any],
    parent_conf_maps: Dict[str, np.ndarray],
    b1_conf_maps: Dict[str, np.ndarray],
    annotator_variants_by_obs: Dict[str, List[Any]],
    info: Dict[str, Any],
    reports_dir: Path,
    frozen_winner_path: Path,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    deadline_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate exclusively the frozen ensemble winner on the 9 comparison physical groups / 15 annotator entries."""
    check_deadline(deadline_ts, "comparison evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)
    cfg = frozen_selection["selected_config"]
    alpha = cfg["alpha"]
    print(f"\n[Comparison] Evaluating frozen winner (alpha={alpha:.2f}, low={cfg['low_threshold']:.2f}) on 9 confirmation groups...")

    res = evaluate_setting_on_maps_with_evidence(
        parent_maps=parent_conf_maps,
        b1_maps=b1_conf_maps,
        annotator_variants_by_obs=annotator_variants_by_obs,
        alpha=alpha,
        high_threshold=cfg["high_threshold"],
        low_threshold=cfg["low_threshold"],
        min_area=cfg["min_area"],
        max_instances=cfg["max_instances"],
        method=cfg.get("method", FIXED_METHOD),
    )

    conf_pq = float(res["metrics"]["pq"])
    gate2_passed = bool(conf_pq >= GATE2_CONF_FLOOR)
    frozen_selection_sha256 = compute_file_sha256(frozen_winner_path)

    comp_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint_sha256": info["parent_checkpoint_sha256"],
        "parent_model_state_sha256": info["parent_model_state_sha256"],
        "b1_checkpoint_sha256": info["b1_checkpoint_sha256"],
        "b1_model_state_sha256": info["b1_model_state_sha256"],
        "frozen_selection_sha256": frozen_selection_sha256,
        "frozen_config": cfg,
        "tile_batch_size": tile_batch_size,
        "partitions_sha256": info["partitions_sha256"],
        "manifest_sha256": info["manifest_sha256"],
        "annotations_sha256": info["annotations_sha256"],
        "evaluated_physical_ids": sorted(list(parent_conf_maps.keys())),
        "total_entries": sum(len(v) for v in annotator_variants_by_obs.values()),
        "confirmation_metrics": res["metrics"],
        "per_entry_evidence": res["per_entry_evidence"],
        "gate2_floor": GATE2_CONF_FLOOR,
        "gate2_passed": gate2_passed,
    }

    comp_path = reports_dir / "foreground_ensemble_comparison_evaluation.json"
    if comp_path.is_file():
        with open(comp_path, "r", encoding="utf-8") as f:
            existing_comp = json.load(f)
        if (
            existing_comp.get("parent_checkpoint_sha256") != info["parent_checkpoint_sha256"]
            or existing_comp.get("b1_checkpoint_sha256") != info["b1_checkpoint_sha256"]
            or existing_comp.get("frozen_selection_sha256") != frozen_selection_sha256
        ):
            raise FileExistsError(f"Conflicting existing comparison evaluation at {comp_path}. Refusing to overwrite.")

    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_report, f, indent=2)

    print(f"[Comparison] Results:")
    print(f"  Confirmation PQ: {conf_pq:.6f} (Floor: {GATE2_CONF_FLOOR:.6f})")
    print(f"  Gate 2:          {'PASSED' if gate2_passed else 'FAILED'}")
    print(f"  Report saved:    {comp_path}")

    return comp_report


def generate_ensemble_candidate_submission(
    frozen_selection: Dict[str, Any],
    info: Dict[str, Any],
    cache_dir: Path,
    output_csv: Path,
    output_manifest: Path,
    output_cfg: Path,
    tile_batch_size: int = BATCH_SIZE_POLICY,
    deadline_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate audited candidate 4 submission package for all 180 test observations using cached ensemble maps."""
    check_deadline(deadline_ts, "generate candidate submission")
    cfg = frozen_selection["selected_config"]
    alpha = cfg["alpha"]
    test_dir = REPO_ROOT / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"

    if output_csv.is_file():
        raise FileExistsError(f"Submission CSV already exists at {output_csv}. Refusing to overwrite.")

    # Discover test images
    image_paths = sorted(
        list(test_dir.glob("*.jpeg"))
        + list(test_dir.glob("*.jpg"))
        + list(test_dir.glob("*.png"))
    )
    obs_to_img: Dict[str, Path] = {}
    for p in image_paths:
        oid = canonical_observation_id(p.stem)
        if oid not in obs_to_img:
            obs_to_img[oid] = p

    if len(obs_to_img) != 180:
        raise ValueError(f"Expected exactly 180 canonical test observations, got {len(obs_to_img)}")

    expected_obs_ids = set(obs_to_img.keys())

    # Build selection config artifact
    selection_cfg_data = {
        "candidate_id": "candidate_4_foreground_ensemble",
        "description": "Two-component foreground probability ensemble: parent (epoch 006) + B1 (epoch 001)",
        "alpha": alpha,
        "components": [
            {
                "name": "parent",
                "checkpoint_path": str(Path(info["parent_path"]).resolve()),
                "checkpoint_sha256": info["parent_checkpoint_sha256"],
                "model_state_sha256": info["parent_model_state_sha256"],
                "weight": float(1.0 - alpha),
            },
            {
                "name": "b1",
                "checkpoint_path": str(Path(info["b1_path"]).resolve()),
                "checkpoint_sha256": info["b1_checkpoint_sha256"],
                "model_state_sha256": info["b1_model_state_sha256"],
                "weight": float(alpha),
            },
        ],
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
    selection_cfg_sha = compute_file_sha256(output_cfg)

    # Compute ensemble definition SHA
    comp_defs = [
        {
            "name": "parent",
            "checkpoint_sha256": info["parent_checkpoint_sha256"],
            "model_state_sha256": info["parent_model_state_sha256"],
            "weight": float(1.0 - alpha),
        },
        {
            "name": "b1",
            "checkpoint_sha256": info["b1_checkpoint_sha256"],
            "model_state_sha256": info["b1_model_state_sha256"],
            "weight": float(alpha),
        },
    ]
    pp_params = {
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
    }
    ensemble_def_sha = compute_ensemble_definition_sha256(comp_defs, pp_params)

    # Load cached maps and extract instances
    print(f"\n[Generate] Combining component maps and extracting instances for {len(expected_obs_ids)} test observations...")
    csv_rows = []
    manifest_obs = []

    for idx, (obs_id, img_path) in enumerate(sorted(obs_to_img.items()), start=1):
        check_deadline(deadline_ts, f"generation for {obs_id}")
        if idx % 30 == 0 or idx == len(obs_to_img):
            print(f"  [Generate] Processing {idx}/{len(obs_to_img)}: {obs_id}...", flush=True)

        img_rgb = load_solar_image(img_path)
        img_sha = compute_bytes_sha256(img_rgb.tobytes())

        # Load parent test map
        p_cached = load_cached_prediction(
            cache_dir=cache_dir,
            ckpt_hash=info["parent_checkpoint_sha256"],
            obs_id=obs_id,
            expected_image_sha256=img_sha,
            expected_model_state_sha256=info["parent_model_state_sha256"],
            expected_spatial_shape=NATIVE_IMAGE_SHAPE,
            expected_tile_size=TILE_SIZE,
            expected_stride=STRIDE,
            expected_norm_mode=NORM_MODE,
            expected_precision=PRECISION,
            expected_preprocessing_version="v3",
            expected_inference_policy=INFERENCE_POLICY,
            strict=True,
        )
        if p_cached is None or p_cached[0] is None:
            raise FileNotFoundError(f"Missing parent test map for observation {obs_id} in {cache_dir}. Run cache-test stage.")

        # Load B1 test map
        b_cached = load_cached_prediction(
            cache_dir=cache_dir,
            ckpt_hash=info["b1_checkpoint_sha256"],
            obs_id=obs_id,
            expected_image_sha256=img_sha,
            expected_model_state_sha256=info["b1_model_state_sha256"],
            expected_spatial_shape=NATIVE_IMAGE_SHAPE,
            expected_tile_size=TILE_SIZE,
            expected_stride=STRIDE,
            expected_norm_mode=NORM_MODE,
            expected_precision=PRECISION,
            expected_preprocessing_version="v3",
            expected_inference_policy=INFERENCE_POLICY,
            strict=True,
        )
        if b_cached is None or b_cached[0] is None:
            raise FileNotFoundError(f"Missing B1 test map for observation {obs_id} in {cache_dir}. Run cache-test stage.")

        # Use shared instance extraction helper
        ens_fg, instances = extract_ensemble_instances(
            fg_parent=p_cached[0],
            fg_b1=b_cached[0],
            alpha=alpha,
            obs_id=obs_id,
            high_threshold=cfg["high_threshold"],
            low_threshold=cfg["low_threshold"],
            min_area=cfg["min_area"],
            max_instances=cfg["max_instances"],
            method=cfg.get("method", FIXED_METHOD),
        )

        if len(instances) > 0:
            manifest_obs.append({"observation_id": obs_id, "status": "processed", "instance_count": len(instances)})
            for inst in instances:
                csv_rows.append({"filament_id": inst.filament_id, "segmentation_rle": inst.rle_counts})
        else:
            manifest_obs.append({"observation_id": obs_id, "status": "abstained", "instance_count": 0})

    # Write submission CSV with strict QUOTE_NONE
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        for r in csv_rows:
            writer.writerow([r["filament_id"], r["segmentation_rle"]])
    csv_sha = compute_file_sha256(output_csv)

    total_inst = len(csv_rows)
    proc_count = sum(1 for o in manifest_obs if o["status"] == "processed")
    abst_count = sum(1 for o in manifest_obs if o["status"] == "abstained")

    # Write completion manifest
    manifest_data = {
        "csv_sha256": csv_sha,
        "selection_config_path": str(output_cfg.resolve()),
        "selection_config_sha256": selection_cfg_sha,
        "ensemble_definition_sha256": ensemble_def_sha,
        "ensemble_type": "two_component_foreground_ensemble",
        "inference_policy": INFERENCE_POLICY,
        "alpha": alpha,
        "components": [
            {
                "name": "parent",
                "checkpoint_path": str(Path(info["parent_path"]).resolve()),
                "checkpoint_sha256": info["parent_checkpoint_sha256"],
                "model_state_sha256": info["parent_model_state_sha256"],
                "weight": float(1.0 - alpha),
            },
            {
                "name": "b1",
                "checkpoint_path": str(Path(info["b1_path"]).resolve()),
                "checkpoint_sha256": info["b1_checkpoint_sha256"],
                "model_state_sha256": info["b1_model_state_sha256"],
                "weight": float(alpha),
            },
        ],
        "postprocess_params": pp_params,
        "total_test_observations": len(expected_obs_ids),
        "total_instances": total_inst,
        "detected_observations_count": proc_count,
        "abstained_observations_count": abst_count,
        "observations": manifest_obs,
    }
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    man_sha = compute_file_sha256(output_manifest)

    # Audit submission and manifest
    audit_rep = audit_submission_and_manifest(
        csv_path=output_csv,
        manifest_path=output_manifest,
        expected_observation_ids=expected_obs_ids,
    )
    if not audit_rep["is_valid"]:
        raise ValueError(f"Candidate package audit failed: {audit_rep}")

    print(f"\n[Generate] Candidate 4 package created and audited successfully:")
    print(f"  CSV:      {output_csv} (SHA256: {csv_sha})")
    print(f"  Manifest: {output_manifest} (SHA256: {man_sha})")
    print(f"  Config:   {output_cfg} (SHA256: {selection_cfg_sha})")
    print(f"  Ensemble Definition SHA: {ensemble_def_sha}")
    print(f"  Instances: {total_inst}, Processed: {proc_count}, Abstained: {abst_count}")

    return {
        "csv_path": str(output_csv),
        "csv_sha256": csv_sha,
        "manifest_path": str(output_manifest),
        "manifest_sha256": man_sha,
        "selection_config_path": str(output_cfg),
        "selection_config_sha256": selection_cfg_sha,
        "ensemble_definition_sha256": ensemble_def_sha,
        "audit_report": audit_rep,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate Two-Component Foreground Ensemble and Generate Candidate 4")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["preflight", "cache-tuning", "calibrate", "evaluate-frozen", "cache-test", "generate-frozen", "all"],
                        help="Execution stage")
    parser.add_argument("--parent-checkpoint", type=str, default=str(DEFAULT_PARENT_PATH),
                        help="Path to parent epoch 006 checkpoint")
    parser.add_argument("--parent-checkpoint-sha256", type=str, default=EXPECTED_PARENT_FILE_SHA256,
                        help="Expected SHA256 of parent checkpoint")
    parser.add_argument("--b1-checkpoint", type=str, default=str(DEFAULT_B1_PATH),
                        help="Path to Arm B epoch 1 checkpoint")
    parser.add_argument("--b1-checkpoint-sha256", type=str, default=EXPECTED_B1_FILE_SHA256,
                        help="Expected SHA256 of B1 checkpoint")
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
    parser.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "artifacts" / "cache" / "ensemble_v1"),
                        help="Primary cache directory for ensemble component maps")
    parser.add_argument("--b1-fallback-cache-dir", type=str,
                        default=str(REPO_ROOT / "artifacts" / "cache" / "b_epoch1_daily_20261001"),
                        help="Existing cache directory for verified B1 maps")
    parser.add_argument("--reports-dir", type=str, default=str(REPO_ROOT / "artifacts" / "reports" / "ensemble_v1"),
                        help="Reports directory for ensemble calibration")
    parser.add_argument("--tile-batch-size", type=int, default=BATCH_SIZE_POLICY,
                        help="Tiling batch size (memory control, default 4)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--deadline", type=float, default=None,
                        help="Absolute deadline timestamp (seconds since epoch)")
    args = parser.parse_args()

    if args.tile_batch_size <= 0:
        raise ValueError(f"Require positive tile_batch_size (> 0), got {args.tile_batch_size}")

    parent_path = Path(args.parent_checkpoint)
    b1_path = Path(args.b1_checkpoint)
    cache_dir = Path(args.cache_dir)
    b1_fallback = Path(args.b1_fallback_cache_dir)
    reports_dir = Path(args.reports_dir)
    partitions_path = Path(args.partitions)
    manifest_path = Path(args.manifest)
    annotations_path = Path(args.annotations)
    deadline_ts = args.deadline
    if args.stage != "preflight":
        if deadline_ts is None:
            raise ValueError(f"Computational stage '{args.stage}' requires a finite future absolute deadline (--deadline)")
    check_deadline(deadline_ts, "initialization")

    # 1. Cryptographic preflight audit
    info = verify_ensemble_inputs(
        parent_path=parent_path,
        b1_path=b1_path,
        expected_parent_sha=args.parent_checkpoint_sha256,
        expected_b1_sha=args.b1_checkpoint_sha256,
        partitions_path=partitions_path,
        expected_partitions_sha=args.partitions_sha256,
        manifest_path=manifest_path,
        expected_manifest_sha=args.manifest_sha256,
        annotations_path=annotations_path,
        expected_annotations_sha=args.annotations_sha256,
    )

    tuning_obs = info["tuning_observations"]
    conf_obs = info["confirmation_observations"]

    # Load annotations and validation dataset for canonical image resolution
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
        print("=== STAGE 0: ENSEMBLE REAL-INPUT PREFLIGHT PASSED    ===")
        print("========================================================")
        print(f"  Parent Ckpt:   {info['parent_path']} (SHA: {info['parent_checkpoint_sha256'][:16]}...)")
        print(f"  B1 Ckpt:       {info['b1_path']} (SHA: {info['b1_checkpoint_sha256'][:16]}...)")
        print(f"  Partitions:    {info['partitions_path']} (SHA: {info['partitions_sha256'][:16]}...)")
        print(f"  Manifest:      {info['manifest_path']} (SHA: {info['manifest_sha256'][:16]}...)")
        print(f"  Annotations:   {info['annotations_path']} (SHA: {info['annotations_sha256'][:16]}...)")
        print(f"  Tuning Obs:    {len(tuning_obs)} unique IDs ({tune_entry_count} entries)")
        print(f"  Confirm Obs:   {len(conf_obs)} unique IDs ({conf_entry_count} entries)")
        print(f"  Tile Batch:    {args.tile_batch_size}")
        print("Preflight audit completely satisfied with zero inference calls.")
        return

    # Stage: cache-tuning
    if args.stage in ("cache-tuning", "all"):
        check_deadline(deadline_ts, "Stage 1: cache-tuning")
        print("\n========================================================")
        print("=== STAGE 1: CACHING TUNING MAPS (Parent & B1)       ===")
        print("========================================================")
        # 1. Parent tuning maps
        cache_component_maps_sequentially(
            component_name="Parent (epoch 006)",
            ckpt_path=parent_path,
            ckpt_hash=info["parent_checkpoint_sha256"],
            model_state_sha=info["parent_model_state_sha256"],
            ckpt_data=info["parent_ckpt_data"],
            obs_ids=tuning_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )
        # 2. B1 tuning maps
        cache_component_maps_sequentially(
            component_name="B1 (epoch 001)",
            ckpt_path=b1_path,
            ckpt_hash=info["b1_checkpoint_sha256"],
            model_state_sha=info["b1_model_state_sha256"],
            ckpt_data=info["b1_ckpt_data"],
            obs_ids=tuning_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

    # Stage: calibrate
    frozen_selection: Optional[Dict[str, Any]] = None
    frozen_path = reports_dir / "foreground_ensemble_frozen_winner.json"

    if args.stage in ("calibrate", "all"):
        check_deadline(deadline_ts, "Stage 2: calibrate")
        print("\n========================================================")
        print("=== STAGE 2: 6-SETTING ENSEMBLE GRID SEARCH          ===")
        print("========================================================")
        parent_tuning_maps = load_component_verified_maps(
            ckpt_hash=info["parent_checkpoint_sha256"],
            model_state_sha=info["parent_model_state_sha256"],
            obs_ids=tuning_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
        )
        b1_tuning_maps = load_component_verified_maps(
            ckpt_hash=info["b1_checkpoint_sha256"],
            model_state_sha=info["b1_model_state_sha256"],
            obs_ids=tuning_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
        )
        frozen_selection, _ = run_ensemble_calibration_grid(
            parent_maps=parent_tuning_maps,
            b1_maps=b1_tuning_maps,
            annotator_variants_by_obs=variants_tuning,
            info=info,
            reports_dir=reports_dir,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

    # Load frozen selection if running downstream stages directly
    if frozen_selection is None and args.stage in ("evaluate-frozen", "cache-test", "generate-frozen"):
        if not frozen_path.is_file():
            raise FileNotFoundError(f"Frozen winner report not found at {frozen_path}. Run calibrate stage first.")
        with open(frozen_path, "r", encoding="utf-8") as f:
            frozen_selection = json.load(f)

    # Stage: evaluate-frozen
    comp_report: Optional[Dict[str, Any]] = None

    if args.stage in ("evaluate-frozen", "all"):
        check_deadline(deadline_ts, "Stage 3: evaluate-frozen")
        print("\n========================================================")
        print("=== STAGE 3: EVALUATE FROZEN ENSEMBLE ON COMPARISON  ===")
        print("========================================================")
        if frozen_selection is None:
            if not frozen_path.is_file():
                raise FileNotFoundError(f"Frozen winner report not found at {frozen_path}. Run calibrate stage first.")
            with open(frozen_path, "r", encoding="utf-8") as f:
                frozen_selection = json.load(f)

        # Early validation of frozen winner binding and Gate 1 before caching comparison maps
        validate_frozen_winner_binding(frozen_selection, info)

        # Cache comparison maps sequentially
        cache_component_maps_sequentially(
            component_name="Parent (epoch 006)",
            ckpt_path=parent_path,
            ckpt_hash=info["parent_checkpoint_sha256"],
            model_state_sha=info["parent_model_state_sha256"],
            ckpt_data=info["parent_ckpt_data"],
            obs_ids=conf_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )
        cache_component_maps_sequentially(
            component_name="B1 (epoch 001)",
            ckpt_path=b1_path,
            ckpt_hash=info["b1_checkpoint_sha256"],
            model_state_sha=info["b1_model_state_sha256"],
            ckpt_data=info["b1_ckpt_data"],
            obs_ids=conf_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

        parent_conf_maps = load_component_verified_maps(
            ckpt_hash=info["parent_checkpoint_sha256"],
            model_state_sha=info["parent_model_state_sha256"],
            obs_ids=conf_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
        )
        b1_conf_maps = load_component_verified_maps(
            ckpt_hash=info["b1_checkpoint_sha256"],
            model_state_sha=info["b1_model_state_sha256"],
            obs_ids=conf_obs,
            img_path_resolver=val_ds._resolve_image_path,
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
        )
        comp_report = evaluate_frozen_ensemble_on_confirmation(
            frozen_selection=frozen_selection,
            parent_conf_maps=parent_conf_maps,
            b1_conf_maps=b1_conf_maps,
            annotator_variants_by_obs=variants_conf,
            info=info,
            reports_dir=reports_dir,
            frozen_winner_path=frozen_path,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

        # Halt downstream stages if Gate 2 failed during --stage all
        if not comp_report.get("gate2_passed", False):
            c_pq = float(comp_report["confirmation_metrics"]["pq"])
            blocker = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": f"Gate 2 comparison floor failed: PQ {c_pq:.6f} < floor {GATE2_CONF_FLOOR:.6f}",
                "tuning_pq": float(frozen_selection["tuning_metrics"]["pq"]),
                "confirmation_pq": c_pq,
                "gate2_floor": GATE2_CONF_FLOOR,
                "gate2_passed": False,
            }
            with open(reports_dir / "foreground_ensemble_blocker.json", "w", encoding="utf-8") as f:
                json.dump(blocker, f, indent=2)
            print(f"[Gate 2 Failed] Confirmation PQ {c_pq:.6f} < {GATE2_CONF_FLOOR:.6f}.")
            print("Stopping pipeline cleanly before test caching to preserve resources. Blocker report written.")
            return

    # Stage: cache-test
    test_dir = REPO_ROOT / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
    test_images = sorted(list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    test_obs_to_img: Dict[str, Path] = {}
    for p in test_images:
        toid = canonical_observation_id(p.stem)
        if toid not in test_obs_to_img:
            test_obs_to_img[toid] = p
    test_obs_list = sorted(list(test_obs_to_img.keys()))

    if args.stage in ("cache-test", "all"):
        check_deadline(deadline_ts, "Stage 4: cache-test")
        print("\n========================================================")
        print("=== STAGE 4: CACHING TEST MAPS (180 IDs, Parent & B1) ===")
        print("========================================================")
        if frozen_selection is None:
            if not frozen_path.is_file():
                raise FileNotFoundError(f"Frozen winner report not found at {frozen_path}. Run calibrate stage first.")
            with open(frozen_path, "r", encoding="utf-8") as f:
                frozen_selection = json.load(f)
        validate_frozen_winner_binding(frozen_selection, info)

        if comp_report is None:
            comp_path = reports_dir / "foreground_ensemble_comparison_evaluation.json"
            if not comp_path.is_file():
                raise FileNotFoundError(
                    f"Saved comparison report not found at {comp_path}. Stage 'evaluate-frozen' must be completed first."
                )
            with open(comp_path, "r", encoding="utf-8") as f:
                comp_report = json.load(f)
        validate_comparison_report_binding(comp_report, frozen_selection, info, compute_file_sha256(frozen_path))

        cache_component_maps_sequentially(
            component_name="Parent (epoch 006)",
            ckpt_path=parent_path,
            ckpt_hash=info["parent_checkpoint_sha256"],
            model_state_sha=info["parent_model_state_sha256"],
            ckpt_data=info["parent_ckpt_data"],
            obs_ids=test_obs_list,
            img_path_resolver=lambda oid: test_obs_to_img[oid],
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )
        cache_component_maps_sequentially(
            component_name="B1 (epoch 001)",
            ckpt_path=b1_path,
            ckpt_hash=info["b1_checkpoint_sha256"],
            model_state_sha=info["b1_model_state_sha256"],
            ckpt_data=info["b1_ckpt_data"],
            obs_ids=test_obs_list,
            img_path_resolver=lambda oid: test_obs_to_img[oid],
            cache_dir=cache_dir,
            fallback_cache_dirs=[b1_fallback],
            device_str=args.device,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

    # Stage: generate-frozen
    if args.stage in ("generate-frozen", "all"):
        check_deadline(deadline_ts, "Stage 5: generate-frozen")
        print("\n========================================================")
        print("=== STAGE 5: GENERATE CANDIDATE 4 PACKAGE            ===")
        print("========================================================")
        if frozen_selection is None:
            if not frozen_path.is_file():
                raise FileNotFoundError(f"Frozen winner report not found at {frozen_path}. Run calibrate stage first.")
            with open(frozen_path, "r", encoding="utf-8") as f:
                frozen_selection = json.load(f)
        validate_frozen_winner_binding(frozen_selection, info)

        if comp_report is None:
            comp_path = reports_dir / "foreground_ensemble_comparison_evaluation.json"
            if not comp_path.is_file():
                raise FileNotFoundError(
                    f"Saved comparison report not found at {comp_path}. "
                    "Stage 'evaluate-frozen' must be completed before 'generate-frozen'."
                )
            with open(comp_path, "r", encoding="utf-8") as f:
                comp_report = json.load(f)
        validate_comparison_report_binding(comp_report, frozen_selection, info, compute_file_sha256(frozen_path))

        # Recompute quality gates from finite numeric measurements
        t_pq = float(frozen_selection["tuning_metrics"]["pq"])
        c_pq = float(comp_report["confirmation_metrics"]["pq"])
        if not (math.isfinite(t_pq) and math.isfinite(c_pq)):
            raise ValueError(f"Non-finite PQ metrics detected: tuning={t_pq}, confirmation={c_pq}")

        gate1_passed = bool(t_pq >= GATE1_TUNING_TARGET)
        gate2_passed = bool(c_pq >= GATE2_CONF_FLOOR)

        if not (gate1_passed and gate2_passed):
            blocker = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "Quality gates not satisfied for candidate 4 generation",
                "tuning_pq": t_pq,
                "gate1_target": GATE1_TUNING_TARGET,
                "gate1_passed": gate1_passed,
                "confirmation_pq": c_pq,
                "gate2_floor": GATE2_CONF_FLOOR,
                "gate2_passed": gate2_passed,
            }
            with open(reports_dir / "foreground_ensemble_blocker.json", "w", encoding="utf-8") as f:
                json.dump(blocker, f, indent=2)
            print("[Gate Audit] Candidate 4 generation inhibited because quality gates were not fully satisfied:")
            print(f"  Gate 1 (Tuning PQ {t_pq:.6f} >= {GATE1_TUNING_TARGET:.6f}):        {'PASSED' if gate1_passed else 'FAILED'}")
            print(f"  Gate 2 (Confirmation PQ {c_pq:.6f} >= {GATE2_CONF_FLOOR:.6f}):  {'PASSED' if gate2_passed else 'FAILED'}")
            print("Candidate 4 files NOT generated. Blocker report written.")
            return

        out_csv = REPO_ROOT / "artifacts" / "submission_candidate_4_foreground_ensemble.csv"
        out_man = REPO_ROOT / "artifacts" / "submission_candidate_4_foreground_ensemble.manifest.json"
        out_cfg = REPO_ROOT / "artifacts" / "submission_candidate_4_foreground_ensemble.selection_config.json"

        package = generate_ensemble_candidate_submission(
            frozen_selection=frozen_selection,
            info=info,
            cache_dir=cache_dir,
            output_csv=out_csv,
            output_manifest=out_man,
            output_cfg=out_cfg,
            tile_batch_size=args.tile_batch_size,
            deadline_ts=deadline_ts,
        )

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidate_id": "candidate_4_foreground_ensemble",
            "parent_checkpoint_sha256": info["parent_checkpoint_sha256"],
            "b1_checkpoint_sha256": info["b1_checkpoint_sha256"],
            "frozen_config": frozen_selection["selected_config"],
            "tile_batch_size": args.tile_batch_size,
            "tuning_pq": t_pq,
            "confirmation_pq": c_pq,
            "package": package,
        }
        with open(reports_dir / "foreground_ensemble_candidate_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
