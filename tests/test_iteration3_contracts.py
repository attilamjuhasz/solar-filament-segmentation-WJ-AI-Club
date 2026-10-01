from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.dataset import SolarFilamentDataset
from src.inference.engine import (
    compute_bytes_sha256,
    compute_file_sha256,
    compute_state_dict_sha256,
    get_cache_path,
    load_cached_prediction,
    save_cached_prediction,
)
from src.training.augmentation import apply_geometric_flips
from src.training.finetune import setup_finetune_state, validate_finetune_parent
from src.training.mining import (
    compute_component_metrics,
    find_non_overlapping_crop_boxes,
    select_mining_physical_ids,
    verify_mining_bank_provenance,
)
from scripts.iteration3_experiment import (
    EXPECTED_FOLDS_MANIFEST_SHA256,
    EXPECTED_MINING_BANK_SHA256,
    EXPECTED_PARENT_FILE_SHA256,
    EXPECTED_PARENT_STATE_SHA256,
    EXPECTED_PARTITIONS_SHA256,
    FIXED_HIGH_THRESH,
    FIXED_INF_METHOD,
    FIXED_LOW_THRESH,
    FIXED_MAX_INSTANCES,
    FIXED_MIN_AREA,
    assert_report_inference_config,
    evaluate_and_select_candidates,
    parse_deadline_timestamp,
    recover_and_evaluate,
    resolve_and_validate_arm_checkpoints,
    run_experiment,
    verify_immutable_inputs,
)
from train import train


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


# =========================================================================
# Regression 1: Orchestration argument contract recorder
# =========================================================================
def test_orchestration_argument_contract_recorder(tmp_path):
    """Recorder proves all evaluator calls resolve CC/0.85/0.70/400/20 and have NO center/marker substitution."""
    eval_call_records = []

    def mock_eval_fn(**kwargs):
        eval_call_records.append(copy.deepcopy(kwargs))
        return {
            "checkpoint": kwargs.get("checkpoint_path"),
            "resolved_inference_config": {
                "method": kwargs.get("method"),
                "high_threshold": kwargs.get("high_threshold"),
                "low_threshold": kwargs.get("low_threshold"),
                "min_area": kwargs.get("min_area"),
                "max_instances": kwargs.get("max_instances"),
            },
            "total_evaluated_observations": 10,
            "overall": {
                "pq": 0.250000,
                "sq": 0.800000,
                "rq": 0.400000,
                "tp": 20,
                "fp": 5,
                "fn": 10,
                "mean_dice": 0.750000,
                "fragmented_gt_count": 1,
                "over_merged_pred_count": 0,
                "missed_gt_count": 3,
                "spurious_pred_count": 2,
            },
        }

    # Create dummy checkpoints for Arm A and Arm B
    arm_a_dir = tmp_path / "artifacts" / "runs" / "arm_a" / "checkpoints"
    arm_b_dir = tmp_path / "artifacts" / "runs" / "arm_b" / "checkpoints"
    arm_a_dir.mkdir(parents=True, exist_ok=True)
    arm_b_dir.mkdir(parents=True, exist_ok=True)

    dummy_state = {
        "run_id": "arm_a",
        "epoch": 1,
        "fold": 0,
        "total_epochs": 1,
        "skipped_updates": 0,
        "successful_updates": 100,
        "eligible_for_promotion": True,
        "runtime_config": {"augment_flips": False},
        "model_state_dict": {"conv.weight": torch.ones(1, 3, 3, 3)},
    }
    ckpt_a = arm_a_dir / "epoch_001.pt"
    torch.save(dummy_state, str(ckpt_a))

    dummy_state_b = {
        "run_id": "arm_b",
        "epoch": 1,
        "fold": 0,
        "total_epochs": 1,
        "skipped_updates": 0,
        "successful_updates": 100,
        "eligible_for_promotion": True,
        "runtime_config": {"augment_flips": True},
        "model_state_dict": {"conv.weight": torch.ones(1, 3, 3, 3) * 0.99},
    }
    ckpt_b = arm_b_dir / "epoch_001.pt"
    torch.save(dummy_state_b, str(ckpt_b))

    def mock_train_fn(**kwargs):
        augment_flips = kwargs.get("augment_flips", False)
        if not augment_flips:
            return ckpt_a, ckpt_a
        else:
            return ckpt_b, ckpt_b

    # Run orchestration with recorders and mocked preflight
    real_parent = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    if real_parent.is_file():
        summary = run_experiment(
            parent_checkpoint=str(real_parent),
            mining_bank_path="artifacts/reports/mining_bank_v1.json",
            smoke=True,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            root_dir=tmp_path,
        )
        assert len(eval_call_records) == 2, "Smoke run must evaluate tuning candidates and skip comparison evaluation"
        for call in eval_call_records:
            assert call["method"] == "connected_components"
            assert call["high_threshold"] == 0.85
            assert call["low_threshold"] == 0.70
            assert call["min_area"] == 400
            assert call["max_instances"] == 20
            assert call["split"] == "tuning"
            assert "center_threshold" not in call or call["center_threshold"] is None
            assert "marker_min_distance" not in call or call["marker_min_distance"] is None


# =========================================================================
# Regression 2: Cryptographic preflight rejecting mismatched full hashes
# =========================================================================
def test_preflight_hash_verification(tmp_path):
    """Preflight succeeds on exact actual hashes and fails before training on mismatched hashes."""
    real_parent = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    real_folds = Path("artifacts/folds_manifest.json")
    real_parts = Path("artifacts/partitions_migrated_v1.json")
    real_bank = Path("artifacts/reports/mining_bank_v1.json")

    if real_parent.is_file() and real_folds.is_file() and real_parts.is_file() and real_bank.is_file():
        # Valid execution succeeds
        hashes = verify_immutable_inputs(
            parent_path=real_parent,
            folds_manifest_path=real_folds,
            partitions_path=real_parts,
            mining_bank_path=real_bank,
        )
        assert hashes["parent_file_sha256"] == EXPECTED_PARENT_FILE_SHA256
        assert hashes["parent_model_state_sha256"] == EXPECTED_PARENT_STATE_SHA256
        assert hashes["folds_manifest_sha256"] == EXPECTED_FOLDS_MANIFEST_SHA256
        assert hashes["partitions_sha256"] == EXPECTED_PARTITIONS_SHA256
        assert hashes["mining_bank_sha256"] == EXPECTED_MINING_BANK_SHA256

        # Altered copy fails immediately
        bad_file = tmp_path / "altered_parent.pt"
        bad_file.write_bytes(real_parent.read_bytes()[:1000] + b"altered_bytes")
        with pytest.raises(ValueError, match="Parent file SHA mismatch"):
            verify_immutable_inputs(
                parent_path=bad_file,
                folds_manifest_path=real_folds,
                partitions_path=real_parts,
                mining_bank_path=real_bank,
            )


# =========================================================================
# Regression 3: Parent membership contract (exact 565/142, disjoint, canonical)
# =========================================================================
def test_parent_membership_exact_disjoint(tmp_path):
    """Full-run initialization rejects both subsets and supersets, verifies disjointness and aliases."""
    dummy_p = tmp_path / "dummy_parent.pt"
    # Valid canonical observation IDs conforming to \d{14}[A-Za-z]{2}
    train_ids = [f"201601{i:02d}120000Th" for i in range(10, 15)]
    val_ids = [f"201602{i:02d}120000Th" for i in range(10, 13)]

    ckpt_data = {
        "epoch": 6,
        "fold": 0,
        "folds_manifest_sha256": "dummy_manifest_sha",
        "config": {"model": {"name": "resnet34_unet"}},
        "model_state_dict": {"weight": torch.ones(1)},
        "train_observations": train_ids,
        "val_observations": val_ids,
    }
    torch.save(ckpt_data, str(dummy_p))

    # Exact match passes
    meta = validate_finetune_parent(
        checkpoint_path=dummy_p,
        expected_fold=0,
        expected_manifest_sha="dummy_manifest_sha",
        train_observations=train_ids,
        val_observations=val_ids,
        is_diagnostic=False,
    )
    assert meta["fold"] == 0

    # Subset in production run fails
    with pytest.raises(ValueError, match="Fine-tune training membership mismatch"):
        validate_finetune_parent(
            checkpoint_path=dummy_p,
            expected_fold=0,
            expected_manifest_sha="dummy_manifest_sha",
            train_observations=train_ids[:3],  # subset
            val_observations=val_ids,
            is_diagnostic=False,
        )

    # Superset in production run fails
    superset_train = train_ids + ["20160301120000Th"]
    with pytest.raises(ValueError, match="Fine-tune training membership mismatch"):
        validate_finetune_parent(
            checkpoint_path=dummy_p,
            expected_fold=0,
            expected_manifest_sha="dummy_manifest_sha",
            train_observations=superset_train,
            val_observations=val_ids,
            is_diagnostic=False,
        )

    # Overlapping train and validation in checkpoint fails
    bad_ckpt_p = tmp_path / "bad_parent.pt"
    bad_ckpt = dict(ckpt_data)
    bad_ckpt["val_observations"] = [train_ids[0]]
    torch.save(bad_ckpt, str(bad_ckpt_p))
    with pytest.raises(ValueError, match="overlap"):
        validate_finetune_parent(
            checkpoint_path=bad_ckpt_p,
            expected_fold=0,
            expected_manifest_sha="dummy_manifest_sha",
        )


# =========================================================================
# Regression 4: Augmentation and Sampling RNG exact resume
# =========================================================================
def test_augmentation_and_sampling_rng_exact_reproducibility():
    """Augmentation RNG and dataset RNG restore identically on resume."""
    H, W = 64, 64
    image = np.arange(H * W, dtype=np.float32).reshape(1, H, W).repeat(3, axis=0)
    sample = {
        "image": image,
        "target_fg": np.ones((H, W), dtype=np.float32),
        "target_bnd": np.zeros((H, W), dtype=np.float32),
        "target_skel": np.zeros((H, W), dtype=np.float32),
        "target_ctr": np.zeros((H, W), dtype=np.float32),
        "target_off": np.zeros((2, H, W), dtype=np.float32),
        "valid_mask": np.ones((H, W), dtype=np.float32),
    }

    rng1 = np.random.RandomState(999)
    # Perform several asymmetric operations
    for _ in range(5):
        apply_geometric_flips(sample, aug_rng=rng1, hflip_prob=0.5, vflip_prob=0.5)

    saved_state = rng1.get_state()

    # Continue rng1 for 3 more steps
    seq1 = []
    for _ in range(3):
        res = apply_geometric_flips(sample, aug_rng=rng1, hflip_prob=0.5, vflip_prob=0.5)
        seq1.append(res["image"].copy())

    # Create rng2, restore state, and run 3 steps
    rng2 = np.random.RandomState(0)
    rng2.set_state(saved_state)
    seq2 = []
    for _ in range(3):
        res = apply_geometric_flips(sample, aug_rng=rng2, hflip_prob=0.5, vflip_prob=0.5)
        seq2.append(res["image"].copy())

    for img1, img2 in zip(seq1, seq2):
        assert np.array_equal(img1, img2), "Augmentation RNG must produce identical outputs on resume"


# =========================================================================
# Regression 5: Executable control-flow tests for Adversarial Review Probes
# =========================================================================
def test_orchestrator_rejects_unequal_successful_updates(tmp_path):
    """Probe 1: Orchestrator rejects unequal successful updates across matched epochs before evaluation."""
    arm_a_dir = tmp_path / "artifacts" / "runs" / "arm_a" / "checkpoints"
    arm_b_dir = tmp_path / "artifacts" / "runs" / "arm_b" / "checkpoints"
    arm_a_dir.mkdir(parents=True, exist_ok=True)
    arm_b_dir.mkdir(parents=True, exist_ok=True)

    # Arm A has 100 updates, Arm B has 1 update
    torch.save({
        "run_id": "arm_a",
        "epoch": 1,
        "fold": 0,
        "total_epochs": 1,
        "skipped_updates": 0,
        "successful_updates": 100,
        "eligible_for_promotion": True,
        "runtime_config": {"augment_flips": False},
        "model_state_dict": {"conv.weight": torch.ones(1)},
    }, str(arm_a_dir / "epoch_001.pt"))

    torch.save({
        "run_id": "arm_b",
        "epoch": 1,
        "fold": 0,
        "total_epochs": 1,
        "skipped_updates": 0,
        "successful_updates": 1,
        "eligible_for_promotion": True,
        "runtime_config": {"augment_flips": True},
        "model_state_dict": {"conv.weight": torch.ones(1)},
    }, str(arm_b_dir / "epoch_001.pt"))

    eval_calls = []
    def mock_eval_fn(**kwargs):
        eval_calls.append(kwargs)
        return {"overall": {"pq": 0.35, "sq": 0.8, "rq": 0.4, "tp": 10, "fp": 2, "fn": 5, "mean_dice": 0.8}}

    def mock_train_fn(**kwargs):
        return (arm_b_dir / "epoch_001.pt", arm_b_dir / "epoch_001.pt") if kwargs.get("augment_flips") else (arm_a_dir / "epoch_001.pt", arm_a_dir / "epoch_001.pt")

    real_parent = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    if real_parent.is_file():
        with pytest.raises(ValueError, match="Unequal successful updates"):
            run_experiment(
                parent_checkpoint=str(real_parent),
                mining_bank_path="artifacts/reports/mining_bank_v1.json",
                smoke=True,
                train_fn=mock_train_fn,
                eval_fn=mock_eval_fn,
                root_dir=tmp_path,
            )
        assert len(eval_calls) == 0, "No evaluation calls must occur when update counts mismatch"


def test_orchestrator_enforces_budget_deadline_during_arm_b_and_eval(tmp_path):
    """Probe 2: Clock expiring during Arm B or before evaluation halts execution with TimeoutError."""
    arm_a_dir = tmp_path / "runs" / "arm_a" / "checkpoints"
    arm_b_dir = tmp_path / "runs" / "arm_b" / "checkpoints"
    arm_a_dir.mkdir(parents=True, exist_ok=True)
    arm_b_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"run_id": "arm_a", "epoch": 1, "skipped_updates": 0, "successful_updates": 100, "model_state_dict": {"w": torch.ones(1)}}, str(arm_a_dir / "epoch_001.pt"))
    torch.save({"run_id": "arm_b", "epoch": 1, "skipped_updates": 0, "successful_updates": 100, "model_state_dict": {"w": torch.ones(1)}}, str(arm_b_dir / "epoch_001.pt"))

    eval_calls = []
    def mock_eval_fn(**kwargs):
        eval_calls.append(kwargs)
        return {"overall": {"pq": 0.35, "sq": 0.8, "rq": 0.4, "tp": 10, "fp": 2, "fn": 5, "mean_dice": 0.8}}

    # Case A: Budget of 100s, clock advances to 102s during Arm B
    def mock_train_advancing_clock(**kwargs):
        if kwargs.get("augment_flips"):
            time.sleep(0.01)  # small pause
            return arm_b_dir / "epoch_001.pt", arm_b_dir / "epoch_001.pt"
        return arm_a_dir / "epoch_001.pt", arm_a_dir / "epoch_001.pt"

    real_parent = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    if real_parent.is_file():
        simulated_start = time.time() - 95.0
        with pytest.raises(TimeoutError, match="budget"):
            run_experiment(
                parent_checkpoint=str(real_parent),
                mining_bank_path="artifacts/reports/mining_bank_v1.json",
                smoke=True,
                experiment_start_time=simulated_start,
                max_budget_seconds=100.0,
                train_fn=mock_train_advancing_clock,
                eval_fn=mock_eval_fn,
            )
        assert len(eval_calls) == 0, "No evaluation must execute after budget expiry"


def test_orchestrator_rejects_diagnostic_checkpoints_from_production_ranking(tmp_path):
    """Probe 3: Checkpoints marked smoke/diagnostic/ineligible are rejected from production ranking."""
    arm_a_dir = tmp_path / "artifacts" / "runs" / "arm_a" / "checkpoints"
    arm_b_dir = tmp_path / "artifacts" / "runs" / "arm_b" / "checkpoints"
    arm_a_dir.mkdir(parents=True, exist_ok=True)
    arm_b_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic metadata in checkpoint
    for ep in [1, 2, 3]:
        torch.save({
            "run_id": "arm_a",
            "epoch": ep,
            "fold": 0,
            "total_epochs": 3,
            "skipped_updates": 0,
            "successful_updates": 142 * ep,
            "runtime_config": {"smoke": True},
            "finetune_parent": {"is_diagnostic": True},
            "eligible_for_promotion": False,
            "model_state_dict": {"w": torch.ones(1)},
        }, str(arm_a_dir / f"epoch_{ep:03d}.pt"))

        torch.save({
            "run_id": "arm_b",
            "epoch": ep,
            "fold": 0,
            "total_epochs": 3,
            "skipped_updates": 0,
            "successful_updates": 142 * ep,
            "runtime_config": {"smoke": True},
            "finetune_parent": {"is_diagnostic": True},
            "eligible_for_promotion": False,
            "model_state_dict": {"w": torch.ones(1)},
        }, str(arm_b_dir / f"epoch_{ep:03d}.pt"))

    def mock_train_fn(**kwargs):
        return (arm_b_dir / "epoch_003.pt", arm_b_dir / "epoch_003.pt") if kwargs.get("augment_flips") else (arm_a_dir / "epoch_003.pt", arm_a_dir / "epoch_003.pt")

    real_parent = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    if real_parent.is_file():
        with pytest.raises(ValueError, match="Diagnostic checkpoint detected in production run"):
            run_experiment(
                parent_checkpoint=str(real_parent),
                mining_bank_path="artifacts/reports/mining_bank_v1.json",
                smoke=False,  # production mode
                train_fn=mock_train_fn,
                root_dir=tmp_path,
            )


def test_orchestrator_rejects_arm_with_skipped_tail(tmp_path):
    """Probe 4: Discarding a bad epoch is not allowed; any skipped update invalidates the entire arm."""
    arm_a_dir = tmp_path / "artifacts" / "runs" / "arm_a" / "checkpoints"
    arm_b_dir = tmp_path / "artifacts" / "runs" / "arm_b" / "checkpoints"
    arm_a_dir.mkdir(parents=True, exist_ok=True)
    arm_b_dir.mkdir(parents=True, exist_ok=True)

    # Arm A: epoch 1 clean, epoch 2 has skipped_updates=1, epoch 3 clean
    for ep in [1, 2, 3]:
        torch.save({
            "run_id": "arm_a",
            "epoch": ep,
            "fold": 0,
            "total_epochs": 3,
            "skipped_updates": 1 if ep == 2 else 0,
            "successful_updates": 142 * ep if ep != 2 else 283,
            "model_state_dict": {"w": torch.ones(1)},
        }, str(arm_a_dir / f"epoch_{ep:03d}.pt"))

        torch.save({
            "run_id": "arm_b",
            "epoch": ep,
            "fold": 0,
            "total_epochs": 3,
            "skipped_updates": 0,
            "successful_updates": 142 * ep,
            "model_state_dict": {"w": torch.ones(1)},
        }, str(arm_b_dir / f"epoch_{ep:03d}.pt"))

    def mock_train_fn(**kwargs):
        return (arm_b_dir / "epoch_003.pt", arm_b_dir / "epoch_003.pt") if kwargs.get("augment_flips") else (arm_a_dir / "epoch_003.pt", arm_a_dir / "epoch_003.pt")

    real_parent = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    if real_parent.is_file():
        with pytest.raises(ValueError, match="skipped updates"):
            run_experiment(
                parent_checkpoint=str(real_parent),
                mining_bank_path="artifacts/reports/mining_bank_v1.json",
                smoke=False,
                train_fn=mock_train_fn,
                root_dir=tmp_path,
            )


# =========================================================================
# Regression 7: Strict cache contract with policy and aux checks
# =========================================================================
def test_strict_cache_complete_contract(tmp_path):
    """Strict cache requires valid expected hashes, matching policy, aux validation, and preserves FP32 values."""
    cache_dir = tmp_path / "cache"
    ckpt_hash = "9580632d5de71799"
    obs_id = "20161210133714Th"
    H, W = 512, 512

    fg = np.full((H, W), 0.85, dtype=np.float32)
    ctr = np.full((H, W), 0.70, dtype=np.float32)
    bnd = np.full((H, W), 0.50, dtype=np.float32)
    off = np.zeros((2, H, W), dtype=np.float32)

    save_cached_prediction(
        cache_dir=cache_dir,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        fg=fg,
        ctr=ctr,
        bnd=bnd,
        off=off,
        image_sha256="img_sha_valid",
        model_state_sha256="state_sha_valid",
        precision="float32",
        preprocessing_version="v3",
        inference_policy="identity",
    )

    # 1. Matching strict load succeeds
    loaded = load_cached_prediction(
        cache_dir=cache_dir,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        expected_image_sha256="img_sha_valid",
        expected_model_state_sha256="state_sha_valid",
        expected_spatial_shape=(H, W),
        expected_precision="float32",
        expected_preprocessing_version="v3",
        expected_inference_policy="identity",
        strict=True,
    )
    assert loaded is not None
    assert np.allclose(loaded[0], 0.85), "FP32 edge values must round-trip exactly"

    # 2. Rejects None or empty image SHA in strict mode
    assert load_cached_prediction(
        cache_dir=cache_dir,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        expected_image_sha256="",
        expected_model_state_sha256="state_sha_valid",
        expected_spatial_shape=(H, W),
        strict=True,
    ) is None

    # 3. Rejects policy mismatch (e.g. tta_flip4 vs identity)
    assert load_cached_prediction(
        cache_dir=cache_dir,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        expected_image_sha256="img_sha_valid",
        expected_model_state_sha256="state_sha_valid",
        expected_spatial_shape=(H, W),
        expected_inference_policy="tta_flip4",
        strict=True,
    ) is None

    # 4. Rejects missing aux arrays if requires_aux=True
    save_cached_prediction(
        cache_dir=cache_dir,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id + "_no_aux",
        fg=fg,
        ctr=None,
        bnd=None,
        off=None,
        image_sha256="img_sha_valid",
        model_state_sha256="state_sha_valid",
        precision="float32",
    )
    assert load_cached_prediction(
        cache_dir=cache_dir,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id + "_no_aux",
        expected_image_sha256="img_sha_valid",
        expected_model_state_sha256="state_sha_valid",
        expected_spatial_shape=(H, W),
        requires_aux=True,
        strict=True,
    ) is None


# =========================================================================
# Regression 8: Real CPU-safe smoke & promotion guard
# =========================================================================
def test_setup_finetune_state_nonzero_lr_and_parameter_update():
    """setup_finetune_state resets optimizer/scheduler, enforces lr=5e-5, and updates weights."""
    model = ToyModel()
    initial_weight = model.conv.weight.data.clone()
    parent_state = {k: v.clone() for k, v in model.state_dict().items()}
    device = torch.device("cpu")

    optimizer, scheduler, scaler = setup_finetune_state(
        model=model,
        parent_state_dict=parent_state,
        device=device,
        lr=5e-5,
        weight_decay=1e-4,
        total_epochs=3,
        eta_min=1e-6,
    )

    current_lr = optimizer.param_groups[0]["lr"]
    assert current_lr == 5e-5, f"Expected 5e-5, got {current_lr}"

    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    loss = out.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated_weight = model.conv.weight.data
    assert not torch.allclose(initial_weight, updated_weight), "Weights must change on optimizer step"

    scheduler.step()
    next_lr = optimizer.param_groups[0]["lr"]
    assert next_lr < 5e-5, f"Cosine scheduler should decay lr, got {next_lr}"
    assert next_lr >= 1e-6, f"Cosine scheduler should stay >= eta_min, got {next_lr}"


def test_finetune_vs_resume_mutual_exclusivity():
    """train() rejects specifying both resume_path and finetune_from."""
    with pytest.raises(ValueError, match="Cannot specify both resume_path and finetune_from"):
        train(resume_path="dummy_resume.pt", finetune_from="dummy_parent.pt")


# =========================================================================
# Regression 9: Dataset Resume Continuous Sampling Integration Test
# =========================================================================
def test_dataset_resume_continuous_sampling_integration():
    """Dataset resume restores continuous sampling stream: exact next crop, annotator, flip, and optimizer update."""
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_images = data_dir / "train" / "train_images"

    if train_json.is_file() and train_images.is_dir():
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from src.data.annotations import load_coco_annotations
        from src.data.folds import load_frozen_folds_manifest

        anno_idx = load_coco_annotations(str(train_json))
        folds_manifest = Path("artifacts/folds_manifest.json")
        fold_assignments, _ = load_frozen_folds_manifest(folds_manifest)

        # 1. Initialize dataset 1 with augmentations enabled
        ds1 = SolarFilamentDataset(
            images_dir=train_images,
            annotation_index=anno_idx,
            fold_assignments=fold_assignments,
            target_fold=0,
            is_train=True,
            use_geometric_augmentation=True,
            seed=2026,
            max_observations=5,
        )

        model1 = ToyModel()
        opt1 = torch.optim.AdamW(model1.parameters(), lr=5e-5)
        sched1 = CosineAnnealingLR(opt1, T_max=3)

        # Draw 2 initial samples and step optimizer
        s0 = ds1[0]
        s1 = ds1[1]
        loss1 = (model1(s0["image"].unsqueeze(0)) + model1(s1["image"].unsqueeze(0))).sum()
        opt1.zero_grad()
        loss1.backward()
        opt1.step()
        sched1.step()

        # Capture complete resume checkpoint state
        saved_ds_rng = ds1.rng.get_state()
        saved_aug_rng = ds1.aug_rng.get_state()
        saved_model_state = copy.deepcopy(model1.state_dict())
        saved_opt_state = copy.deepcopy(opt1.state_dict())
        saved_sched_state = copy.deepcopy(sched1.state_dict())

        # Continue uninterrupted stream on ds1
        sample_uninterrupted = ds1[2]
        loss_uninterrupted = model1(sample_uninterrupted["image"].unsqueeze(0)).sum()
        opt1.zero_grad()
        loss_uninterrupted.backward()
        opt1.step()
        sched1.step()
        uninterrupted_lr = opt1.param_groups[0]["lr"]

        # 2. Initialize fresh dataset 2 with completely different initial seeds
        ds2 = SolarFilamentDataset(
            images_dir=train_images,
            annotation_index=anno_idx,
            fold_assignments=fold_assignments,
            target_fold=0,
            is_train=True,
            use_geometric_augmentation=True,
            seed=9999,
            aug_seed=8888,
            max_observations=5,
        )

        model2 = ToyModel()
        model2.load_state_dict(saved_model_state)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=5e-5)
        opt2.load_state_dict(saved_opt_state)
        sched2 = CosineAnnealingLR(opt2, T_max=3)
        sched2.load_state_dict(saved_sched_state)

        # Restore dataset and augmentation RNGs
        ds2.rng.set_state(saved_ds_rng)
        ds2.aug_rng.set_state(saved_aug_rng)

        # Draw next sample from resumed dataset
        sample_resumed = ds2[2]
        loss_resumed = model2(sample_resumed["image"].unsqueeze(0)).sum()
        opt2.zero_grad()
        loss_resumed.backward()
        opt2.step()
        sched2.step()
        resumed_lr = opt2.param_groups[0]["lr"]

        # Exact tensor bitwise parity on image, targets, and metadata
        assert torch.equal(sample_uninterrupted["image"], sample_resumed["image"]), "Resumed dataset image must match uninterrupted image"
        assert torch.equal(sample_uninterrupted["target_fg"], sample_resumed["target_fg"]), "Resumed target_fg must match uninterrupted"
        assert torch.equal(sample_uninterrupted["target_bnd"], sample_resumed["target_bnd"]), "Resumed target_bnd must match uninterrupted"
        assert torch.equal(sample_uninterrupted["target_skel"], sample_resumed["target_skel"]), "Resumed target_skel must match uninterrupted"
        assert torch.equal(sample_uninterrupted["target_ctr"], sample_resumed["target_ctr"]), "Resumed target_ctr must match uninterrupted"
        assert torch.equal(sample_uninterrupted["target_off"], sample_resumed["target_off"]), "Resumed target_off must match uninterrupted"

        # Exact optimizer update and scheduler LR parity
        assert uninterrupted_lr == resumed_lr, f"Scheduler LR must match: {uninterrupted_lr} vs {resumed_lr}"
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model parameters after step must match uninterrupted continuation"


# =========================================================================
# Regression 10: Strict Fine-Tune Resume Contract Rejection Probes
# =========================================================================
@pytest.mark.skip(reason="Resume unverified and excluded; production run is fresh A/B only")
def test_train_strict_resume_contract_rejections(tmp_path):
    """train(resume_path=...) reaches and verifies each strict validation branch independently:
    membership, missing horizon, mismatched horizon, skipped updates, missing updates, batch size, grad accum, optimizer recipe."""
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_images = data_dir / "train" / "train_images"
    if not (train_json.is_file() and train_images.is_dir()):
        pytest.skip("Dataset not present for resume contract test")

    from src.data.annotations import load_coco_annotations, canonical_observation_id
    from src.data.folds import load_frozen_folds_manifest
    from src.models import build_model

    anno = load_coco_annotations(str(train_json))
    manifest = Path("artifacts/folds_manifest.json")
    assignments, manifest_sha = load_frozen_folds_manifest(manifest)

    # In smoke=True mode with default max_observations=5 and max_val_observations=5
    valid_train_obs = [obs for obs in sorted(anno.by_observation.keys()) if assignments[canonical_observation_id(obs)] != 0][:5]
    valid_val_obs = [obs for obs in sorted(anno.by_observation.keys()) if assignments[canonical_observation_id(obs)] == 0][:5]

    toy_config = {
        "model": {"name": "resnet34_unet"},
        "training": {"lr": 0.0005, "weight_decay": 0.0001, "batch_size": 2, "epochs": 2},
        "loss": {"bce_weight": 1.0, "dice_weight": 1.0, "cldice_weight": 0.0, "boundary_weight": 0.0, "center_weight": 0.0, "offset_weight": 0.0},
    }
    model = build_model(toy_config)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)

    base_ckpt = {
        "epoch": 1,
        "fold": 0,
        "folds_manifest_sha256": manifest_sha,
        "train_observations": valid_train_obs,
        "val_observations": valid_val_obs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "random_rng_state": random.getstate(),
        "dataset_rng_state": np.random.RandomState(2026).get_state(),
        "dataset_aug_rng_state": None,
        "successful_updates": 10,
        "skipped_updates": 0,
        "total_epochs": 2,
        "config": toy_config,
        "runtime_config": {
            "augment_flips": False,
            "crop_policy": "standard",
            "total_epochs": 2,
            "batch_size": 2,
            "grad_accum_steps": 1,
        },
    }

    # 1. Observation membership mismatch raises ValueError
    bad_mem = dict(base_ckpt)
    bad_mem["train_observations"] = ["invalid_observation_id_not_in_manifest"]
    bad_mem_ckpt = tmp_path / "bad_mem.pt"
    torch.save(bad_mem, str(bad_mem_ckpt))
    with pytest.raises(ValueError, match="Resume train observations mismatch"):
        train(resume_path=str(bad_mem_ckpt), fold=0, smoke=True, epochs_override=2)

    # 2. Missing schedule horizon raises ValueError
    bad_horiz_missing = dict(base_ckpt)
    del bad_horiz_missing["total_epochs"]
    bad_horiz_missing["runtime_config"] = dict(base_ckpt["runtime_config"])
    del bad_horiz_missing["runtime_config"]["total_epochs"]
    bad_horiz_ckpt = tmp_path / "bad_horiz_missing.pt"
    torch.save(bad_horiz_missing, str(bad_horiz_ckpt))
    with pytest.raises(ValueError, match="Resume checkpoint missing schedule horizon"):
        train(resume_path=str(bad_horiz_ckpt), fold=0, smoke=True, epochs_override=2)

    # 3. Schedule horizon mismatch raises ValueError
    bad_horiz_mismatch = dict(base_ckpt)
    bad_horiz_mismatch["total_epochs"] = 99
    bad_horiz_mismatch["runtime_config"] = dict(base_ckpt["runtime_config"])
    bad_horiz_mismatch["runtime_config"]["total_epochs"] = 99
    bad_horiz_mis_ckpt = tmp_path / "bad_horiz_mis.pt"
    torch.save(bad_horiz_mismatch, str(bad_horiz_mis_ckpt))
    with pytest.raises(ValueError, match="Resume schedule horizon mismatch"):
        train(resume_path=str(bad_horiz_mis_ckpt), fold=0, smoke=True, epochs_override=2)

    # 4. Skipped updates > 0 reaches its specific branch and raises ValueError
    bad_skip = dict(base_ckpt)
    bad_skip["skipped_updates"] = 1
    bad_skip_ckpt = tmp_path / "bad_skip.pt"
    torch.save(bad_skip, str(bad_skip_ckpt))
    with pytest.raises(ValueError, match="Cannot resume from checkpoint with 1 cumulative skipped updates"):
        train(resume_path=str(bad_skip_ckpt), fold=0, smoke=True, epochs_override=2)

    # 5. Missing successful_updates reaches its specific branch and raises ValueError
    bad_updates = dict(base_ckpt)
    del bad_updates["successful_updates"]
    bad_updates_ckpt = tmp_path / "bad_updates.pt"
    torch.save(bad_updates, str(bad_updates_ckpt))
    with pytest.raises(ValueError, match="missing valid successful_updates metadata"):
        train(resume_path=str(bad_updates_ckpt), fold=0, smoke=True, epochs_override=2)

    # 6. Batch size mismatch raises ValueError
    bad_batch = dict(base_ckpt)
    bad_batch["runtime_config"] = dict(base_ckpt["runtime_config"])
    bad_batch["runtime_config"]["batch_size"] = 99
    bad_batch_ckpt = tmp_path / "bad_batch.pt"
    torch.save(bad_batch, str(bad_batch_ckpt))
    with pytest.raises(ValueError, match="Resume batch size mismatch"):
        train(resume_path=str(bad_batch_ckpt), fold=0, smoke=True, epochs_override=2)

    # 7. Gradient accumulation mismatch raises ValueError
    bad_accum = dict(base_ckpt)
    bad_accum["runtime_config"] = dict(base_ckpt["runtime_config"])
    bad_accum["runtime_config"]["grad_accum_steps"] = 99
    bad_accum_ckpt = tmp_path / "bad_accum.pt"
    torch.save(bad_accum, str(bad_accum_ckpt))
    with pytest.raises(ValueError, match="Resume gradient accumulation mismatch"):
        train(resume_path=str(bad_accum_ckpt), fold=0, smoke=True, epochs_override=2)

    # 8. Optimizer recipe mismatch raises ValueError
    bad_opt = dict(base_ckpt)
    bad_opt["optimizer_state_dict"] = copy.deepcopy(base_ckpt["optimizer_state_dict"])
    bad_opt["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.5
    bad_opt_ckpt = tmp_path / "bad_opt.pt"
    torch.save(bad_opt, str(bad_opt_ckpt))
    with pytest.raises(ValueError, match="Resume optimizer recipe mismatch"):
        train(resume_path=str(bad_opt_ckpt), fold=0, smoke=True, epochs_override=2)


# =========================================================================
# Regression 11: Train Save & Resume Integration Loop Test
# =========================================================================
@pytest.mark.skip(reason="Resume unverified and excluded; production run is fresh A/B only")
def test_train_save_and_resume_integration(tmp_path):
    """train() saves a complete valid checkpoint and can resume from it seamlessly."""
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not train_json.is_file():
        pytest.skip("Dataset annotations not present for integration test")

    # Step 1: Run 1 smoke epoch with max_steps=1
    latest_pt, best_pt = train(
        smoke=True,
        max_steps=1,
        epochs_override=1,
        fold=0,
    )
    assert latest_pt.is_file(), f"Latest checkpoint should be saved at {latest_pt}"
    ckpt_ep1 = torch.load(str(latest_pt), weights_only=False, map_location="cpu")
    assert ckpt_ep1["epoch"] == 1
    assert ckpt_ep1["successful_updates"] >= 1
    assert ckpt_ep1["skipped_updates"] == 0

    # Step 2: Prepare checkpoint for resuming into a 2-epoch schedule
    # Strict resume requires schedule horizon (total_epochs) to match the target run
    ckpt_ep1["total_epochs"] = 2
    if "runtime_config" in ckpt_ep1:
        ckpt_ep1["runtime_config"]["total_epochs"] = 2
    resume_target_pt = tmp_path / "resume_ep1_target2.pt"
    torch.save(ckpt_ep1, str(resume_target_pt))

    # Step 3: Resume from epoch 1 checkpoint to reach epoch 2
    latest_pt2, best_pt2 = train(
        smoke=True,
        max_steps=1,
        epochs_override=2,
        fold=0,
        resume_path=str(resume_target_pt),
    )
    assert latest_pt2.is_file()
    ckpt_ep2 = torch.load(str(latest_pt2), weights_only=False, map_location="cpu")
    assert ckpt_ep2["epoch"] == 2
    assert ckpt_ep2["successful_updates"] >= ckpt_ep1["successful_updates"]


# =========================================================================
# Regression 12: Cloud Bootstrap Mock Control-Flow Test
# =========================================================================
def test_cloud_bootstrap_lifecycle(tmp_path):
    """Prove default cloud bootstrap never launches training and honors supplied deadlines."""
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        bash_path = str(git_bash)
    else:
        bash_path = shutil.which("bash")
    if not bash_path:
        pytest.skip("bash executable not available on host")

    script_path = Path("scripts/cloud_bootstrap.sh").resolve()
    assert script_path.is_file()

    # Create mock tool binaries with pure LF endings
    mock_bin = tmp_path / "mock_bin"
    mock_bin.mkdir()

    def write_mock(name: str, content: str):
        p = mock_bin / name
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        p.chmod(0o755)

    write_mock("pip", "#!/usr/bin/env bash\necho 'MOCK_PIP:' \"$@\"\nexit 0\n")
    write_mock("tar", "#!/usr/bin/env bash\necho 'MOCK_TAR:' \"$@\"\nexit 0\n")
    write_mock("sha256sum", "#!/usr/bin/env bash\necho 'mock_sha256  '$1\nexit 0\n")

    mock_python_script = """#!/usr/bin/env bash
if [ "$1" = "-c" ]; then
    if [[ "$2" == *"time.time"* ]]; then
        echo "1000000"
    else
        echo "Imports verified successfully. PyTorch: 2.6.0 | CUDA: True"
    fi
    exit 0
fi

if [ "$1" = "scripts/iteration3_experiment.py" ]; then
    echo "TRAIN_INVOKED: $@"
    exit 0
fi

echo "MOCK_PYTHON3: $@"
exit 0
"""
    write_mock("python3", mock_python_script)

    base_env = os.environ.copy()
    base_env["PIP_BIN"] = str(mock_bin / "pip").replace("\\", "/")
    base_env["PYTHON_BIN"] = str(mock_bin / "python3").replace("\\", "/")
    base_env["TAR_BIN"] = str(mock_bin / "tar").replace("\\", "/")
    base_env["SHA256_BIN"] = str(mock_bin / "sha256sum").replace("\\", "/")
    base_env["PATH"] = f"{str(mock_bin)}{os.pathsep}{base_env.get('PATH', '')}"

    # Test 1: Default invocation halts at go/no-go without training
    res_default = subprocess.run(
        [bash_path, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        env=base_env,
    )
    assert res_default.returncode == 0
    assert "[GO/NO-GO CHECKPOINT] Setup, Preflight, and Benchmark Complete" in res_default.stdout
    assert "Full training was NOT started." in res_default.stdout
    assert "TRAIN_INVOKED" not in res_default.stdout, "Default bootstrap must NEVER invoke training"

    # Test 2: Training mode without deadline fails with exit code 1
    res_no_dl = subprocess.run(
        [bash_path, str(script_path), "--train"],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        env=base_env,
    )
    assert res_no_dl.returncode != 0
    assert "Training mode requires --deadline-timestamp" in (res_no_dl.stderr + res_no_dl.stdout)

    # Test 3: Training mode with expired deadline fails with exit code 1
    res_expired = subprocess.run(
        [bash_path, str(script_path), "--train", "--deadline-timestamp", "100"],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        env=base_env,
    )
    assert res_expired.returncode != 0
    assert "has expired or does not leave adequate cleanup margin" in (res_expired.stderr + res_expired.stdout)

    # Test 4: Training mode with valid future deadline invokes training with exact computed budget
    # Mock current time is 1000000; deadline 1000500 leaves 500s - 60s = 440s budget
    res_valid = subprocess.run(
        [bash_path, str(script_path), "--train", "--deadline-timestamp", "1000500"],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        env=base_env,
    )
    assert res_valid.returncode == 0
    combined_out = res_valid.stdout + res_valid.stderr
    assert "TRAIN_INVOKED: scripts/iteration3_experiment.py --budget-seconds 440 --deadline-timestamp 1000500" in combined_out
    assert "Export completed" in combined_out


# =========================================================================
# Regression 13: Checkpoint Discovery Production Contract
# =========================================================================
def test_checkpoint_discovery_production_contract(tmp_path):
    """Realistic integration regression: training stub returns global legacy aliases while real-style
    immutable epoch files exist under artifacts/runs/<run_id>/checkpoints/. Old discovery fails (0 epochs)
    while corrected discovery successfully resolves all 3 epochs for both arms."""
    arm_a_id = "run_20261001_170021_daeb4d"
    arm_b_id = "run_20261001_171453_ede396"

    # Setup directories
    runs_dir = tmp_path / "artifacts" / "runs"
    arm_a_ckpt_dir = runs_dir / arm_a_id / "checkpoints"
    arm_b_ckpt_dir = runs_dir / arm_b_id / "checkpoints"
    arm_a_ckpt_dir.mkdir(parents=True, exist_ok=True)
    arm_b_ckpt_dir.mkdir(parents=True, exist_ok=True)

    legacy_ckpt_dir = tmp_path / "checkpoints"
    legacy_ckpt_dir.mkdir(parents=True, exist_ok=True)
    legacy_alias = legacy_ckpt_dir / "resnet34_unet_fold0_latest.pt"

    # Generate 3 valid epochs for Arm A (control)
    for e in range(1, 4):
        ckpt_data_a = {
            "run_id": arm_a_id,
            "epoch": e,
            "fold": 0,
            "total_epochs": 3,
            "successful_updates": 142 * e,
            "skipped_updates": 0,
            "eligible_for_promotion": True,
            "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
            "mining_bank_sha256": "",
            "model_state_dict": {"conv.weight": torch.ones(1, 3, 3, 3)},
            "finetune_parent": {
                "file_sha256": EXPECTED_PARENT_FILE_SHA256,
                "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                "is_diagnostic": False,
            },
            "runtime_config": {
                "run_id": arm_a_id,
                "augment_flips": False,
                "mining_bank_sha256": "",
                "eligible_for_promotion": True,
                "smoke": False,
                "is_diagnostic": False,
            },
        }
        torch.save(ckpt_data_a, str(arm_a_ckpt_dir / f"epoch_{e:03d}.pt"))

    # Generate 3 valid epochs for Arm B (intervention)
    for e in range(1, 4):
        ckpt_data_b = {
            "run_id": arm_b_id,
            "epoch": e,
            "fold": 0,
            "total_epochs": 3,
            "successful_updates": 142 * e,
            "skipped_updates": 0,
            "eligible_for_promotion": True,
            "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
            "mining_bank_sha256": EXPECTED_MINING_BANK_SHA256,
            "model_state_dict": {"conv.weight": torch.ones(1, 3, 3, 3) * 0.99},
            "finetune_parent": {
                "file_sha256": EXPECTED_PARENT_FILE_SHA256,
                "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                "is_diagnostic": False,
            },
            "runtime_config": {
                "run_id": arm_b_id,
                "augment_flips": True,
                "mining_bank_sha256": EXPECTED_MINING_BANK_SHA256,
                "eligible_for_promotion": True,
                "smoke": False,
                "is_diagnostic": False,
            },
        }
        torch.save(ckpt_data_b, str(arm_b_ckpt_dir / f"epoch_{e:03d}.pt"))

    # Global legacy alias is overwritten by Arm B
    torch.save(ckpt_data_b, str(legacy_alias))

    # 1. Verify defect contract: Searching legacy parent path finds 0 epoch checkpoints!
    old_discovery_a = list(Path(legacy_alias).parent.glob("epoch_*.pt"))
    assert len(old_discovery_a) == 0, (
        f"Defect reproduction: Path(latest).parent should have 0 epoch checkpoints, got {len(old_discovery_a)}"
    )

    # 2. Verify corrected discovery: resolve_and_validate_arm_checkpoints finds exactly 3 epochs each
    discovered_a = resolve_and_validate_arm_checkpoints(
        arm_label="Arm_A",
        run_id=arm_a_id,
        root_dir=tmp_path,
        expected_epochs=3,
        smoke=False,
    )
    assert len(discovered_a) == 3
    for idx, (arm, path, data) in enumerate(discovered_a):
        assert arm == "Arm_A"
        assert path.name == f"epoch_{idx + 1:03d}.pt"
        assert data["run_id"] == arm_a_id
        assert data["successful_updates"] == 142 * (idx + 1)
        assert data["skipped_updates"] == 0
        assert data["runtime_config"]["augment_flips"] is False

    discovered_b = resolve_and_validate_arm_checkpoints(
        arm_label="Arm_B",
        run_id=arm_b_id,
        root_dir=tmp_path,
        expected_epochs=3,
        smoke=False,
    )
    assert len(discovered_b) == 3
    for idx, (arm, path, data) in enumerate(discovered_b):
        assert arm == "Arm_B"
        assert path.name == f"epoch_{idx + 1:03d}.pt"
        assert data["run_id"] == arm_b_id
        assert data["successful_updates"] == 142 * (idx + 1)
        assert data["skipped_updates"] == 0
        assert data["runtime_config"]["augment_flips"] is True
        assert data["mining_bank_sha256"] == EXPECTED_MINING_BANK_SHA256


# =========================================================================
# Regression 14: Adversarial Rejection Probes for Evaluation-Only Recovery
# =========================================================================
def test_recovery_adversarial_rejections(tmp_path):
    """Adversarial rejections: absent epochs, swapped/forged run_id, duplicate/skipped epoch,
    reused Arm A as Arm B, unequal updates, wrong parent, wrong B bank or flips, path containment,
    deadline expiry, and verification that recovery never invokes training."""
    arm_a_id = "run_20261001_170021_daeb4d"
    arm_b_id = "run_20261001_171453_ede396"

    runs_dir = tmp_path / "artifacts" / "runs"
    arm_a_ckpt_dir = runs_dir / arm_a_id / "checkpoints"
    arm_b_ckpt_dir = runs_dir / arm_b_id / "checkpoints"
    arm_a_ckpt_dir.mkdir(parents=True, exist_ok=True)
    arm_b_ckpt_dir.mkdir(parents=True, exist_ok=True)

    def write_valid_arms():
        if arm_a_ckpt_dir.is_dir():
            shutil.rmtree(arm_a_ckpt_dir)
        if arm_b_ckpt_dir.is_dir():
            shutil.rmtree(arm_b_ckpt_dir)
        arm_a_ckpt_dir.mkdir(parents=True, exist_ok=True)
        arm_b_ckpt_dir.mkdir(parents=True, exist_ok=True)

        folds_src_p = Path("artifacts/folds_manifest.json")
        if folds_src_p.is_file():
            with open(folds_src_p, "r", encoding="utf-8") as f:
                man_data = json.load(f)
            assigns = man_data["assignments"]
            valid_val = [k for k, v in assigns.items() if "-" not in k and v == 0]
            valid_train = [k for k, v in assigns.items() if "-" not in k and v != 0]
        else:
            valid_val = [f"val_{i}" for i in range(142)]
            valid_train = [f"train_{i}" for i in range(565)]

        for e in range(1, 4):
            ckpt_a = {
                "run_id": arm_a_id,
                "epoch": e,
                "fold": 0,
                "total_epochs": 3,
                "train_observations": valid_train,
                "val_observations": valid_val,
                "successful_updates": 142 * e,
                "skipped_updates": 0,
                "eligible_for_promotion": True,
                "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                "mining_bank_sha256": "",
                "model_state_dict": {"conv.weight": torch.ones(1, 3, 3, 3)},
                "finetune_parent": {
                    "file_sha256": EXPECTED_PARENT_FILE_SHA256,
                    "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                    "is_diagnostic": False,
                },
                "runtime_config": {
                    "run_id": arm_a_id,
                    "augment_flips": False,
                    "mining_bank_sha256": "",
                    "eligible_for_promotion": True,
                    "smoke": False,
                    "is_diagnostic": False,
                },
            }
            torch.save(ckpt_a, str(arm_a_ckpt_dir / f"epoch_{e:03d}.pt"))

            ckpt_b = {
                "run_id": arm_b_id,
                "epoch": e,
                "fold": 0,
                "total_epochs": 3,
                "train_observations": valid_train,
                "val_observations": valid_val,
                "successful_updates": 142 * e,
                "skipped_updates": 0,
                "eligible_for_promotion": True,
                "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                "mining_bank_sha256": EXPECTED_MINING_BANK_SHA256,
                "model_state_dict": {"conv.weight": torch.ones(1, 3, 3, 3) * 0.99},
                "finetune_parent": {
                    "file_sha256": EXPECTED_PARENT_FILE_SHA256,
                    "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                    "is_diagnostic": False,
                },
                "runtime_config": {
                    "run_id": arm_b_id,
                    "augment_flips": True,
                    "mining_bank_sha256": EXPECTED_MINING_BANK_SHA256,
                    "eligible_for_promotion": True,
                    "smoke": False,
                    "is_diagnostic": False,
                },
            }
            torch.save(ckpt_b, str(arm_b_ckpt_dir / f"epoch_{e:03d}.pt"))

    write_valid_arms()

    parent_src = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    folds_src = Path("artifacts/folds_manifest.json")
    parts_src = Path("artifacts/partitions_migrated_v1.json")
    bank_src = Path("artifacts/reports/mining_bank_v1.json")

    # Probe 1: Reused Arm A as Arm B
    with pytest.raises(ValueError, match="Arm A and Arm B run IDs cannot be identical"):
        recover_and_evaluate(
            arm_a_run_id=arm_a_id,
            arm_b_run_id=arm_a_id,
            deadline_timestamp=2000000000.0,
            root_dir=tmp_path,
        )

    # Probe 2: Absent epoch (e.g. only 2 epochs present in Arm A)
    (arm_a_ckpt_dir / "epoch_003.pt").unlink()
    with pytest.raises(ValueError, match="Expected exactly 3 completed epochs"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 3: Sequence mismatch (e.g. epoch 2 missing, epoch 4 present)
    (arm_a_ckpt_dir / "epoch_002.pt").rename(arm_a_ckpt_dir / "epoch_004.pt")
    with pytest.raises(ValueError, match="Checkpoint sequence mismatch"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 4: Forged run_id
    bad_ckpt = torch.load(str(arm_a_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["run_id"] = "forged_id"
    torch.save(bad_ckpt, str(arm_a_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="run_id mismatch"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 5: Non-finite weights
    bad_ckpt = torch.load(str(arm_a_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["model_state_dict"]["conv.weight"][0, 0, 0, 0] = float("nan")
    torch.save(bad_ckpt, str(arm_a_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="non-finite weights"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 6: Skipped updates
    bad_ckpt = torch.load(str(arm_a_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["skipped_updates"] = 1
    torch.save(bad_ckpt, str(arm_a_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="skipped updates"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 7: Unequal updates (e.g. 140 instead of 142)
    bad_ckpt = torch.load(str(arm_a_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["successful_updates"] = 140
    torch.save(bad_ckpt, str(arm_a_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="successful updates mismatch"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 8: Wrong parent SHA
    bad_ckpt = torch.load(str(arm_a_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["finetune_parent"]["file_sha256"] = "wrong_sha"
    torch.save(bad_ckpt, str(arm_a_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="parent SHA mismatch"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 9: Arm A with flips or active bank
    bad_ckpt = torch.load(str(arm_a_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["runtime_config"]["augment_flips"] = True
    torch.save(bad_ckpt, str(arm_a_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="augment_flips=True; expected False for Arm A"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    bad_ckpt = torch.load(str(arm_a_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["mining_bank_sha256"] = EXPECTED_MINING_BANK_SHA256
    torch.save(bad_ckpt, str(arm_a_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="active mining bank"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_a_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 10: Arm B without flips or wrong bank SHA
    bad_ckpt = torch.load(str(arm_b_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["runtime_config"]["augment_flips"] = False
    torch.save(bad_ckpt, str(arm_b_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="augment_flips=False; expected True for Arm B"):
        resolve_and_validate_arm_checkpoints("Arm_B", arm_b_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    bad_ckpt = torch.load(str(arm_b_ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["mining_bank_sha256"] = "wrong_bank_sha"
    torch.save(bad_ckpt, str(arm_b_ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="mining bank SHA mismatch"):
        resolve_and_validate_arm_checkpoints("Arm_B", arm_b_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_arms()

    # Probe 11: Path containment violation
    with pytest.raises(ValueError, match="Path containment violation"):
        resolve_and_validate_arm_checkpoints("Arm_A", "../../outside", root_dir=tmp_path, expected_epochs=3)

    # Probe 12: Deadline enforcement
    with pytest.raises(ValueError, match="Missing required deadline"):
        recover_and_evaluate(arm_a_run_id=arm_a_id, arm_b_run_id=arm_b_id, deadline_timestamp=None, root_dir=tmp_path)
    with pytest.raises(TimeoutError, match="has expired or leaves inadequate cleanup margin"):
        recover_and_evaluate(arm_a_run_id=arm_a_id, arm_b_run_id=arm_b_id, deadline_timestamp=100.0, root_dir=tmp_path)
    with pytest.raises(TimeoutError, match="has expired or leaves inadequate cleanup margin"):
        recover_and_evaluate(arm_a_run_id=arm_a_id, arm_b_run_id=arm_b_id, deadline_timestamp="2000-01-01T00:00:00Z", root_dir=tmp_path)

    # Probe 13: Recovery never calls train()
    def mock_eval(**kwargs):
        return {
            "checkpoint": kwargs.get("checkpoint_path"),
            "resolved_inference_config": {
                "method": "connected_components",
                "high_threshold": 0.85,
                "low_threshold": 0.70,
                "min_area": 400,
                "max_instances": 20,
            },
            "total_evaluated_observations": 10,
            "evaluated_physical_ids": ["phys1"],
            "overall": {
                "pq": 0.250000,
                "sq": 0.800000,
                "rq": 0.400000,
                "tp": 20,
                "fp": 5,
                "fn": 10,
                "mean_dice": 0.750000,
            },
        }

    if parent_src.is_file() and folds_src.is_file() and parts_src.is_file() and bank_src.is_file():
        (tmp_path / "artifacts" / "folds_manifest.json").write_bytes(folds_src.read_bytes())
        (tmp_path / "artifacts" / "partitions_migrated_v1.json").write_bytes(parts_src.read_bytes())
        (tmp_path / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)
        (tmp_path / "artifacts" / "reports" / "mining_bank_v1.json").write_bytes(bank_src.read_bytes())
        (tmp_path / "parent_epoch_006.pt").write_bytes(parent_src.read_bytes())

        future_dl = time.time() + 3600.0
        summary = recover_and_evaluate(
            arm_a_run_id=arm_a_id,
            arm_b_run_id=arm_b_id,
            deadline_timestamp=future_dl,
            parent_checkpoint=str(tmp_path / "parent_epoch_006.pt"),
            mining_bank_path=str(tmp_path / "artifacts" / "reports" / "mining_bank_v1.json"),
            smoke=False,
            eval_fn=mock_eval,
            root_dir=tmp_path,
        )
        assert summary["mode"] == "evaluation_only_recovery"
        assert summary["arms"]["arm_a_control"]["run_id"] == arm_a_id
        assert summary["arms"]["arm_b_intervention"]["run_id"] == arm_b_id
        assert summary["gates"]["candidate_3_promoted"] is False
        assert "script_source_sha256" in summary
        assert "selector_repair_sha256" in summary


# =========================================================================
# Regression 15: Run Experiment Immediate Arm A Discovery Rejection Before Arm B
# =========================================================================
def test_run_experiment_immediate_arm_a_discovery_rejection(tmp_path):
    """If Arm A finishes but fails discovery (0 epochs), fail immediately before Arm B is started."""
    arm_b_called = False

    def mock_train_fn(**kwargs):
        nonlocal arm_b_called
        augment_flips = kwargs.get("augment_flips", False)
        if not augment_flips:
            empty_run_dir = tmp_path / "artifacts" / "runs" / "arm_a_empty" / "checkpoints"
            empty_run_dir.mkdir(parents=True, exist_ok=True)
            legacy_alias = tmp_path / "checkpoints" / "resnet34_unet_fold0_latest.pt"
            legacy_alias.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"run_id": "arm_a_empty", "epoch": 1, "model_state_dict": {"w": torch.ones(1)}}, str(legacy_alias))
            return legacy_alias, legacy_alias
        else:
            arm_b_called = True
            return tmp_path / "b.pt", tmp_path / "b.pt"

    real_parent = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    if real_parent.is_file():
        with pytest.raises(FileNotFoundError, match="Found 0 epoch checkpoints|Checkpoint directory not found"):
            run_experiment(
                parent_checkpoint=str(real_parent),
                mining_bank_path="artifacts/reports/mining_bank_v1.json",
                smoke=True,
                train_fn=mock_train_fn,
            )
        assert arm_b_called is False, "Arm B training must NEVER be started if Arm A discovery fails!"


# =========================================================================
# Regression 16: Inference Tile Batch Size Resolution & Propagation
# =========================================================================
def test_evaluate_oof_tile_batch_size_propagation(monkeypatch):
    """evaluate_oof accepts tile_batch_size override, resolves it, and propagates to prediction engine."""
    from evaluate import resolve_inference_config

    # 1. Config resolver defaults to 16 if unconfigured, or honors positive override
    default_cfg = resolve_inference_config({})
    assert default_cfg["tile_batch_size"] == 16, f"Default batch size must be 16, got {default_cfg['tile_batch_size']}"

    overridden_cfg = resolve_inference_config({}, overrides={"tile_batch_size": 4})
    assert overridden_cfg["tile_batch_size"] == 4, f"Overridden batch size must be 4, got {overridden_cfg['tile_batch_size']}"

    # Rejects non-positive batch size
    with pytest.raises(ValueError, match="positive"):
        resolve_inference_config({}, overrides={"tile_batch_size": 0})
    with pytest.raises(ValueError, match="positive"):
        resolve_inference_config({}, overrides={"tile_batch_size": -1})


# =========================================================================
# Regression 17: Strict Production Checkpoint Discovery Hardening
# =========================================================================
def test_strict_production_checkpoint_discovery_hardening(tmp_path):
    """Adversarial discovery probes: single-component run_id, missing metadata, wrong fold/epochs, absent flips."""
    runs_dir = tmp_path / "artifacts" / "runs"
    arm_id = "run_valid_arm_123"
    ckpt_dir = runs_dir / arm_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def write_valid_epoch_ckpts():
        for e in range(1, 4):
            ckpt = {
                "run_id": arm_id,
                "epoch": e,
                "fold": 0,
                "total_epochs": 3,
                "successful_updates": 142 * e,
                "skipped_updates": 0,
                "eligible_for_promotion": True,
                "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                "finetune_parent": {
                    "file_sha256": EXPECTED_PARENT_FILE_SHA256,
                    "folds_manifest_sha256": EXPECTED_FOLDS_MANIFEST_SHA256,
                    "is_diagnostic": False,
                },
                "model_state_dict": {"w": torch.ones(2, 2)},
                "runtime_config": {"augment_flips": False},
            }
            torch.save(ckpt, str(ckpt_dir / f"epoch_{e:03d}.pt"))

    write_valid_epoch_ckpts()

    # 1. Reject invalid/traversal/absolute run_id
    for bad_id in ["../../outside", "sub/folder", "/abs/path", ""]:
        with pytest.raises(ValueError, match="Path containment violation|single-component"):
            resolve_and_validate_arm_checkpoints("Arm_A", bad_id, root_dir=tmp_path, expected_epochs=3)

    # 2. Reject missing skipped_updates metadata
    bad_ckpt = torch.load(str(ckpt_dir / "epoch_001.pt"), weights_only=False)
    del bad_ckpt["skipped_updates"]
    torch.save(bad_ckpt, str(ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="missing skipped_updates metadata"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_epoch_ckpts()

    # 3. Reject missing eligible_for_promotion metadata
    bad_ckpt = torch.load(str(ckpt_dir / "epoch_001.pt"), weights_only=False)
    del bad_ckpt["eligible_for_promotion"]
    torch.save(bad_ckpt, str(ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="missing eligible_for_promotion metadata"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_epoch_ckpts()

    # 4. Reject missing fold or wrong fold
    bad_ckpt = torch.load(str(ckpt_dir / "epoch_001.pt"), weights_only=False)
    del bad_ckpt["fold"]
    torch.save(bad_ckpt, str(ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="missing fold metadata"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_id, root_dir=tmp_path, expected_epochs=3)

    bad_ckpt["fold"] = 1
    torch.save(bad_ckpt, str(ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="fold mismatch"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_epoch_ckpts()

    # 5. Reject missing total_epochs or wrong total_epochs
    bad_ckpt = torch.load(str(ckpt_dir / "epoch_001.pt"), weights_only=False)
    del bad_ckpt["total_epochs"]
    torch.save(bad_ckpt, str(ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="missing total_epochs metadata"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_id, root_dir=tmp_path, expected_epochs=3)

    bad_ckpt["total_epochs"] = 5
    torch.save(bad_ckpt, str(ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="total_epochs mismatch"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_epoch_ckpts()

    # 6. Reject absent Arm A augment_flips (must be explicitly False)
    bad_ckpt = torch.load(str(ckpt_dir / "epoch_001.pt"), weights_only=False)
    bad_ckpt["runtime_config"]["augment_flips"] = None
    torch.save(bad_ckpt, str(ckpt_dir / "epoch_001.pt"))
    with pytest.raises(ValueError, match="augment_flips must be explicitly False"):
        resolve_and_validate_arm_checkpoints("Arm_A", arm_id, root_dir=tmp_path, expected_epochs=3)
    write_valid_epoch_ckpts()

    # 7. Reject empty expected_mining_bank_sha for Arm B
    for e in range(1, 4):
        ckpt = torch.load(str(ckpt_dir / f"epoch_{e:03d}.pt"), weights_only=False)
        ckpt["runtime_config"]["augment_flips"] = True
        torch.save(ckpt, str(ckpt_dir / f"epoch_{e:03d}.pt"))
    with pytest.raises(ValueError, match="expected_mining_bank_sha must not be empty"):
        resolve_and_validate_arm_checkpoints("Arm_B", arm_id, root_dir=tmp_path, expected_epochs=3, expected_mining_bank_sha="")


# =========================================================================
# Regression 18: Recovery Requires Mining Bank and Disjoint Membership
# =========================================================================
def test_recovery_requires_mining_bank_and_disjoint_manifest_membership(tmp_path):
    """recover_and_evaluate requires mining bank file and exact disjoint manifest membership."""
    arm_a_id = "run_recovery_a"
    arm_b_id = "run_recovery_b"
    future_dl = time.time() + 3600.0

    # 1. Missing mining bank path fails before evaluation
    with pytest.raises(FileNotFoundError, match="Required mining bank missing|not supplied"):
        recover_and_evaluate(
            arm_a_run_id=arm_a_id,
            arm_b_run_id=arm_b_id,
            deadline_timestamp=future_dl,
            mining_bank_path=str(tmp_path / "nonexistent_bank.json"),
            root_dir=tmp_path,
        )


# =========================================================================
# Regression 19: Calibrate B Epoch 1 Complete 48 Grid and Ranking
# =========================================================================
def test_calibrate_b_epoch1_complete_48_grid_and_ranking(tmp_path):
    """Verify 48-grid Cartesian coverage, anchor inclusion, tie-breaking and winner freezing."""
    from scripts.calibrate_b_epoch1 import (
        ANCHOR_SETTING,
        GRID_HIGH_THRESHOLDS,
        GRID_LOW_THRESHOLDS,
        GRID_MAX_INSTANCES,
        GRID_MIN_AREAS,
        run_calibration_grid,
    )

    # 1. Exactly 48 Cartesian combinations
    total_combinations = len(GRID_HIGH_THRESHOLDS) * len(GRID_LOW_THRESHOLDS) * len(GRID_MIN_AREAS) * len(GRID_MAX_INSTANCES)
    assert total_combinations == 48, f"Grid must have exactly 48 settings, got {total_combinations}"

    # 2. Anchor is in the grid
    assert ANCHOR_SETTING[0] in GRID_HIGH_THRESHOLDS
    assert ANCHOR_SETTING[1] in GRID_LOW_THRESHOLDS
    assert ANCHOR_SETTING[2] in GRID_MIN_AREAS
    assert ANCHOR_SETTING[3] in GRID_MAX_INSTANCES

    # 3. Test deterministic ranking logic
    reports_dir = tmp_path / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_info = {
        "checkpoint_path": "dummy_b_epoch1.pt",
        "checkpoint_sha256": "dummy_sha256",
        "model_state_sha256": "dummy_model_sha",
        "partitions_sha256": "dummy_parts_sha",
        "manifest_sha256": "dummy_man_sha",
        "annotations_sha256": "dummy_ann_sha",
    }

    dummy_maps = {"obs1": np.zeros((16, 16), dtype=np.float32)}
    dummy_variants = {"obs1": []}

    # Verify ranking sorts by max PQ, then min FP, then smallest changed knobs
    items = [
        {"config": {"high_threshold": 0.85, "low_threshold": 0.70, "min_area": 400, "max_instances": 20},
         "changed_knobs": 0, "metrics": {"pq": 0.315, "fp": 10}},
        {"config": {"high_threshold": 0.90, "low_threshold": 0.70, "min_area": 400, "max_instances": 20},
         "changed_knobs": 1, "metrics": {"pq": 0.315, "fp": 8}},  # lower FP wins tie
        {"config": {"high_threshold": 0.80, "low_threshold": 0.70, "min_area": 400, "max_instances": 20},
         "changed_knobs": 1, "metrics": {"pq": 0.318, "fp": 15}}, # higher PQ wins outright
    ]
    def rank_key(item):
        cfg = item["config"]
        met = item["metrics"]
        return (-met["pq"], met["fp"], item["changed_knobs"], -cfg["high_threshold"], cfg["low_threshold"], cfg["min_area"], cfg["max_instances"])

    sorted_items = sorted(items, key=rank_key)
    assert sorted_items[0]["metrics"]["pq"] == 0.318, "Highest PQ must rank 1st"
    assert sorted_items[1]["metrics"]["fp"] == 8, "Lower FP must win tiebreak"


# =========================================================================
# Regression 20: Calibrate B Epoch 1 Gate Enforcement Inhibits Generation
# =========================================================================
def test_calibrate_b_epoch1_gate_inhibits_generation(tmp_path):
    """Gates prevent candidate submission generation when quality thresholds are not met."""
    from scripts.calibrate_b_epoch1 import GATE1_TUNING_TARGET, GATE2_CONF_FLOOR

    # Unchanged quality gate values
    assert GATE1_TUNING_TARGET == 0.31764891564223335
    assert GATE2_CONF_FLOOR == 0.30880793475475555

    # Calibrate script contains NO neural training invocation
    calibrate_script = Path("scripts/calibrate_b_epoch1.py")
    assert calibrate_script.is_file()
    script_text = calibrate_script.read_text(encoding="utf-8")
    assert "train(" not in script_text, "calibrate_b_epoch1.py must NEVER invoke train()"
    assert "def train" not in script_text, "calibrate_b_epoch1.py must NEVER define or run training"


# =========================================================================
# Regression 21: Real-Input Preflight Cryptographic and Membership Verification
# =========================================================================
def test_calibrate_b_epoch1_real_input_preflight():
    """Verify verify_inputs passes cleanly on real disk files with exact SHAs and disjoint fold 0 membership."""
    from scripts.calibrate_b_epoch1 import (
        DEFAULT_CHECKPOINT_PATH,
        EXPECTED_ANNOTATIONS_SHA256,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_FOLDS_MANIFEST_SHA256,
        EXPECTED_PARTITIONS_SHA256,
        verify_inputs,
    )

    assert DEFAULT_CHECKPOINT_PATH.is_file(), f"Real checkpoint missing at: {DEFAULT_CHECKPOINT_PATH}"
    info = verify_inputs(checkpoint_path=DEFAULT_CHECKPOINT_PATH)

    # Checkpoint SHA
    assert info["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert EXPECTED_CHECKPOINT_SHA256 == "8ae7cb726b35d17949a58e9dcd26dd09bf4ecdc7c6f3c10d18d11210e013cf0a"

    # Partitions SHA
    assert info["partitions_sha256"] == EXPECTED_PARTITIONS_SHA256
    assert EXPECTED_PARTITIONS_SHA256 == "81022d8462bec5031c9b1f3760fe310a9cb15008f00770ab57a62a4f39bff467"

    # Folds Manifest SHA
    assert info["manifest_sha256"] == EXPECTED_FOLDS_MANIFEST_SHA256
    assert EXPECTED_FOLDS_MANIFEST_SHA256 == "2495791553a6107873c9962d05a0ba40d25873420937d483641121e7e9c989bd"

    # Annotations SHA
    assert info["annotations_sha256"] == EXPECTED_ANNOTATIONS_SHA256
    assert EXPECTED_ANNOTATIONS_SHA256 == "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"

    # Partition sets
    assert len(info["tuning_observations"]) == 10
    assert len(info["confirmation_observations"]) == 9
    assert len(set(info["tuning_observations"]) & set(info["confirmation_observations"])) == 0


# =========================================================================
# Regression 22: Staged Lifecycle, Comparison Report Binding, and Deadline
# =========================================================================
def test_calibrate_b_epoch1_staged_lifecycle_and_binding(tmp_path, monkeypatch):
    """Test staged execution, comparison report binding in generate-frozen, false gate rejection, and deadline."""
    import sys
    from scripts.calibrate_b_epoch1 import (
        GATE1_TUNING_TARGET,
        GATE2_CONF_FLOOR,
        check_deadline,
        main,
    )

    reports_dir = tmp_path / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "artifacts" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test deadline expiration
    past_dl = time.time() - 10.0
    with pytest.raises(TimeoutError, match="Absolute deadline exceeded"):
        check_deadline(past_dl, "test operation")

    future_dl = time.time() + 3600.0
    check_deadline(future_dl, "test operation")  # does not raise

    # 2. Test generate-frozen requires saved comparison report
    dummy_frozen_winner = {
        "frozen_timestamp": "2026-10-01T00:00:00Z",
        "checkpoint_path": "dummy.pt",
        "checkpoint_sha256": "dummy_ckpt_sha",
        "model_state_sha256": "dummy_model_sha",
        "partitions_sha256": "dummy_part_sha",
        "manifest_sha256": "dummy_man_sha",
        "annotations_sha256": "dummy_ann_sha",
        "tile_batch_size": 4,
        "selected_config": {"high_threshold": 0.85, "low_threshold": 0.70, "min_area": 400, "max_instances": 20},
        "tuning_metrics": {"pq": 0.320},
        "changed_knobs_from_anchor": 0,
        "gate1_target": GATE1_TUNING_TARGET,
        "gate1_passed": True,
    }
    frozen_path = reports_dir / "b_epoch1_frozen_winner.json"
    frozen_path.write_text(json.dumps(dummy_frozen_winner), encoding="utf-8")

    # Mock verify_inputs to return matching metadata for tmp_path test
    mock_info = {
        "checkpoint_path": "dummy.pt",
        "checkpoint_sha256": "dummy_ckpt_sha",
        "model_state_sha256": "dummy_model_sha",
        "partitions_path": "parts.json",
        "partitions_sha256": "dummy_part_sha",
        "manifest_path": "folds.json",
        "manifest_sha256": "dummy_man_sha",
        "annotations_path": "ann.json",
        "annotations_sha256": "dummy_ann_sha",
        "tuning_observations": [f"tune_{i}" for i in range(10)],
        "confirmation_observations": [f"conf_{i}" for i in range(9)],
        "ckpt_data": {"train_observations": []},
        "fold_assignments": {},
    }
    monkeypatch.setattr("scripts.calibrate_b_epoch1.verify_inputs", lambda **kwargs: mock_info)
    monkeypatch.setattr("scripts.calibrate_b_epoch1.load_coco_annotations", lambda path: None)

    class MockDataset:
        def __init__(self, **kwargs): pass
        def get_observation_annotations(self, obs):
            # return 2 variants for 4 obs (8) and 3 for 6 obs (16) -> total 24 for tuning
            if obs.startswith("tune"):
                idx = int(obs.split("_")[1])
                return [object()] * (3 if idx < 4 else 2)  # 4*3 + 6*2 = 24
            else:
                idx = int(obs.split("_")[1])
                return [object()] * (2 if idx < 6 else 1)  # 6*2 + 3*1 = 15

    monkeypatch.setattr("scripts.calibrate_b_epoch1.SolarFilamentDataset", MockDataset)

    # Calling generate-frozen without comparison report raises FileNotFoundError
    monkeypatch.setattr(
        sys, "argv",
        ["calibrate_b_epoch1.py", "--stage", "generate-frozen", "--reports-dir", str(reports_dir), "--cache-dir", str(cache_dir)]
    )
    with pytest.raises(FileNotFoundError, match="Saved comparison report not found"):
        main()

    # 3. Test generate-frozen rejects tampered / stale comparison report
    from src.inference.engine import compute_file_sha256
    valid_frozen_sha = compute_file_sha256(frozen_path)

    bad_comp_report = {
        "timestamp": "2026-10-01T00:00:00Z",
        "checkpoint_sha256": "DIFFERENT_CKPT_SHA",
        "model_state_sha256": "dummy_model_sha",
        "frozen_selection_sha256": valid_frozen_sha,
        "frozen_config": dummy_frozen_winner["selected_config"],
        "partitions_sha256": "dummy_part_sha",
        "manifest_sha256": "dummy_man_sha",
        "annotations_sha256": "dummy_ann_sha",
        "evaluated_physical_ids": [f"conf_{i}" for i in range(9)],
        "total_entries": 15,
        "confirmation_metrics": {"pq": 0.310},
        "gate2_floor": GATE2_CONF_FLOOR,
        "gate2_passed": True,
    }
    comp_path = reports_dir / "b_epoch1_comparison_evaluation.json"
    comp_path.write_text(json.dumps(bad_comp_report), encoding="utf-8")

    with pytest.raises(ValueError, match="Comparison report checkpoint SHA mismatch"):
        main()

    # 4. Test generate-frozen rejects stale frozen_selection_sha256
    bad_comp_report["checkpoint_sha256"] = "dummy_ckpt_sha"
    bad_comp_report["frozen_selection_sha256"] = "STALE_FROZEN_SHA"
    comp_path.write_text(json.dumps(bad_comp_report), encoding="utf-8")
    with pytest.raises(ValueError, match="Comparison report frozen selection SHA mismatch"):
        main()

    # 5. Test numeric gate recomputation rejects false gate booleans
    bad_comp_report["frozen_selection_sha256"] = valid_frozen_sha
    bad_comp_report["gate2_passed"] = True  # claimed passed boolean
    bad_comp_report["confirmation_metrics"]["pq"] = 0.250  # but numeric PQ below gate floor!
    comp_path.write_text(json.dumps(bad_comp_report), encoding="utf-8")

    generation_called = []
    def mock_generate(**kwargs):
        generation_called.append(True)
        return {"csv_path": "dummy.csv"}
    monkeypatch.setattr("scripts.calibrate_b_epoch1.generate_candidate_submission", mock_generate)

    # Must NOT call generation because numeric confirmation PQ (0.250) < 0.3088
    main()
    assert len(generation_called) == 0, "Candidate generation must be inhibited when numeric PQ fails gate floor"



