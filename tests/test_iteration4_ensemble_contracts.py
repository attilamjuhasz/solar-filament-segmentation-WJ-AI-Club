from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from src.contracts import NATIVE_IMAGE_SHAPE
from src.inference.engine import compute_file_sha256, compute_state_dict_sha256
from src.inference.ensemble import (
    canonical_ensemble_definition_json,
    compute_ensemble_definition_sha256,
    compute_ensemble_foreground_prob,
)
from src.inference.rle import audit_submission_and_manifest, encode_instance
from scripts.calibrate_foreground_ensemble import (
    GATE1_TUNING_TARGET,
    GATE2_CONF_FLOOR,
    GRID_ALPHAS,
    GRID_LOW_THRESHOLDS,
    FIXED_HIGH_THRESHOLD,
    FIXED_MIN_AREA,
    FIXED_MAX_INSTANCES,
    DEFAULT_PARENT_PATH,
    DEFAULT_B1_PATH,
    EXPECTED_PARENT_FILE_SHA256,
    EXPECTED_PARENT_STATE_SHA256,
    EXPECTED_B1_FILE_SHA256,
    EXPECTED_B1_STATE_SHA256,
    EXPECTED_FOLDS_MANIFEST_SHA256,
    EXPECTED_PARTITIONS_SHA256,
    EXPECTED_ANNOTATIONS_SHA256,
    verify_ensemble_inputs,
    validate_frozen_winner_binding,
    validate_comparison_report_binding,
    extract_ensemble_instances,
    evaluate_setting_on_maps_with_evidence,
    main,
)
from scripts.calibrate_b_epoch1 import check_deadline


# =========================================================================
# Fixture Helpers
# =========================================================================
def make_tiny_checkpoint(path: Path, tensor_val: float = 1.0) -> tuple[str, str]:
    """Create a real tiny serialized PyTorch checkpoint with valid model_state_dict."""
    sd = {"conv.weight": torch.full((1, 1, 3, 3), tensor_val, dtype=torch.float32)}
    ckpt = {"model_state_dict": sd, "epoch": 1}
    torch.save(ckpt, path)
    f_sha = compute_file_sha256(path)
    s_sha = compute_state_dict_sha256(sd)
    return f_sha, s_sha


def make_valid_selection_config(path: Path, comp_defs: list, pp_params: dict) -> str:
    """Create a real selection config artifact on disk and return its SHA256."""
    sel_cfg = {
        "components": comp_defs,
        "inference_config": dict(pp_params),
        "postprocess_params": dict(pp_params),
    }
    content = json.dumps(sel_cfg, indent=2, sort_keys=True)
    path.write_text(content, encoding="utf-8")
    return compute_file_sha256(path)


# =========================================================================
# Regression 1: Convex Probability Averaging & FP32 Input Guarantee
# =========================================================================
def test_ensemble_convex_probability_averaging_and_fp32():
    """Verify FP32 convex combination, explicit rejection of float64/non-FP32, and range/shape checks."""
    # 1. Valid convex combination
    p1 = np.full(NATIVE_IMAGE_SHAPE, 0.2, dtype=np.float32)
    p2 = np.full(NATIVE_IMAGE_SHAPE, 0.8, dtype=np.float32)
    alpha = 0.5
    ens = compute_ensemble_foreground_prob(p1, p2, alpha)

    assert ens.dtype == np.float32
    assert ens.shape == NATIVE_IMAGE_SHAPE
    assert np.isclose(ens[0, 0], 0.5, atol=1e-6)

    # Unequal weights
    ens_025 = compute_ensemble_foreground_prob(p1, p2, 0.25)
    assert np.isclose(ens_025[0, 0], 0.35, atol=1e-6)

    # 2. Adversarial Gap 6: Explicitly reject float64 (no silent casting)
    p1_f64 = np.full(NATIVE_IMAGE_SHAPE, 0.2, dtype=np.float64)
    with pytest.raises(TypeError, match="must have float32 dtype"):
        compute_ensemble_foreground_prob(p1_f64, p2, 0.5)

    p2_f64 = np.full(NATIVE_IMAGE_SHAPE, 0.8, dtype=np.float64)
    with pytest.raises(TypeError, match="must have float32 dtype"):
        compute_ensemble_foreground_prob(p1, p2_f64, 0.5)

    p1_int = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.int32)
    with pytest.raises(TypeError, match="must have float32 dtype"):
        compute_ensemble_foreground_prob(p1_int, p2, 0.5)

    # 3. Reject invalid alpha values (non-finite, out of range, bool, string)
    for bad_alpha in [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), float("-inf"), True, False, "0.5"]:
        with pytest.raises((ValueError, TypeError)):
            compute_ensemble_foreground_prob(p1, p2, bad_alpha)

    # 4. Reject shape mismatch
    bad_shape = np.zeros((512, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_ensemble_foreground_prob(bad_shape, p2, 0.5)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_ensemble_foreground_prob(p1, bad_shape, 0.5)

    # 5. Reject non-finite values (NaN / Inf)
    p1_nan = p1.copy()
    p1_nan[10, 10] = np.nan
    with pytest.raises(ValueError, match="Non-finite values"):
        compute_ensemble_foreground_prob(p1_nan, p2, 0.5)

    p2_inf = p2.copy()
    p2_inf[10, 10] = np.inf
    with pytest.raises(ValueError, match="Non-finite values"):
        compute_ensemble_foreground_prob(p1, p2_inf, 0.5)

    # 6. Reject out of probability range (< 0 or > 1)
    p1_neg = p1.copy()
    p1_neg[0, 0] = -0.01
    with pytest.raises(ValueError, match="out of probability range"):
        compute_ensemble_foreground_prob(p1_neg, p2, 0.5)

    p2_over = p2.copy()
    p2_over[0, 0] = 1.01
    with pytest.raises(ValueError, match="out of probability range"):
        compute_ensemble_foreground_prob(p1, p2_over, 0.5)


# =========================================================================
# Regression 2: Six-Grid Coverage and Deterministic Tie-Breaking
# =========================================================================
def test_ensemble_six_grid_coverage_and_tie_breaking():
    """Verify exactly 6 Cartesian settings and the multi-tier deterministic tie-breaking rule."""
    assert len(GRID_ALPHAS) == 3
    assert len(GRID_LOW_THRESHOLDS) == 2
    total_settings = len(GRID_ALPHAS) * len(GRID_LOW_THRESHOLDS)
    assert total_settings == 6

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

    # Case A: Outright higher PQ wins
    item1 = {"config": {"alpha": 0.50, "low_threshold": 0.65}, "metrics": {"pq": 0.325, "fp": 50}}
    item2 = {"config": {"alpha": 0.50, "low_threshold": 0.60}, "metrics": {"pq": 0.320, "fp": 40}}
    sorted_a = sorted([item2, item1], key=rank_key)
    assert sorted_a[0]["config"]["low_threshold"] == 0.65

    # Case B: Equal PQ, lower FP wins
    item_fp1 = {"config": {"alpha": 0.50, "low_threshold": 0.65}, "metrics": {"pq": 0.325, "fp": 50}}
    item_fp2 = {"config": {"alpha": 0.50, "low_threshold": 0.60}, "metrics": {"pq": 0.325, "fp": 45}}
    sorted_b = sorted([item_fp1, item_fp2], key=rank_key)
    assert sorted_b[0]["metrics"]["fp"] == 45

    # Case C: Equal PQ, equal FP, alpha closest to 0.50 wins
    item_al1 = {"config": {"alpha": 0.25, "low_threshold": 0.65}, "metrics": {"pq": 0.325, "fp": 50}}
    item_al2 = {"config": {"alpha": 0.50, "low_threshold": 0.65}, "metrics": {"pq": 0.325, "fp": 50}}
    sorted_c = sorted([item_al1, item_al2], key=rank_key)
    assert sorted_c[0]["config"]["alpha"] == 0.50

    # Case D: Equal PQ, equal FP, equal alpha distance (0.25 vs 0.75), low closest to 0.65 wins
    item_low1 = {"config": {"alpha": 0.25, "low_threshold": 0.60}, "metrics": {"pq": 0.325, "fp": 50}}
    item_low2 = {"config": {"alpha": 0.25, "low_threshold": 0.65}, "metrics": {"pq": 0.325, "fp": 50}}
    sorted_d = sorted([item_low1, item_low2], key=rank_key)
    assert sorted_d[0]["config"]["low_threshold"] == 0.65


# =========================================================================
# Regression 3: Canonical Ensemble Definition & Strict Manifest Audit
# =========================================================================
def test_ensemble_definition_and_strict_manifest_audit(tmp_path):
    """Test positive audit with real tiny serialized checkpoints and real selection config."""
    csv_file = tmp_path / "submission.csv"
    manifest_file = tmp_path / "submission.manifest.json"
    ckpt_parent = tmp_path / "parent.pt"
    ckpt_b1 = tmp_path / "b1.pt"
    sel_cfg_path = tmp_path / "selection_config.json"

    # 1. Create real tiny serialized checkpoints
    parent_file_sha, parent_state_sha = make_tiny_checkpoint(ckpt_parent, 1.0)
    b1_file_sha, b1_state_sha = make_tiny_checkpoint(ckpt_b1, 2.0)

    obs1 = "20110120105534Ch"
    obs2 = "20110306082634Lh"
    dummy_mask = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1
    valid_rle = encode_instance(dummy_mask)

    # Valid CSV
    rows = [(f"{obs1}_1", valid_rle), (f"{obs2}_1", valid_rle)]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        for fid, rle in rows:
            writer.writerow([fid, rle])
    csv_sha = compute_file_sha256(csv_file)

    comp_records = [
        {"name": "parent", "checkpoint_sha256": parent_file_sha, "model_state_sha256": parent_state_sha, "weight": 0.5},
        {"name": "b1", "checkpoint_sha256": b1_file_sha, "model_state_sha256": b1_state_sha, "weight": 0.5},
    ]
    pp_params = {
        "method": "connected_components",
        "high_threshold": 0.85,
        "low_threshold": 0.65,
        "min_area": 400,
        "max_instances": 20,
        "tile_size": 512,
        "stride": 256,
        "tile_batch_size": 4,
        "norm_mode": "imagenet",
        "precision": "float32",
        "inference_policy": "identity",
    }
    ens_def_sha = compute_ensemble_definition_sha256(comp_records, pp_params)

    # Create real selection config artifact
    sel_cfg_sha = make_valid_selection_config(sel_cfg_path, comp_records, pp_params)

    manifest_comps = [
        {"name": "parent", "checkpoint_path": str(ckpt_parent), "checkpoint_sha256": parent_file_sha, "model_state_sha256": parent_state_sha, "weight": 0.5},
        {"name": "b1", "checkpoint_path": str(ckpt_b1), "checkpoint_sha256": b1_file_sha, "model_state_sha256": b1_state_sha, "weight": 0.5},
    ]

    valid_manifest = {
        "csv_sha256": csv_sha,
        "ensemble_type": "two_component_foreground_ensemble",
        "inference_policy": "identity",
        "ensemble_definition_sha256": ens_def_sha,
        "selection_config_path": str(sel_cfg_path),
        "selection_config_sha256": sel_cfg_sha,
        "components": manifest_comps,
        "postprocess_params": pp_params,
        "total_test_observations": 2,
        "total_instances": 2,
        "detected_observations_count": 2,
        "abstained_observations_count": 0,
        "observations": [
            {"observation_id": obs1, "status": "processed", "instance_count": 1},
            {"observation_id": obs2, "status": "processed", "instance_count": 1},
        ],
    }
    manifest_file.write_text(json.dumps(valid_manifest), encoding="utf-8")

    # Positive test: passes cleanly
    audit_rep = audit_submission_and_manifest(csv_file, manifest_file, expected_observation_ids={obs1, obs2})
    assert audit_rep["is_valid"] is True

    # Tampered ensemble definition hash fails
    bad_def = dict(valid_manifest)
    bad_def["ensemble_definition_sha256"] = "0" * 64
    manifest_file.write_text(json.dumps(bad_def), encoding="utf-8")
    with pytest.raises(ValueError, match="Tampered ensemble definition hash"):
        audit_submission_and_manifest(csv_file, manifest_file)

    # Tampered weights (not summing to 1.0) fails
    bad_w = json.loads(json.dumps(valid_manifest))
    bad_w["components"][0]["weight"] = 0.7
    manifest_file.write_text(json.dumps(bad_w), encoding="utf-8")
    with pytest.raises(ValueError, match="Ensemble component weights must sum to 1.0"):
        audit_submission_and_manifest(csv_file, manifest_file)

    # Non-identity top-level policy fails
    bad_pol = dict(valid_manifest)
    bad_pol["inference_policy"] = "flip_tta"
    manifest_file.write_text(json.dumps(bad_pol), encoding="utf-8")
    with pytest.raises(ValueError, match="must be 'identity'"):
        audit_submission_and_manifest(csv_file, manifest_file)

    # Legacy single-model compatibility passes
    single_model_manifest = {
        "csv_sha256": csv_sha,
        "total_test_observations": 2,
        "total_instances": 2,
        "detected_observations_count": 2,
        "abstained_observations_count": 0,
        "checkpoint_sha256": "e" * 64,
        "postprocess_params": pp_params,
        "observations": [
            {"observation_id": obs1, "status": "processed", "instance_count": 1},
            {"observation_id": obs2, "status": "processed", "instance_count": 1},
        ],
    }
    manifest_file.write_text(json.dumps(single_model_manifest), encoding="utf-8")
    audit_single = audit_submission_and_manifest(csv_file, manifest_file, expected_observation_ids={obs1, obs2})
    assert audit_single["is_valid"] is True


# =========================================================================
# Regression 4: Six Adversarial Cases from Rejection Review
# =========================================================================
def test_adversarial_case_1_missing_component_files_and_selection_config(tmp_path):
    """Adversarial Case 1: missing checkpoint files or selection config must fail audit."""
    csv_file = tmp_path / "sub.csv"
    manifest_file = tmp_path / "sub.manifest.json"
    ckpt_parent = tmp_path / "parent.pt"
    ckpt_b1 = tmp_path / "b1.pt"
    sel_cfg_path = tmp_path / "selection_config.json"

    p_fsha, p_ssha = make_tiny_checkpoint(ckpt_parent, 1.0)
    b_fsha, b_ssha = make_tiny_checkpoint(ckpt_b1, 2.0)

    dummy_mask = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1
    rle = encode_instance(dummy_mask)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerow(["20110120105534Ch_1", rle])
    csv_sha = compute_file_sha256(csv_file)

    comp_records = [
        {"name": "parent", "checkpoint_sha256": p_fsha, "model_state_sha256": p_ssha, "weight": 0.5},
        {"name": "b1", "checkpoint_sha256": b_fsha, "model_state_sha256": b_ssha, "weight": 0.5},
    ]
    pp_params = {
        "method": "connected_components", "high_threshold": 0.85, "low_threshold": 0.65,
        "min_area": 400, "max_instances": 20, "tile_size": 512, "stride": 256,
        "tile_batch_size": 4, "norm_mode": "imagenet", "precision": "float32", "inference_policy": "identity",
    }
    ens_def_sha = compute_ensemble_definition_sha256(comp_records, pp_params)
    sel_cfg_sha = make_valid_selection_config(sel_cfg_path, comp_records, pp_params)

    # 1. Missing checkpoint_path entirely
    m_no_path = {
        "csv_sha256": csv_sha,
        "ensemble_type": "two_component_foreground_ensemble",
        "inference_policy": "identity",
        "ensemble_definition_sha256": ens_def_sha,
        "selection_config_path": str(sel_cfg_path),
        "selection_config_sha256": sel_cfg_sha,
        "components": [
            {"name": "parent", "checkpoint_sha256": p_fsha, "model_state_sha256": p_ssha, "weight": 0.5},
            {"name": "b1", "checkpoint_path": str(ckpt_b1), "checkpoint_sha256": b_fsha, "model_state_sha256": b_ssha, "weight": 0.5},
        ],
        "postprocess_params": pp_params,
        "observations": [{"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 1}],
    }
    manifest_file.write_text(json.dumps(m_no_path), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required checkpoint_path"):
        audit_submission_and_manifest(csv_file, manifest_file)

    # 2. Nonexistent checkpoint file
    m_nonexist_path = json.loads(json.dumps(m_no_path))
    m_nonexist_path["components"][0]["checkpoint_path"] = str(tmp_path / "nonexistent.pt")
    manifest_file.write_text(json.dumps(m_nonexist_path), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="checkpoint file not found"):
        audit_submission_and_manifest(csv_file, manifest_file)

    # 3. Missing selection_config_path
    m_no_sel = json.loads(json.dumps(m_no_path))
    m_no_sel["components"][0]["checkpoint_path"] = str(ckpt_parent)
    del m_no_sel["selection_config_path"]
    manifest_file.write_text(json.dumps(m_no_sel), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'selection_config_path'"):
        audit_submission_and_manifest(csv_file, manifest_file)

    # 4. Tampered selection config bytes
    m_tampered_cfg = json.loads(json.dumps(m_no_path))
    m_tampered_cfg["components"][0]["checkpoint_path"] = str(ckpt_parent)
    m_tampered_cfg["selection_config_sha256"] = "1" * 64
    manifest_file.write_text(json.dumps(m_tampered_cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_config file content SHA mismatch"):
        audit_submission_and_manifest(csv_file, manifest_file)

    # 5. Tampered model_state_sha256
    m_tampered_state = json.loads(json.dumps(m_no_path))
    m_tampered_state["components"][0]["checkpoint_path"] = str(ckpt_parent)
    m_tampered_state["components"][0]["model_state_sha256"] = "2" * 64
    manifest_file.write_text(json.dumps(m_tampered_state), encoding="utf-8")
    with pytest.raises(ValueError, match="model_state_sha256 mismatch"):
        audit_submission_and_manifest(csv_file, manifest_file)


def test_adversarial_case_2_duplicate_component_names():
    """Adversarial Case 2: duplicate component names must be rejected."""
    pp = {"method": "connected_components", "high_threshold": 0.85, "low_threshold": 0.65}
    dup_comps = [
        {"name": "parent", "checkpoint_sha256": "a" * 64, "model_state_sha256": "b" * 64, "weight": 0.5},
        {"name": "parent", "checkpoint_sha256": "c" * 64, "model_state_sha256": "d" * 64, "weight": 0.5},
    ]
    with pytest.raises(ValueError, match="Duplicate component name"):
        canonical_ensemble_definition_json(dup_comps, pp)


def test_adversarial_case_3_deadline_missing(monkeypatch, tmp_path):
    """Adversarial Case 3: computational stages must reject missing or None deadline."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    for comp_stage in ["cache-tuning", "calibrate", "evaluate-frozen", "cache-test", "generate-frozen", "all"]:
        monkeypatch.setattr(
            sys, "argv",
            ["calibrate_foreground_ensemble.py", "--stage", comp_stage, "--reports-dir", str(reports_dir), "--cache-dir", str(cache_dir)]
        )
        with pytest.raises(ValueError, match="requires a finite future absolute deadline"):
            main()


def test_adversarial_case_4_and_5_deadline_nan_and_infinite(monkeypatch, tmp_path):
    """Adversarial Cases 4 & 5: check_deadline and CLI must reject NaN and Inf."""
    # check_deadline direct rejection
    for bad_dl in [float("nan"), float("inf"), float("-inf"), True, False, "12345"]:
        with pytest.raises((ValueError, TypeError)):
            check_deadline(bad_dl, "test_operation")

    # CLI rejection
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setattr(
        sys, "argv",
        ["calibrate_foreground_ensemble.py", "--stage", "cache-tuning", "--deadline", "nan", "--reports-dir", str(reports_dir), "--cache-dir", str(cache_dir)]
    )
    with pytest.raises(ValueError, match="deadline must be finite"):
        main()

    monkeypatch.setattr(
        sys, "argv",
        ["calibrate_foreground_ensemble.py", "--stage", "cache-tuning", "--deadline", "inf", "--reports-dir", str(reports_dir), "--cache-dir", str(cache_dir)]
    )
    with pytest.raises(ValueError, match="deadline must be finite"):
        main()


# =========================================================================
# Regression 5: Staged Lifecycle, Binding Validators, and Gate 2 Halting
# =========================================================================
def test_binding_validators_and_gate_inhibition(tmp_path, monkeypatch):
    """Test validate_frozen_winner_binding, validate_comparison_report_binding, and gate inhibition."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    verified_info = {
        "parent_path": "parent.pt",
        "parent_checkpoint_sha256": "parent_sha",
        "parent_model_state_sha256": "parent_model_sha",
        "b1_path": "b1.pt",
        "b1_checkpoint_sha256": "b1_sha",
        "b1_model_state_sha256": "b1_model_sha",
        "partitions_path": "parts.json",
        "partitions_sha256": "part_sha",
        "manifest_path": "folds.json",
        "manifest_sha256": "man_sha",
        "annotations_path": "ann.json",
        "annotations_sha256": "ann_sha",
        "tuning_observations": [f"tune_{i}" for i in range(10)],
        "confirmation_observations": [f"conf_{i}" for i in range(9)],
        "fold_assignments": {},
        "parent_ckpt_data": {"train_observations": []},
        "b1_ckpt_data": {"train_observations": []},
    }

    dummy_frozen = {
        "frozen_timestamp": "2026-10-01T00:00:00Z",
        "parent_checkpoint_sha256": "parent_sha",
        "parent_model_state_sha256": "parent_model_sha",
        "b1_checkpoint_sha256": "b1_sha",
        "b1_model_state_sha256": "b1_model_sha",
        "partitions_sha256": "part_sha",
        "manifest_sha256": "man_sha",
        "annotations_sha256": "ann_sha",
        "tile_batch_size": 4,
        "selected_config": {
            "alpha": 0.50, "high_threshold": 0.85, "low_threshold": 0.65,
            "min_area": 400, "max_instances": 20, "method": "connected_components"
        },
        "tuning_metrics": {
            "tp": 20, "fp": 10, "fn": 10,
            "sq": 0.5, "rq": 2.0 / 3.0,
            "pq": 1.0 / 3.0,
        },
        "gate1_target": GATE1_TUNING_TARGET,
        "gate1_passed": True,
        "evaluated_physical_ids": [f"tune_{i}" for i in range(10)],
    }
    frozen_path = reports_dir / "foreground_ensemble_frozen_winner.json"
    frozen_path.write_text(json.dumps(dummy_frozen), encoding="utf-8")

    # 1. validate_frozen_winner_binding passes cleanly
    loaded_frozen = validate_frozen_winner_binding(frozen_path, verified_info)
    assert loaded_frozen["selected_config"]["alpha"] == 0.50

    # Tampered b1 sha in frozen winner raises ValueError
    bad_frozen = dict(dummy_frozen)
    bad_frozen["b1_checkpoint_sha256"] = "WRONG_B1"
    bad_frozen_path = reports_dir / "bad_frozen.json"
    bad_frozen_path.write_text(json.dumps(bad_frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="b1_checkpoint_sha256 mismatch"):
        validate_frozen_winner_binding(bad_frozen_path, verified_info)

    # 2. validate_comparison_report_binding
    frozen_sha = compute_file_sha256(frozen_path)
    dummy_comp = {
        "timestamp": "2026-10-01T00:00:00Z",
        "parent_checkpoint_sha256": "parent_sha",
        "parent_model_state_sha256": "parent_model_sha",
        "b1_checkpoint_sha256": "b1_sha",
        "b1_model_state_sha256": "b1_model_sha",
        "frozen_selection_sha256": frozen_sha,
        "frozen_config": dummy_frozen["selected_config"],
        "partitions_sha256": "part_sha",
        "manifest_sha256": "man_sha",
        "annotations_sha256": "ann_sha",
        "evaluated_physical_ids": [f"conf_{i}" for i in range(9)],
        "total_entries": 15,
        "confirmation_metrics": {
            "tp": 12, "fp": 6, "fn": 6,
            "sq": 0.5, "rq": 2.0 / 3.0,
            "pq": 1.0 / 3.0,
        },
        "gate2_floor": GATE2_CONF_FLOOR,
        "gate2_passed": True,
    }
    comp_path = reports_dir / "foreground_ensemble_comparison_evaluation.json"
    comp_path.write_text(json.dumps(dummy_comp), encoding="utf-8")

    loaded_comp = validate_comparison_report_binding(comp_path, frozen_path, verified_info)
    assert loaded_comp["gate2_passed"] is True

    # 3. Numeric PQ below GATE2_CONF_FLOOR fails even if gate2_passed is claimed True
    bad_comp = json.loads(json.dumps(dummy_comp))
    bad_comp["confirmation_metrics"] = {
        "tp": 6, "fp": 12, "fn": 12,
        "sq": 0.5, "rq": 1.0 / 3.0,
        "pq": 1.0 / 6.0,  # 0.166666 < GATE2_CONF_FLOOR (0.310187)
    }
    bad_comp["gate2_passed"] = True
    bad_comp_path = reports_dir / "bad_comp.json"
    bad_comp_path.write_text(json.dumps(bad_comp), encoding="utf-8")
    with pytest.raises(ValueError, match="failed Gate 2"):
        validate_comparison_report_binding(bad_comp_path, frozen_path, verified_info)

    # 4. Generate-frozen via CLI requires comparison report and passing Gate 2
    monkeypatch.setattr("scripts.calibrate_foreground_ensemble.verify_ensemble_inputs", lambda **kwargs: verified_info)
    monkeypatch.setattr("scripts.calibrate_foreground_ensemble.load_coco_annotations", lambda path: None)

    class MockDataset:
        def __init__(self, **kwargs): pass
        def get_observation_annotations(self, obs):
            if obs.startswith("tune"):
                idx = int(obs.split("_")[1])
                return [object()] * (3 if idx < 4 else 2)  # 24 total entries
            else:
                idx = int(obs.split("_")[1])
                return [object()] * (2 if idx < 6 else 1)  # 15 total entries

    monkeypatch.setattr("scripts.calibrate_foreground_ensemble.SolarFilamentDataset", MockDataset)
    generation_called = []
    def mock_generate(**kwargs):
        generation_called.append(True)
        return {"csv_path": "dummy.csv"}
    monkeypatch.setattr("scripts.calibrate_foreground_ensemble.generate_ensemble_candidate_submission", mock_generate)

    # With bad comp report on standard path, main() must reject
    comp_path.write_text(json.dumps(bad_comp), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["calibrate_foreground_ensemble.py", "--stage", "generate-frozen", "--deadline", str(time.time() + 3600),
         "--reports-dir", str(reports_dir), "--cache-dir", str(cache_dir)]
    )
    with pytest.raises(ValueError, match="failed Gate 2"):
        main()
    assert len(generation_called) == 0


def test_deadline_expiration_during_loop(tmp_path, monkeypatch):
    """Test monkeypatched clock triggering deadline expiration between items."""
    # Start with 5 seconds left, then jump past deadline
    start_time = 1000.0
    deadline = 1005.0

    current_time = [start_time]
    def mock_time():
        return current_time[0]
    monkeypatch.setattr(time, "time", mock_time)

    # Step 1: before deadline -> passes
    check_deadline(deadline, "item 1")

    # Step 2: time jumps past deadline -> raises TimeoutError
    current_time[0] = 1006.0
    with pytest.raises(TimeoutError, match="Absolute deadline exceeded"):
        check_deadline(deadline, "item 2")


# =========================================================================
# Regression 6: Real-Input Preflight Cryptographic and Membership Audit
# =========================================================================
def test_ensemble_real_input_preflight():
    """Verify verify_ensemble_inputs passes cleanly on real disk files with exact SHAs and disjoint fold 0 membership."""
    assert DEFAULT_PARENT_PATH.is_file(), f"Real parent checkpoint missing at: {DEFAULT_PARENT_PATH}"
    assert DEFAULT_B1_PATH.is_file(), f"Real B1 checkpoint missing at: {DEFAULT_B1_PATH}"

    info = verify_ensemble_inputs(
        parent_path=DEFAULT_PARENT_PATH,
        b1_path=DEFAULT_B1_PATH,
    )

    # Parent checkpoint SHA and state SHA
    assert info["parent_checkpoint_sha256"] == EXPECTED_PARENT_FILE_SHA256
    assert EXPECTED_PARENT_FILE_SHA256 == "9580632d5de717999bb1a60ee940e3f14ee716e853d87d1220be456db62d344a"
    assert info["parent_model_state_sha256"] == EXPECTED_PARENT_STATE_SHA256
    assert EXPECTED_PARENT_STATE_SHA256 == "f7d19980cd2ed382ca4683b6b0932ae404e98359976b5bd9eaebbc44bb4f202c"

    # B1 checkpoint SHA and state SHA
    assert info["b1_checkpoint_sha256"] == EXPECTED_B1_FILE_SHA256
    assert EXPECTED_B1_FILE_SHA256 == "8ae7cb726b35d17949a58e9dcd26dd09bf4ecdc7c6f3c10d18d11210e013cf0a"
    assert info["b1_model_state_sha256"] == EXPECTED_B1_STATE_SHA256
    assert EXPECTED_B1_STATE_SHA256 == "0824c4972126ca4cd65347c2b13ba58413932c5ffea53675e4355f453b7eadb5"

    # Partitions SHA
    assert info["partitions_sha256"] == EXPECTED_PARTITIONS_SHA256
    assert EXPECTED_PARTITIONS_SHA256 == "81022d8462bec5031c9b1f3760fe310a9cb15008f00770ab57a62a4f39bff467"

    # Folds Manifest SHA
    assert info["manifest_sha256"] == EXPECTED_FOLDS_MANIFEST_SHA256
    assert EXPECTED_FOLDS_MANIFEST_SHA256 == "2495791553a6107873c9962d05a0ba40d25873420937d483641121e7e9c989bd"

    # Annotations SHA
    assert info["annotations_sha256"] == EXPECTED_ANNOTATIONS_SHA256
    assert EXPECTED_ANNOTATIONS_SHA256 == "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"

    # Partition sets: 10 tuning, 9 confirmation, completely disjoint
    assert len(info["tuning_observations"]) == 10
    assert len(info["confirmation_observations"]) == 9
    assert len(set(info["tuning_observations"]) & set(info["confirmation_observations"])) == 0
