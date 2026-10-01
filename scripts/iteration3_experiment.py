from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.contracts import NATIVE_IMAGE_SHAPE
from src.inference.engine import compute_file_sha256, compute_state_dict_sha256
from evaluate import evaluate_oof
from train import train


EXPECTED_PARENT_FILE_SHA256 = "9580632d5de717999bb1a60ee940e3f14ee716e853d87d1220be456db62d344a"
EXPECTED_PARENT_STATE_SHA256 = "f7d19980cd2ed382ca4683b6b0932ae404e98359976b5bd9eaebbc44bb4f202c"
EXPECTED_FOLDS_MANIFEST_SHA256 = "2495791553a6107873c9962d05a0ba40d25873420937d483641121e7e9c989bd"
EXPECTED_PARTITIONS_SHA256 = "81022d8462bec5031c9b1f3760fe310a9cb15008f00770ab57a62a4f39bff467"
EXPECTED_MINING_BANK_SHA256 = "500abcba5d4928d0cb5a31ce9ccefd896c14c08be6df5eeae77edece3885bb03"

# Candidate 2 Benchmarks and Candidate 3 Targets
CANDIDATE_2_TUNING_PQ = 0.312649
CANDIDATE_2_CONFIRMATION_PQ = 0.313808
CANDIDATE_3_GATE1_TUNING_TARGET = 0.31764891564223335  # tuning >= 0.317649
CANDIDATE_3_GATE2_CONF_FLOOR = 0.30880793475475555     # comparison >= 0.308808

# Fixed Candidate 2 Inference Settings (Connected Components, h=0.85, l=0.70, area=400, cap=20)
FIXED_INF_METHOD = "connected_components"
FIXED_HIGH_THRESH = 0.85
FIXED_LOW_THRESH = 0.70
FIXED_MIN_AREA = 400
FIXED_MAX_INSTANCES = 20
FIXED_TILE_SIZE = 512
FIXED_STRIDE = 256
FIXED_TILE_BATCH_SIZE = 16
FIXED_NORM_MODE = "imagenet"
FIXED_PRECISION = "float32"

CLEANUP_MARGIN_SECONDS = 30.0


def is_checkpoint_diagnostic(ckpt_data: Dict[str, Any]) -> bool:
    """Determine whether checkpoint is diagnostic or smoke output, rendering it ineligible for production."""
    rc = ckpt_data.get("runtime_config", {})
    fp = ckpt_data.get("finetune_parent", {})
    fprov = ckpt_data.get("finetune_provenance", {})
    return bool(
        rc.get("smoke", False)
        or rc.get("is_diagnostic", False)
        or fp.get("is_diagnostic", False)
        or fprov.get("is_diagnostic", False)
        or (ckpt_data.get("eligible_for_promotion", True) is False)
    )


def verify_immutable_inputs(
    parent_path: Path,
    folds_manifest_path: Path,
    partitions_path: Path,
    mining_bank_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Verify cryptographic hashes of all immutable inputs prior to experiment execution."""
    print("=== Step 0: Cryptographic Provenance Verification ===")
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent checkpoint not found at {parent_path}")
    parent_file_sha = compute_file_sha256(parent_path)
    if parent_file_sha != EXPECTED_PARENT_FILE_SHA256:
        raise ValueError(f"Parent file SHA mismatch! Expected {EXPECTED_PARENT_FILE_SHA256}, got {parent_file_sha}")

    parent_ckpt = torch.load(str(parent_path), map_location="cpu", weights_only=False)
    parent_state_sha = compute_state_dict_sha256(parent_ckpt["model_state_dict"])
    if parent_state_sha != EXPECTED_PARENT_STATE_SHA256:
        raise ValueError(f"Parent state dict SHA mismatch! Expected {EXPECTED_PARENT_STATE_SHA256}, got {parent_state_sha}")

    if not folds_manifest_path.is_file():
        raise FileNotFoundError(f"Folds manifest not found at {folds_manifest_path}")
    folds_sha = compute_file_sha256(folds_manifest_path)
    if folds_sha != EXPECTED_FOLDS_MANIFEST_SHA256:
        raise ValueError(f"Folds manifest SHA mismatch! Expected {EXPECTED_FOLDS_MANIFEST_SHA256}, got {folds_sha}")

    if not partitions_path.is_file():
        raise FileNotFoundError(f"Partitions file not found at {partitions_path}")
    parts_sha = compute_file_sha256(partitions_path)
    if parts_sha != EXPECTED_PARTITIONS_SHA256:
        raise ValueError(f"Partitions SHA mismatch! Expected {EXPECTED_PARTITIONS_SHA256}, got {parts_sha}")

    bank_sha = ""
    if mining_bank_path:
        if not mining_bank_path.is_file():
            raise FileNotFoundError(f"Mining bank not found at {mining_bank_path}")
        bank_sha = compute_file_sha256(mining_bank_path)
        if bank_sha != EXPECTED_MINING_BANK_SHA256:
            raise ValueError(f"Mining bank SHA mismatch! Expected {EXPECTED_MINING_BANK_SHA256}, got {bank_sha}")

    print(f"  Parent Checkpoint File SHA256 : {parent_file_sha}")
    print(f"  Parent Model State SHA256     : {parent_state_sha}")
    print(f"  Folds Manifest SHA256         : {folds_sha}")
    print(f"  Partitions SHA256             : {parts_sha}")
    if bank_sha:
        print(f"  Mining Bank SHA256            : {bank_sha}")
    print("All immutable inputs verified successfully.\n")

    return {
        "parent_file_sha256": parent_file_sha,
        "parent_model_state_sha256": parent_state_sha,
        "folds_manifest_sha256": folds_sha,
        "partitions_sha256": parts_sha,
        "mining_bank_sha256": bank_sha,
    }


def assert_report_inference_config(eval_report: Dict[str, Any]) -> None:
    """Assert that evaluation strictly used Candidate 2 resolved inference configuration."""
    act_cfg = eval_report.get("resolved_inference_config", {})
    expected = {
        "method": FIXED_INF_METHOD,
        "high_threshold": FIXED_HIGH_THRESH,
        "low_threshold": FIXED_LOW_THRESH,
        "min_area": FIXED_MIN_AREA,
        "max_instances": FIXED_MAX_INSTANCES,
    }
    for k, v in expected.items():
        if act_cfg.get(k) != v:
            raise ValueError(
                f"Inference config mismatch in report: {k}={act_cfg.get(k)}, expected {v}"
            )


def parse_deadline_timestamp(val: Any) -> float:
    """Parse absolute deadline timestamp supporting float/int seconds or ISO 8601 strings."""
    if val is None:
        raise ValueError("Missing required deadline timestamp")
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    try:
        return float(val_str)
    except ValueError:
        pass
    dt_str = val_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.timestamp()
    except Exception as e:
        raise ValueError(f"Could not parse deadline timestamp '{val}': {e}") from e


def resolve_and_validate_arm_checkpoints(
    arm_label: str,
    run_id: str,
    root_dir: Path,
    expected_epochs: int = 3,
    smoke: bool = False,
    expected_parent_sha: str = EXPECTED_PARENT_FILE_SHA256,
    expected_folds_manifest_sha: str = EXPECTED_FOLDS_MANIFEST_SHA256,
    expected_mining_bank_sha: str = EXPECTED_MINING_BANK_SHA256,
    fallback_parent_dir: Optional[Path] = None,
) -> List[Tuple[str, Path, Dict[str, Any]]]:
    """Resolve and validate immutable epoch checkpoints for an arm with strict integrity checks."""
    if not run_id or not isinstance(run_id, str):
        raise ValueError(f"[{arm_label}] run_id must be a non-empty single-component string, got: {run_id!r}")
    if "/" in run_id or "\\" in run_id or ".." in run_id or Path(run_id).is_absolute() or Path(run_id).name != run_id:
        raise ValueError(f"[{arm_label}] Path containment violation: run_id must be a single-component name without path separators or traversal, got: '{run_id}'")

    root_resolved = root_dir.resolve()
    canonical_runs_root = (root_resolved / "artifacts" / "runs").resolve()
    ckpt_dir = (canonical_runs_root / run_id / "checkpoints").resolve()

    # Strict path containment verification: prevent traversal outside canonical runs root
    try:
        ckpt_dir.relative_to(canonical_runs_root)
    except ValueError:
        raise ValueError(f"Path containment violation: {ckpt_dir} is outside {canonical_runs_root}")

    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"[{arm_label}] Checkpoint directory not found: {ckpt_dir}")

    epoch_files = sorted(list(ckpt_dir.glob("epoch_*.pt")))
    if not epoch_files:
        raise FileNotFoundError(f"[{arm_label}] Found 0 epoch checkpoints in {ckpt_dir}")

    # First pass: Check finiteness, skipped updates, and diagnostic marks across all discovered checkpoints
    loaded_checkpoints: List[Tuple[Path, Dict[str, Any]]] = []
    for p in epoch_files:
        data = torch.load(str(p), map_location="cpu", weights_only=False)
        loaded_checkpoints.append((p, data))

        # Check weights finiteness
        state_dict = data.get("model_state_dict", {})
        if not state_dict:
            raise ValueError(f"[{arm_label}] Checkpoint {p.name} has empty model_state_dict")
        all_finite = all(torch.isfinite(val).all().item() for val in state_dict.values())
        if not all_finite:
            raise ValueError(f"Arm {arm_label} invalid: checkpoint {p.name} contains non-finite weights. Entire arm invalidated.")

        # Check skipped updates metadata
        if "skipped_updates" not in data:
            raise ValueError(f"[{arm_label}] Checkpoint {p.name} missing skipped_updates metadata")
        skipped = int(data["skipped_updates"])
        if skipped > 0:
            raise ValueError(f"Arm {arm_label} invalid: checkpoint {p.name} has {skipped} skipped updates. Entire arm invalidated.")

        # Check diagnostic marks in production mode
        if not smoke and is_checkpoint_diagnostic(data):
            raise ValueError("Diagnostic checkpoint detected in production run. Ineligible for production ranking.")

    # Second pass: Require exact completed epoch counts
    if len(epoch_files) != expected_epochs:
        raise ValueError(
            f"[{arm_label}] Expected exactly {expected_epochs} completed epochs, got {len(epoch_files)} in {ckpt_dir}"
        )

    candidates: List[Tuple[str, Path, Dict[str, Any]]] = []
    for idx, (p, ckpt_data) in enumerate(loaded_checkpoints):
        expected_e = idx + 1
        expected_name = f"epoch_{expected_e:03d}.pt"
        if p.name != expected_name:
            raise ValueError(
                f"[{arm_label}] Checkpoint sequence mismatch at index {idx}: expected {expected_name}, got {p.name}"
            )

        # 1. Verify run_id identity
        ckpt_run_id = ckpt_data.get("run_id")
        if ckpt_run_id != run_id:
            raise ValueError(
                f"[{arm_label}] Checkpoint {p.name} run_id mismatch: expected '{run_id}', got '{ckpt_run_id}'"
            )

        # 2. Verify internal epoch number
        ckpt_e = int(ckpt_data.get("epoch", 0))
        if ckpt_e != expected_e:
            raise ValueError(
                f"[{arm_label}] Checkpoint {p.name} internal epoch mismatch: expected {expected_e}, got {ckpt_e}"
            )

        # 3. Check fold metadata
        if not smoke:
            if "fold" not in ckpt_data:
                raise ValueError(f"[{arm_label}] Checkpoint {p.name} missing fold metadata")
            if ckpt_data["fold"] != 0:
                raise ValueError(f"[{arm_label}] Checkpoint {p.name} fold mismatch: expected fold 0, got {ckpt_data['fold']}")
        elif "fold" in ckpt_data and ckpt_data["fold"] != 0:
            raise ValueError(f"[{arm_label}] Checkpoint {p.name} fold mismatch: expected fold 0, got {ckpt_data['fold']}")

        # 4. Check total_epochs metadata
        if not smoke:
            if "total_epochs" not in ckpt_data:
                raise ValueError(f"[{arm_label}] Checkpoint {p.name} missing total_epochs metadata")
            if ckpt_data["total_epochs"] != expected_epochs:
                raise ValueError(f"[{arm_label}] Checkpoint {p.name} total_epochs mismatch: expected {expected_epochs}, got {ckpt_data['total_epochs']}")
        elif "total_epochs" in ckpt_data and ckpt_data["total_epochs"] != expected_epochs:
            raise ValueError(f"[{arm_label}] Checkpoint {p.name} total_epochs mismatch: expected {expected_epochs}, got {ckpt_data['total_epochs']}")

        # 5. Check eligibility metadata
        if not smoke:
            if "eligible_for_promotion" not in ckpt_data:
                raise ValueError(f"[{arm_label}] Checkpoint {p.name} missing eligible_for_promotion metadata")
            if ckpt_data["eligible_for_promotion"] is not True:
                raise ValueError(f"Arm {arm_label} invalid: checkpoint {p.name} has eligible_for_promotion={ckpt_data['eligible_for_promotion']}")

        # 3. Successful updates metadata and sequence
        if "successful_updates" not in ckpt_data:
            raise ValueError(f"[{arm_label}] Checkpoint {p.name} missing successful_updates metadata")
        updates = int(ckpt_data["successful_updates"])
        if updates <= 0:
            raise ValueError(f"[{arm_label}] Checkpoint {p.name} successful updates must be positive (got {updates})")
        if not smoke:
            expected_updates = 142 * expected_e
            if updates != expected_updates:
                raise ValueError(
                    f"[{arm_label}] Checkpoint {p.name} successful updates mismatch: expected {expected_updates}, got {updates}"
                )

        # 7. Parent verification
        if not smoke and expected_parent_sha:
            fp = ckpt_data.get("finetune_parent", {})
            parent_sha = fp.get("file_sha256", "") if isinstance(fp, dict) else ""
            if parent_sha != expected_parent_sha:
                raise ValueError(
                    f"[{arm_label}] Checkpoint {p.name} parent SHA mismatch: expected {expected_parent_sha}, got '{parent_sha}'"
                )

        # 8. Folds manifest verification
        if not smoke and expected_folds_manifest_sha:
            fp = ckpt_data.get("finetune_parent", {})
            man_sha = ckpt_data.get("folds_manifest_sha256") or (fp.get("folds_manifest_sha256", "") if isinstance(fp, dict) else "")
            if man_sha != expected_folds_manifest_sha:
                raise ValueError(
                    f"[{arm_label}] Checkpoint {p.name} folds manifest SHA mismatch: expected {expected_folds_manifest_sha}, got '{man_sha}'"
                )

        # 9. Control (Arm A) vs Intervention (Arm B) contract
        rc = ckpt_data.get("runtime_config", {})
        flips = rc.get("augment_flips") if isinstance(rc, dict) else None
        bank_sha = ckpt_data.get("mining_bank_sha256", "")
        if not smoke:
            if arm_label == "Arm_A":
                if flips is not False:
                    raise ValueError(
                        f"[Arm A] Checkpoint {p.name} has augment_flips={flips}; expected False for Arm A control "
                        f"(augment_flips must be explicitly False)"
                    )
                if bank_sha:
                    raise ValueError(f"[Arm A] Checkpoint {p.name} has active mining bank ({bank_sha}); expected empty for Arm A control")
            elif arm_label == "Arm_B":
                if flips is not True:
                    raise ValueError(f"[Arm B] Checkpoint {p.name} has augment_flips={flips}; expected True for Arm B intervention")
                if not expected_mining_bank_sha:
                    raise ValueError("[Arm B] expected_mining_bank_sha must not be empty for Arm B verification")
                if bank_sha != expected_mining_bank_sha:
                    raise ValueError(
                        f"[Arm B] Checkpoint {p.name} mining bank SHA mismatch: expected {expected_mining_bank_sha}, got '{bank_sha}'"
                    )

        candidates.append((arm_label, p, ckpt_data))

    return candidates


def evaluate_and_select_candidates(
    candidates_to_eval: List[Tuple[str, Path, Dict[str, Any]]],
    input_provenance: Dict[str, str],
    arm_a_run_id: str,
    arm_b_run_id: str,
    arm_a_duration: float,
    arm_b_duration: float,
    start_time: float,
    deadline_time: float,
    device_str: Optional[str] = None,
    smoke: bool = False,
    is_recovery: bool = False,
    eval_fn: Optional[Any] = None,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Shared model selection, tuning evaluation, single comparison, gate audit, and optional candidate generation."""
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent

    _eval = eval_fn if eval_fn is not None else evaluate_oof

    def check_deadline(stage_name: str, margin: float = CLEANUP_MARGIN_SECONDS) -> None:
        now = time.time()
        if (now + margin) >= deadline_time:
            raise TimeoutError(
                f"Compute budget deadline exceeded before/during {stage_name}: "
                f"now={now:.1f}, deadline={deadline_time:.1f}, margin={margin:.1f}s "
                f"(elapsed={(now - start_time) / 60:.1f}m)."
            )

    has_diagnostic_ckpt = any(is_checkpoint_diagnostic(data) for _, _, data in candidates_to_eval)
    if not smoke and has_diagnostic_ckpt:
        raise ValueError("Diagnostic checkpoint detected in production run. Ineligible for production ranking.")

    is_diagnostic_run = smoke or has_diagnostic_ckpt

    # ==========================================================
    # Tuning Partition Evaluation (10 physical groups / 24 entries)
    # ==========================================================
    print("\n========================================================")
    print("=== EVALUATION ON TUNING PARTITION (Model Selection) ===")
    print("========================================================")
    tuning_results: List[Dict[str, Any]] = []

    for arm_label, ckpt_p, ckpt_data in candidates_to_eval:
        check_deadline(f"Tuning eval for {arm_label} ({ckpt_p.name})")
        print(f"\n[Tuning Eval] Evaluating {arm_label} -> {ckpt_p.name}...")
        eval_report = _eval(
            checkpoint_path=str(ckpt_p),
            fold=0,
            split="tuning",
            method=FIXED_INF_METHOD,
            high_threshold=FIXED_HIGH_THRESH,
            low_threshold=FIXED_LOW_THRESH,
            min_area=FIXED_MIN_AREA,
            max_instances=FIXED_MAX_INSTANCES,
            device_str=device_str,
            strict_cache=True,
        )
        assert_report_inference_config(eval_report)

        t_pq = float(eval_report["overall"]["pq"])
        t_sq = float(eval_report["overall"]["sq"])
        t_rq = float(eval_report["overall"]["rq"])
        t_dice = float(eval_report["overall"]["mean_dice"])
        tp = int(eval_report["overall"]["tp"])
        fp = int(eval_report["overall"]["fp"])
        fn = int(eval_report["overall"]["fn"])
        epoch_num = int(ckpt_data.get("epoch", 1))

        print(f"  Result: PQ={t_pq:.6f} (SQ={t_sq:.6f}, RQ={t_rq:.6f}, Dice={t_dice:.6f}, TP={tp}, FP={fp}, FN={fn})")

        rec = {
            "arm": arm_label,
            "epoch": epoch_num,
            "checkpoint_path": str(ckpt_p),
            "checkpoint_file_sha256": compute_file_sha256(ckpt_p),
            "checkpoint_state_sha256": compute_state_dict_sha256(ckpt_data["model_state_dict"]),
            "tuning_pq": t_pq,
            "tuning_sq": t_sq,
            "tuning_rq": t_rq,
            "tuning_dice": t_dice,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fragmented_gt_count": int(eval_report["overall"].get("fragmented_gt_count", 0)),
            "over_merged_pred_count": int(eval_report["overall"].get("over_merged_pred_count", 0)),
            "missed_gt_count": int(eval_report["overall"].get("missed_gt_count", 0)),
            "spurious_pred_count": int(eval_report["overall"].get("spurious_pred_count", 0)),
            "evaluated_observations": eval_report.get("total_evaluated_observations", 0),
            "evaluated_physical_ids": eval_report.get("evaluated_physical_ids", []),
            "skipped_updates": int(ckpt_data.get("skipped_updates", 0)),
            "successful_updates": int(ckpt_data.get("successful_updates", 0)),
            "eval_report": eval_report,
        }
        tuning_results.append(rec)

    # Ranking with declared ties: highest PQ (-pq), lowest FP (+fp), fewer epochs (+epoch), Arm A before Arm B
    def rank_sort_key(item: Dict[str, Any]):
        arm_pref = 0 if item["arm"] == "Arm_A" else 1
        return (-float(item["tuning_pq"]), int(item["fp"]), int(item["epoch"]), arm_pref)

    ranked_candidates = sorted(tuning_results, key=rank_sort_key)
    winning_rec = ranked_candidates[0]
    winning_arm = winning_rec["arm"]
    winning_checkpoint = Path(winning_rec["checkpoint_path"])
    best_tuning_pq = winning_rec["tuning_pq"]

    print(f"\n[Tuning Winner Selected] {winning_arm} ({winning_checkpoint.name}) with Tuning PQ = {best_tuning_pq:.6f}")

    arm_a_epochs_count = sum(1 for c in candidates_to_eval if c[0] == "Arm_A")
    arm_b_epochs_count = sum(1 for c in candidates_to_eval if c[0] == "Arm_B")

    if is_diagnostic_run:
        print("\n[Diagnostic Run] Storing diagnostic selection outputs in dedicated diagnostic directory.")
        diag_dir = root_dir / "artifacts" / "reports" / "diagnostic"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_selection_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "diagnostic_recovery" if is_recovery else "diagnostic_standard",
            "is_diagnostic": True,
            "eligible_for_promotion": False,
            "winning_arm": winning_arm,
            "winning_checkpoint": str(winning_checkpoint),
            "winning_checkpoint_file_sha256": winning_rec["checkpoint_file_sha256"],
            "best_tuning_pq": best_tuning_pq,
            "all_tuning_evaluations": [
                {k: v for k, v in r.items() if k != "eval_report"}
                for r in ranked_candidates
            ],
        }
        diag_sel_path = diag_dir / "iteration3_diagnostic_selection.json"
        with open(diag_sel_path, "w", encoding="utf-8") as f:
            json.dump(diag_selection_data, f, indent=2)

        total_experiment_time = time.time() - start_time
        diag_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "diagnostic_recovery" if is_recovery else "diagnostic_standard",
            "duration_seconds": total_experiment_time,
            "is_diagnostic": True,
            "eligible_for_promotion": False,
            "inputs": input_provenance,
            "arms": {
                "arm_a_control": {
                    "run_id": arm_a_run_id,
                    "duration_seconds": arm_a_duration,
                    "epochs": arm_a_epochs_count,
                },
                "arm_b_intervention": {
                    "run_id": arm_b_run_id,
                    "duration_seconds": arm_b_duration,
                    "epochs": arm_b_epochs_count,
                },
            },
            "tuning_selection": {
                "winning_arm": winning_arm,
                "winning_checkpoint": str(winning_checkpoint),
                "tuning_pq": best_tuning_pq,
            },
            "gates": {
                "gate1_tuning_target": CANDIDATE_3_GATE1_TUNING_TARGET,
                "gate1_tuning_actual": best_tuning_pq,
                "gate1_passed": False,
                "gate2_conf_floor": CANDIDATE_3_GATE2_CONF_FLOOR,
                "gate2_conf_actual": None,
                "gate2_passed": False,
                "candidate_3_promoted": False,
            },
        }
        diag_sum_path = diag_dir / "iteration3_diagnostic_summary.json"
        with open(diag_sum_path, "w", encoding="utf-8") as f:
            json.dump(diag_summary, f, indent=2)
        print(f"[Diagnostic] Saved diagnostic summary to {diag_sum_path}")
        return diag_summary

    # ==========================================================
    # Production Tuning Selection Persistence
    # ==========================================================
    tuning_selection_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "evaluation_only_recovery" if is_recovery else "standard_training",
        "candidate_2_tuning_pq": CANDIDATE_2_TUNING_PQ,
        "candidate_3_tuning_target": CANDIDATE_3_GATE1_TUNING_TARGET,
        "winning_arm": winning_arm,
        "winning_checkpoint": str(winning_checkpoint),
        "winning_checkpoint_file_sha256": winning_rec["checkpoint_file_sha256"],
        "winning_checkpoint_state_sha256": winning_rec["checkpoint_state_sha256"],
        "best_tuning_pq": best_tuning_pq,
        "all_tuning_evaluations": [
            {k: v for k, v in r.items() if k != "eval_report"}
            for r in ranked_candidates
        ],
    }
    selection_report_path = root_dir / "artifacts" / "reports" / "iteration3_tuning_selection.json"
    selection_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(selection_report_path, "w", encoding="utf-8") as f:
        json.dump(tuning_selection_data, f, indent=2)
    print(f"[Frozen] Saved tuning selection to {selection_report_path}")

    # ==========================================================
    # Single Comparison Partition Evaluation (9 physical groups / 15 entries)
    # ==========================================================
    check_deadline("Comparison evaluation")
    print("\n========================================================")
    print("=== SINGLE EVALUATION ON COMPARISON PARTITION ===")
    print("========================================================")
    print(f"Evaluating winning model {winning_checkpoint.name} on confirmation/comparison split...")
    comp_eval_report = _eval(
        checkpoint_path=str(winning_checkpoint),
        fold=0,
        split="confirmation",
        method=FIXED_INF_METHOD,
        high_threshold=FIXED_HIGH_THRESH,
        low_threshold=FIXED_LOW_THRESH,
        min_area=FIXED_MIN_AREA,
        max_instances=FIXED_MAX_INSTANCES,
        device_str=device_str,
        strict_cache=True,
    )
    assert_report_inference_config(comp_eval_report)

    comp_pq = float(comp_eval_report["overall"]["pq"])
    comp_sq = float(comp_eval_report["overall"]["sq"])
    comp_rq = float(comp_eval_report["overall"]["rq"])
    comp_dice = float(comp_eval_report["overall"]["mean_dice"])
    print(f"[Comparison Result] PQ={comp_pq:.6f} (SQ={comp_sq:.6f}, RQ={comp_rq:.6f}, Dice={comp_dice:.6f})")

    comparison_report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "evaluation_only_recovery" if is_recovery else "standard_training",
        "candidate_2_confirmation_pq": CANDIDATE_2_CONFIRMATION_PQ,
        "candidate_3_conf_floor": CANDIDATE_3_GATE2_CONF_FLOOR,
        "evaluated_checkpoint": str(winning_checkpoint),
        "evaluated_checkpoint_file_sha256": compute_file_sha256(winning_checkpoint),
        "confirmation_pq": comp_pq,
        "confirmation_sq": comp_sq,
        "confirmation_rq": comp_rq,
        "confirmation_dice": comp_dice,
        "full_metrics": comp_eval_report["overall"],
        "evaluated_physical_ids": comp_eval_report.get("evaluated_physical_ids", []),
    }
    comp_report_path = root_dir / "artifacts" / "reports" / "iteration3_comparison_evaluation.json"
    with open(comp_report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report_data, f, indent=2)
    print(f"[Frozen] Saved comparison evaluation to {comp_report_path}")

    # ==========================================================
    # Gate Evaluation & Decision
    # ==========================================================
    print("\n========================================================")
    print("=== ITERATION 3 CANDIDATE 3 GATE AUDIT ===")
    print("========================================================")
    gate1_passed = bool(best_tuning_pq >= CANDIDATE_3_GATE1_TUNING_TARGET)
    gate2_passed = bool(comp_pq >= CANDIDATE_3_GATE2_CONF_FLOOR)
    both_gates_passed = gate1_passed and gate2_passed

    print(f"Gate 1 (Tuning PQ >= {CANDIDATE_3_GATE1_TUNING_TARGET:.6f})   : {best_tuning_pq:.6f} -> {'PASSED' if gate1_passed else 'FAILED'}")
    print(f"Gate 2 (Comparison PQ >= {CANDIDATE_3_GATE2_CONF_FLOOR:.6f}): {comp_pq:.6f} -> {'PASSED' if gate2_passed else 'FAILED'}")
    print(f"Candidate 3 Authorized for Generation: {both_gates_passed}")

    total_experiment_time = time.time() - start_time
    script_source_sha = compute_file_sha256(Path(__file__).resolve())

    summary_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "evaluation_only_recovery" if is_recovery else "standard_training",
        "script_source_sha256": script_source_sha,
        "selector_repair_sha256": script_source_sha,
        "duration_seconds": total_experiment_time,
        "inputs": input_provenance,
        "arms": {
            "arm_a_control": {
                "run_id": arm_a_run_id,
                "duration_seconds": arm_a_duration,
                "epochs": arm_a_epochs_count,
            },
            "arm_b_intervention": {
                "run_id": arm_b_run_id,
                "duration_seconds": arm_b_duration,
                "epochs": arm_b_epochs_count,
            },
        },
        "tuning_selection": {
            "winning_arm": winning_arm,
            "winning_checkpoint": str(winning_checkpoint),
            "tuning_pq": best_tuning_pq,
        },
        "comparison_evaluation": {
            "confirmation_pq": comp_pq,
        },
        "gates": {
            "gate1_tuning_target": CANDIDATE_3_GATE1_TUNING_TARGET,
            "gate1_tuning_actual": best_tuning_pq,
            "gate1_passed": gate1_passed,
            "gate2_conf_floor": CANDIDATE_3_GATE2_CONF_FLOOR,
            "gate2_conf_actual": comp_pq,
            "gate2_passed": gate2_passed,
            "candidate_3_promoted": both_gates_passed,
        },
    }

    summary_path = root_dir / "artifacts" / "reports" / "iteration3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    if both_gates_passed:
        check_deadline("Candidate generation")
        print("\n=== GENERATING AUDITED CANDIDATE 3 SUBMISSION ===")
        from inference import run_inference
        cand3_csv = root_dir / "artifacts" / "submission_candidate_3.csv"
        cand3_man = root_dir / "artifacts" / "submission_candidate_3.manifest.json"
        cand3_cfg = root_dir / "artifacts" / "submission_candidate_3.selection_config.json"

        selection_cfg_data = {
            "candidate_id": 3,
            "selected_checkpoint": str(winning_checkpoint),
            "selected_checkpoint_sha256": compute_file_sha256(winning_checkpoint),
            "inference_config": {
                "method": FIXED_INF_METHOD,
                "high_threshold": FIXED_HIGH_THRESH,
                "low_threshold": FIXED_LOW_THRESH,
                "min_area": FIXED_MIN_AREA,
                "max_instances": FIXED_MAX_INSTANCES,
                "tile_size": FIXED_TILE_SIZE,
                "stride": FIXED_STRIDE,
                "tile_batch_size": FIXED_TILE_BATCH_SIZE,
                "norm_mode": FIXED_NORM_MODE,
                "precision": FIXED_PRECISION,
            },
            "tuning_pq": best_tuning_pq,
            "confirmation_pq": comp_pq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(cand3_cfg, "w", encoding="utf-8") as f:
            json.dump(selection_cfg_data, f, indent=2)

        audit_rep = run_inference(
            checkpoint_path=str(winning_checkpoint),
            test_images_dir=str(root_dir / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"),
            output_csv=str(cand3_csv),
            output_manifest=str(cand3_man),
            method=FIXED_INF_METHOD,
            high_threshold=FIXED_HIGH_THRESH,
            low_threshold=FIXED_LOW_THRESH,
            min_area=FIXED_MIN_AREA,
            max_instances=FIXED_MAX_INSTANCES,
            tile_size=FIXED_TILE_SIZE,
            stride=FIXED_STRIDE,
            tile_batch_size=FIXED_TILE_BATCH_SIZE,
            device_str=device_str,
        )
        summary_report["candidate_3_package"] = {
            "csv_path": str(cand3_csv),
            "csv_sha256": compute_file_sha256(cand3_csv),
            "manifest_path": str(cand3_man),
            "manifest_sha256": compute_file_sha256(cand3_man),
            "selection_config_path": str(cand3_cfg),
            "selection_config_sha256": compute_file_sha256(cand3_cfg),
            "audit_passed": audit_rep["is_valid"],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)
    else:
        print("\n[Notice] Candidate 3 gates not satisfied. Submission candidate NOT generated; preserving Candidate 2.")

    return summary_report


def run_experiment(
    config_path: str = "configs/iteration3_resnet34.yaml",
    parent_checkpoint: str = "artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt",
    mining_bank_path: str = "artifacts/reports/mining_bank_v1.json",
    device_str: Optional[str] = None,
    smoke: bool = False,
    experiment_start_time: Optional[float] = None,
    max_budget_seconds: float = 180 * 60,
    deadline_timestamp: Optional[float | str] = None,
    train_fn: Optional[Any] = None,
    eval_fn: Optional[Any] = None,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete bounded Iteration 3 workflow: Arm A, Arm B, Tuning selection, Comparison eval, and Gates."""
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent
    repo_fallback_root = Path(__file__).resolve().parent.parent

    parent_p = Path(parent_checkpoint) if Path(parent_checkpoint).is_absolute() else (root_dir / parent_checkpoint)
    if not parent_p.is_file() and (repo_fallback_root / parent_checkpoint).is_file():
        parent_p = repo_fallback_root / parent_checkpoint

    manifest_p = root_dir / "artifacts" / "folds_manifest.json"
    if not manifest_p.is_file() and (repo_fallback_root / "artifacts" / "folds_manifest.json").is_file():
        manifest_p = repo_fallback_root / "artifacts" / "folds_manifest.json"

    partitions_p = root_dir / "artifacts" / "partitions_migrated_v1.json"
    if not partitions_p.is_file() and (repo_fallback_root / "artifacts" / "partitions_migrated_v1.json").is_file():
        partitions_p = repo_fallback_root / "artifacts" / "partitions_migrated_v1.json"

    bank_p = (Path(mining_bank_path) if Path(mining_bank_path).is_absolute() else (root_dir / mining_bank_path)) if mining_bank_path else None
    if bank_p and not bank_p.is_file() and (repo_fallback_root / mining_bank_path).is_file():
        bank_p = repo_fallback_root / mining_bank_path

    _train = train_fn if train_fn is not None else train

    start_time = experiment_start_time if experiment_start_time is not None else time.time()
    if deadline_timestamp is not None:
        parsed_dl = parse_deadline_timestamp(deadline_timestamp)
        deadline_time = min(parsed_dl, start_time + max_budget_seconds)
    else:
        deadline_time = start_time + max_budget_seconds

    def check_deadline(stage_name: str, margin: float = CLEANUP_MARGIN_SECONDS) -> None:
        now = time.time()
        if (now + margin) >= deadline_time:
            raise TimeoutError(
                f"Compute budget deadline exceeded before/during {stage_name}: "
                f"now={now:.1f}, deadline={deadline_time:.1f}, margin={margin:.1f}s "
                f"(elapsed={(now - start_time) / 60:.1f}m)."
            )

    input_provenance = verify_immutable_inputs(
        parent_path=parent_p,
        folds_manifest_path=manifest_p,
        partitions_path=partitions_p,
        mining_bank_path=bank_p if (bank_p and bank_p.is_file()) else None,
    )

    epochs = 1 if smoke else 3
    batch_size = 2

    # Check budget before Arm A
    check_deadline("Arm A start")

    train_sig = inspect.signature(_train)
    train_extra_kwargs = {}
    if "deadline_time" in train_sig.parameters:
        train_extra_kwargs["deadline_time"] = deadline_time - CLEANUP_MARGIN_SECONDS

    # ==========================================================
    # Arm A: Control (Equal update baseline, no flips, standard crop)
    # ==========================================================
    print("\n========================================================")
    print("=== ARM A: CONTROL FINE-TUNING (Standard Crops, No Flips) ===")
    print("========================================================")
    arm_a_start = time.time()
    arm_a_latest, arm_a_best = _train(
        config_path=config_path,
        fold=0,
        smoke=smoke,
        epochs_override=epochs,
        batch_size_override=batch_size,
        device_str=device_str,
        finetune_from=str(parent_p),
        mining_bank_path=None,
        augment_flips=False,
        **train_extra_kwargs,
    )
    arm_a_duration = time.time() - arm_a_start
    print(f"[Arm A] Completed in {arm_a_duration:.1f}s ({arm_a_duration / 60:.1f}m). Best: {arm_a_best}")

    # Check budget after Arm A
    check_deadline("Arm A completion")
    estimated_needed = arm_a_duration * 1.3
    if (time.time() - start_time + estimated_needed) > (deadline_time - start_time):
        raise TimeoutError(
            f"Cannot fit remaining work ({estimated_needed/60:.1f}m) within remaining budget. Stopping safely."
        )

    # Immediately locate and validate all saved epoch checkpoints for Arm A
    arm_a_ckpt = torch.load(str(arm_a_latest), map_location="cpu", weights_only=False)
    arm_a_run_id = arm_a_ckpt.get("run_id")
    if not arm_a_run_id:
        raise ValueError("Arm A latest checkpoint missing run_id")

    expected_epochs = 1 if smoke else epochs
    arm_a_candidates = resolve_and_validate_arm_checkpoints(
        arm_label="Arm_A",
        run_id=arm_a_run_id,
        root_dir=root_dir,
        expected_epochs=expected_epochs,
        smoke=smoke,
        expected_parent_sha=input_provenance["parent_file_sha256"],
        expected_folds_manifest_sha=input_provenance["folds_manifest_sha256"],
        expected_mining_bank_sha=input_provenance["mining_bank_sha256"],
        fallback_parent_dir=Path(arm_a_latest).parent,
    )
    print(f"[Arm A] Validated {len(arm_a_candidates)} immutable epoch checkpoints for {arm_a_run_id}")

    # ==========================================================
    # Arm B: Intervention (Mining bank crops + Flip augmentation)
    # ==========================================================
    print("\n========================================================")
    print("=== ARM B: INTERVENTION FINE-TUNING (Mining Bank + Flips) ===")
    print("========================================================")
    check_deadline("Arm B start")
    arm_b_start = time.time()
    arm_b_latest, arm_b_best = _train(
        config_path=config_path,
        fold=0,
        smoke=smoke,
        epochs_override=epochs,
        batch_size_override=batch_size,
        device_str=device_str,
        finetune_from=str(parent_p),
        mining_bank_path=str(bank_p) if bank_p else None,
        augment_flips=True,
        **train_extra_kwargs,
    )
    arm_b_duration = time.time() - arm_b_start
    print(f"[Arm B] Completed in {arm_b_duration:.1f}s ({arm_b_duration / 60:.1f}m). Best: {arm_b_best}")

    # Check budget after Arm B
    check_deadline("Arm B completion")

    # Immediately locate and validate all saved epoch checkpoints for Arm B
    arm_b_ckpt = torch.load(str(arm_b_latest), map_location="cpu", weights_only=False)
    arm_b_run_id = arm_b_ckpt.get("run_id")
    if not arm_b_run_id:
        raise ValueError("Arm B latest checkpoint missing run_id")
    if arm_b_run_id == arm_a_run_id:
        raise ValueError(f"Arm B run_id cannot be identical to Arm A run_id: {arm_b_run_id}")

    arm_b_candidates = resolve_and_validate_arm_checkpoints(
        arm_label="Arm_B",
        run_id=arm_b_run_id,
        root_dir=root_dir,
        expected_epochs=expected_epochs,
        smoke=smoke,
        expected_parent_sha=input_provenance["parent_file_sha256"],
        expected_folds_manifest_sha=input_provenance["folds_manifest_sha256"],
        expected_mining_bank_sha=input_provenance["mining_bank_sha256"],
        fallback_parent_dir=Path(arm_b_latest).parent,
    )
    print(f"[Arm B] Validated {len(arm_b_candidates)} immutable epoch checkpoints for {arm_b_run_id}")

    # Cross-arm update count parity verification
    for e in range(1, expected_epochs + 1):
        u_a = int(arm_a_candidates[e - 1][2]["successful_updates"])
        u_b = int(arm_b_candidates[e - 1][2]["successful_updates"])
        if u_a != u_b:
            raise ValueError(f"Unequal successful updates at epoch {e}: Arm A has {u_a}, Arm B has {u_b}")

    candidates_to_eval = arm_a_candidates + arm_b_candidates

    return evaluate_and_select_candidates(
        candidates_to_eval=candidates_to_eval,
        input_provenance=input_provenance,
        arm_a_run_id=arm_a_run_id,
        arm_b_run_id=arm_b_run_id,
        arm_a_duration=arm_a_duration,
        arm_b_duration=arm_b_duration,
        start_time=start_time,
        deadline_time=deadline_time,
        device_str=device_str,
        smoke=smoke,
        is_recovery=False,
        eval_fn=eval_fn,
        root_dir=root_dir,
    )


def recover_and_evaluate(
    arm_a_run_id: str,
    arm_b_run_id: str,
    deadline_timestamp: float | str,
    config_path: str = "configs/iteration3_resnet34.yaml",
    parent_checkpoint: str = "artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt",
    mining_bank_path: str = "artifacts/reports/mining_bank_v1.json",
    device_str: Optional[str] = None,
    smoke: bool = False,
    arm_a_duration: float = 871.9,
    arm_b_duration: Optional[float] = None,
    eval_fn: Optional[Any] = None,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute evaluation-only recovery for completed Arm A and Arm B training runs without any training invocation."""
    if not arm_a_run_id or not arm_b_run_id:
        raise ValueError("Both arm_a_run_id and arm_b_run_id must be provided for recovery evaluation.")

    if arm_a_run_id == arm_b_run_id:
        raise ValueError(f"Arm A and Arm B run IDs cannot be identical: {arm_a_run_id}")

    if deadline_timestamp is None:
        raise ValueError("Evaluation recovery mode requires mandatory deadline_timestamp: Missing required deadline timestamp.")

    deadline_time = parse_deadline_timestamp(deadline_timestamp)
    start_time = time.time()
    if (start_time + CLEANUP_MARGIN_SECONDS) >= deadline_time:
        raise TimeoutError(
            f"Supplied deadline {deadline_time} has expired or leaves inadequate cleanup margin "
            f"(now={start_time:.1f}, margin={CLEANUP_MARGIN_SECONDS}s)."
        )

    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent
    repo_fallback_root = Path(__file__).resolve().parent.parent

    parent_p = Path(parent_checkpoint) if Path(parent_checkpoint).is_absolute() else (root_dir / parent_checkpoint)
    if not parent_p.is_file() and (repo_fallback_root / parent_checkpoint).is_file():
        parent_p = repo_fallback_root / parent_checkpoint

    manifest_p = root_dir / "artifacts" / "folds_manifest.json"
    if not manifest_p.is_file() and (repo_fallback_root / "artifacts" / "folds_manifest.json").is_file():
        manifest_p = repo_fallback_root / "artifacts" / "folds_manifest.json"

    partitions_p = root_dir / "artifacts" / "partitions_migrated_v1.json"
    if not partitions_p.is_file() and (repo_fallback_root / "artifacts" / "partitions_migrated_v1.json").is_file():
        partitions_p = repo_fallback_root / "artifacts" / "partitions_migrated_v1.json"

    if not mining_bank_path:
        raise FileNotFoundError("Required mining bank path not supplied.")
    bank_p = Path(mining_bank_path) if Path(mining_bank_path).is_absolute() else (root_dir / mining_bank_path)
    if not bank_p.is_file() and (repo_fallback_root / mining_bank_path).is_file():
        bank_p = repo_fallback_root / mining_bank_path
    if not bank_p.is_file():
        raise FileNotFoundError(f"Required mining bank missing at supplied path: {bank_p}")

    # Step 0: Cryptographic provenance verification of immutable inputs
    input_provenance = verify_immutable_inputs(
        parent_path=parent_p,
        folds_manifest_path=manifest_p,
        partitions_path=partitions_p,
        mining_bank_path=bank_p,
    )

    expected_epochs = 1 if smoke else 3

    print("\n========================================================")
    print("=== RECOVERY EVALUATION: DISCOVERING ARM A CHECKPOINTS ===")
    print("========================================================")
    arm_a_candidates = resolve_and_validate_arm_checkpoints(
        arm_label="Arm_A",
        run_id=arm_a_run_id,
        root_dir=root_dir,
        expected_epochs=expected_epochs,
        smoke=smoke,
        expected_parent_sha=input_provenance["parent_file_sha256"],
        expected_folds_manifest_sha=input_provenance["folds_manifest_sha256"],
        expected_mining_bank_sha=input_provenance["mining_bank_sha256"],
    )
    print(f"[Recovery Arm A] Validated {len(arm_a_candidates)} immutable checkpoints for {arm_a_run_id}")

    print("\n========================================================")
    print("=== RECOVERY EVALUATION: DISCOVERING ARM B CHECKPOINTS ===")
    print("========================================================")
    arm_b_candidates = resolve_and_validate_arm_checkpoints(
        arm_label="Arm_B",
        run_id=arm_b_run_id,
        root_dir=root_dir,
        expected_epochs=expected_epochs,
        smoke=smoke,
        expected_parent_sha=input_provenance["parent_file_sha256"],
        expected_folds_manifest_sha=input_provenance["folds_manifest_sha256"],
        expected_mining_bank_sha=input_provenance["mining_bank_sha256"],
    )
    print(f"[Recovery Arm B] Validated {len(arm_b_candidates)} immutable checkpoints for {arm_b_run_id}")

    # Cross-arm update count parity verification
    for e in range(1, expected_epochs + 1):
        u_a = int(arm_a_candidates[e - 1][2]["successful_updates"])
        u_b = int(arm_b_candidates[e - 1][2]["successful_updates"])
        if u_a != u_b:
            raise ValueError(f"Unequal successful updates at epoch {e}: Arm A has {u_a}, Arm B has {u_b}")

    candidates_to_eval = arm_a_candidates + arm_b_candidates

    # Validate exact disjoint physical training/validation membership against frozen manifest assignments
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest_assignments = manifest_data.get("assignments", {})
    expected_val_obs = set(k for k, v in manifest_assignments.items() if "-" not in k and v == 0)
    expected_train_obs = set(k for k, v in manifest_assignments.items() if "-" not in k and v != 0)

    for arm_lbl, p, ckpt_data in candidates_to_eval:
        ckpt_train = set(ckpt_data.get("train_observations", []))
        ckpt_val = set(ckpt_data.get("val_observations", []))
        if not ckpt_train or not ckpt_val:
            raise ValueError(f"[{arm_lbl}] Checkpoint {p.name} missing train_observations or val_observations")
        if len(ckpt_train & ckpt_val) > 0:
            raise ValueError(f"[{arm_lbl}] Checkpoint {p.name} has non-disjoint train and val observations")
        if not smoke:
            if len(ckpt_train) != 565 or len(ckpt_val) != 142:
                raise ValueError(
                    f"[{arm_lbl}] Checkpoint {p.name} observation counts mismatch: "
                    f"expected 565 train and 142 val, got {len(ckpt_train)} and {len(ckpt_val)}"
                )
            if ckpt_val != expected_val_obs:
                raise ValueError(
                    f"[{arm_lbl}] Checkpoint {p.name} val_observations do not match frozen fold 0 assignments in manifest"
                )
            if ckpt_train != expected_train_obs:
                raise ValueError(
                    f"[{arm_lbl}] Checkpoint {p.name} train_observations do not match frozen non-fold-0 assignments in manifest"
                )

    return evaluate_and_select_candidates(
        candidates_to_eval=candidates_to_eval,
        input_provenance=input_provenance,
        arm_a_run_id=arm_a_run_id,
        arm_b_run_id=arm_b_run_id,
        arm_a_duration=arm_a_duration,
        arm_b_duration=arm_b_duration if arm_b_duration is not None else arm_a_duration,
        start_time=start_time,
        deadline_time=deadline_time,
        device_str=device_str,
        smoke=smoke,
        is_recovery=True,
        eval_fn=eval_fn,
        root_dir=root_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate Iteration 3 Fine-Tuning and Evaluation")
    parser.add_argument("--config", type=str, default="configs/iteration3_resnet34.yaml")
    parser.add_argument("--parent", type=str, default="artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    parser.add_argument("--mining-bank", type=str, default="artifacts/reports/mining_bank_v1.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--budget-seconds", "--max-budget-seconds", type=float, default=180 * 60, dest="max_budget_seconds")
    parser.add_argument("--deadline-timestamp", type=str, default=None)

    # Recovery mode arguments
    parser.add_argument("--recover-eval", action="store_true", help="Run evaluation-only recovery on completed arms without training")
    parser.add_argument("--arm-a-run-id", type=str, default=None, help="Explicit immutable run ID for Arm A control")
    parser.add_argument("--arm-b-run-id", type=str, default=None, help="Explicit immutable run ID for Arm B intervention")
    parser.add_argument("--arm-a-duration", type=float, default=871.9, help="Original Arm A training duration in seconds")
    parser.add_argument("--arm-b-duration", type=float, default=None, help="Original Arm B training duration in seconds")

    args = parser.parse_args()

    if args.recover_eval:
        if not args.arm_a_run_id or not args.arm_b_run_id:
            parser.error("--recover-eval requires both --arm-a-run-id and --arm-b-run-id")
        if not args.deadline_timestamp:
            parser.error("--recover-eval requires --deadline-timestamp")
        recover_and_evaluate(
            arm_a_run_id=args.arm_a_run_id,
            arm_b_run_id=args.arm_b_run_id,
            deadline_timestamp=args.deadline_timestamp,
            config_path=args.config,
            parent_checkpoint=args.parent,
            mining_bank_path=args.mining_bank,
            device_str=args.device,
            smoke=args.smoke,
            arm_a_duration=args.arm_a_duration,
            arm_b_duration=args.arm_b_duration,
        )
    else:
        run_experiment(
            config_path=args.config,
            parent_checkpoint=args.parent,
            mining_bank_path=args.mining_bank,
            device_str=args.device,
            smoke=args.smoke,
            max_budget_seconds=args.max_budget_seconds,
            deadline_timestamp=args.deadline_timestamp,
        )
