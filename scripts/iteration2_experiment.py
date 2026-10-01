from datetime import datetime, timezone
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluate import evaluate_oof
from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.dataset import load_solar_image
from src.data.folds import canonical_observation_id
from src.inference.config import resolve_inference_config
from src.inference.engine import (
    compute_file_sha256,
    compute_state_dict_sha256,
    predict_full_observation,
)
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import audit_submission_and_manifest
from src.models import build_model


def run_iteration2():
    start_total_time = time.time()
    ckpt_path = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    partitions_path = Path("artifacts/partitions_migrated_v1.json")
    cache_dir = Path("artifacts/cache_iteration2_fp32")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not partitions_path.is_file():
        raise FileNotFoundError(f"Partitions file not found: {partitions_path}")

    ckpt_file_sha = compute_file_sha256(ckpt_path)
    partitions_sha = compute_file_sha256(partitions_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ITERATION 2 CALIBRATION EXPERIMENT ===")
    print(f"Checkpoint: {ckpt_path.name} (SHA256: {ckpt_file_sha})")
    print(f"Partitions: {partitions_path.name} (SHA256: {partitions_sha})")
    print(f"Device: {device}")
    print(f"Cache dir: {cache_dir} (Versioned FP32)")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd_sha = compute_state_dict_sha256(ckpt["model_state_dict"])
    training_config = ckpt.get("config", {})
    training_cfg_sha = hashlib.sha256(json.dumps(training_config, sort_keys=True).encode("utf-8")).hexdigest()
    print(f"Model State Dict SHA256: {sd_sha}")

    # =========================================================================
    # STEP 1: Baseline Reproduction & FP32 Map Caching on Tuning
    # =========================================================================
    print("\n--- STEP 1: Populate FP32 Cache & Reproduce Baseline on Tuning (10 physical groups) ---")
    base_overrides = {
        "method": "connected_components",
        "high_threshold": 0.85,
        "low_threshold": 0.60,
        "min_area": 400,
        "max_instances": 12,
        "tile_size": 512,
        "stride": 256,
        "norm_mode": "imagenet",
        "precision": "float32",
    }
    
    t0 = time.time()
    baseline_report = evaluate_oof(
        checkpoint_path=str(ckpt_path),
        fold=0,
        split="tuning",
        method=base_overrides["method"],
        high_threshold=base_overrides["high_threshold"],
        low_threshold=base_overrides["low_threshold"],
        min_area=base_overrides["min_area"],
        max_instances=base_overrides["max_instances"],
        tile_size=base_overrides["tile_size"],
        stride=base_overrides["stride"],
        cache_dir=str(cache_dir),
        device_str=str(device),
    )
    t_base = time.time() - t0
    base_ov = baseline_report["overall"]
    print(f"Baseline Tuning Reproduction ({t_base:.1f}s):")
    print(f"  PQ:   {base_ov['pq']:.16f}")
    print(f"  SQ:   {base_ov['sq']:.16f}, RQ: {base_ov['rq']:.16f}")
    print(f"  Dice: {base_ov['mean_dice']:.10f}")
    print(f"  TP: {base_ov['tp']}, FP: {base_ov['fp']}, FN: {base_ov['fn']}")
    print(f"  Frag: {base_ov['fragmented_gt_count']}, Miss: {base_ov['missed_gt_count']}, Spurious: {base_ov['spurious_pred_count']}")

    expected_baseline_pq = 0.30694299508610323
    assert abs(base_ov["pq"] - expected_baseline_pq) < 1e-5, (
        f"Baseline tuning PQ mismatch! Expected ~{expected_baseline_pq}, got {base_ov['pq']}"
    )
    assert base_ov["tp"] == 100 and base_ov["fp"] == 142 and base_ov["fn"] == 91, (
        f"Baseline counts mismatch! Expected TP 100 / FP 142 / FN 91, got TP {base_ov['tp']} / FP {base_ov['fp']} / FN {base_ov['fn']}"
    )
    print("Baseline tuning reproduction verified exactly!")

    # =========================================================================
    # STEP 2: Stage A Calibration - Cartesian Product (min_area x max_instances)
    # =========================================================================
    print("\n--- STEP 2: Stage A Calibration (6 settings) on Cached FP32 Maps ---")
    stage_a_grid = [
        {"min_area": 400, "max_instances": 12, "id": "A1_area400_cap12 (Baseline)"},
        {"min_area": 400, "max_instances": 20, "id": "A2_area400_cap20"},
        {"min_area": 400, "max_instances": None, "id": "A3_area400_capNone"},
        {"min_area": 200, "max_instances": 12, "id": "A4_area200_cap12"},
        {"min_area": 200, "max_instances": 20, "id": "A5_area200_cap20"},
        {"min_area": 200, "max_instances": None, "id": "A6_area200_capNone"},
    ]

    stage_a_cache_path = Path("artifacts/reports/iteration2_stage_a_cache.json")
    if stage_a_cache_path.is_file():
        print(f"Loading cached Stage A results from {stage_a_cache_path}...")
        with open(stage_a_cache_path, "r", encoding="utf-8") as f:
            stage_a_results = json.load(f)
        for s in stage_a_results:
            print(f"  [cached] {s['setting_id']:<26} | PQ: {s['pq']:.4f} (dPQ: {s['delta_pq_vs_baseline']:+.4f}) | TP: {s['tp']} (dTP: {s['delta_tp_vs_baseline']:+d}) | FP: {s['fp']} (dFP: {s['delta_fp_vs_baseline']:+d}) | Dice: {s['mean_dice']:.4f}")
    else:
        stage_a_results = []
        for s_idx, cfg in enumerate(stage_a_grid, start=1):
            t_start = time.time()
            rep = evaluate_oof(
                checkpoint_path=str(ckpt_path),
                fold=0,
                split="tuning",
                method="connected_components",
                high_threshold=0.85,
                low_threshold=0.60,
                min_area=cfg["min_area"],
                max_instances=cfg["max_instances"],
                tile_size=512,
                stride=256,
                cache_dir=str(cache_dir),
                device_str=str(device),
            )
            t_eval = time.time() - t_start
            ov = rep["overall"]
            
            # Check how many observations hit cap 12
            obs_hit_cap12 = [o["observation_id"] for o in rep["per_observation"] if o["predicted_instances"] >= 12]

            res_entry = {
                "setting_index": s_idx,
                "setting_id": cfg["id"],
                "stage": "A",
                "high_threshold": 0.85,
                "low_threshold": 0.60,
                "min_area": cfg["min_area"],
                "max_instances": cfg["max_instances"],
                "pq": ov["pq"],
                "sq": ov["sq"],
                "rq": ov["rq"],
                "mean_dice": ov["mean_dice"],
                "tp": ov["tp"],
                "fp": ov["fp"],
                "fn": ov["fn"],
                "fragmented_gt_count": ov["fragmented_gt_count"],
                "over_merged_pred_count": ov["over_merged_pred_count"],
                "missed_gt_count": ov["missed_gt_count"],
                "spurious_pred_count": ov["spurious_pred_count"],
                "delta_tp_vs_baseline": ov["tp"] - base_ov["tp"],
                "delta_fp_vs_baseline": ov["fp"] - base_ov["fp"],
                "delta_pq_vs_baseline": ov["pq"] - base_ov["pq"],
                "obs_hitting_cap12_count": len(obs_hit_cap12),
                "obs_hitting_cap12_ids": obs_hit_cap12,
                "per_observation": rep["per_observation"],
                "eval_seconds": round(t_eval, 3),
            }
            stage_a_results.append(res_entry)
            print(f"  [{s_idx}/6] {cfg['id']:<26} | PQ: {ov['pq']:.4f} (dPQ: {res_entry['delta_pq_vs_baseline']:+.4f}) | TP: {ov['tp']} (dTP: {res_entry['delta_tp_vs_baseline']:+d}) | FP: {ov['fp']} (dFP: {res_entry['delta_fp_vs_baseline']:+d}) | Dice: {ov['mean_dice']:.4f} ({t_eval:.2f}s)")
        
        with open(stage_a_cache_path, "w", encoding="utf-8") as f:
            json.dump(stage_a_results, f, indent=2)

    # Sort Stage A by: highest PQ, then lower FP, then baseline preference
    def stage_a_sort_key(x):
        is_baseline = (x["min_area"] == 400 and x["max_instances"] == 12)
        return (-x["pq"], x["fp"], 0 if is_baseline else 1)

    stage_a_sorted = sorted(stage_a_results, key=stage_a_sort_key)
    winner_a = stage_a_sorted[0]
    print(f"\nStage A Winner: {winner_a['setting_id']}")
    print(f"  min_area={winner_a['min_area']}, max_instances={winner_a['max_instances']}")
    print(f"  PQ={winner_a['pq']:.4f}, TP={winner_a['tp']}, FP={winner_a['fp']}, FN={winner_a['fn']}")

    # =========================================================================
    # STEP 3: Stage B Calibration - Threshold Pairs with Winning Stage A Area/Cap
    # =========================================================================
    print(f"\n--- STEP 3: Stage B Calibration (4 threshold pairs with area={winner_a['min_area']}, cap={winner_a['max_instances']}) ---")
    threshold_pairs = [
        {"high": 0.80, "low": 0.60, "id": f"B1_h0.80_l0.60"},
        {"high": 0.90, "low": 0.60, "id": f"B2_h0.90_l0.60"},
        {"high": 0.85, "low": 0.50, "id": f"B3_h0.85_l0.50"},
        {"high": 0.85, "low": 0.70, "id": f"B4_h0.85_l0.70"},
    ]

    stage_b_cache_path = Path("artifacts/reports/iteration2_stage_b_cache.json")
    if stage_b_cache_path.is_file():
        print(f"Loading cached Stage B results from {stage_b_cache_path}...")
        with open(stage_b_cache_path, "r", encoding="utf-8") as f:
            stage_b_results = json.load(f)
        for s in stage_b_results:
            print(f"  [cached] {s['setting_id']:<26} | PQ: {s['pq']:.4f} (dPQ: {s['delta_pq_vs_baseline']:+.4f}) | TP: {s['tp']} (dTP: {s['delta_tp_vs_baseline']:+d}) | FP: {s['fp']} (dFP: {s['delta_fp_vs_baseline']:+d}) | Dice: {s['mean_dice']:.4f}")
    else:
        stage_b_results = []
        for b_idx, tp in enumerate(threshold_pairs, start=1):
            t_start = time.time()
            rep = evaluate_oof(
                checkpoint_path=str(ckpt_path),
                fold=0,
                split="tuning",
                method="connected_components",
                high_threshold=tp["high"],
                low_threshold=tp["low"],
                min_area=winner_a["min_area"],
                max_instances=winner_a["max_instances"],
                tile_size=512,
                stride=256,
                cache_dir=str(cache_dir),
                device_str=str(device),
            )
            t_eval = time.time() - t_start
            ov = rep["overall"]
            obs_hit_cap12 = [o["observation_id"] for o in rep["per_observation"] if o["predicted_instances"] >= 12]

            res_entry = {
                "setting_index": 6 + b_idx,
                "setting_id": tp["id"],
                "stage": "B",
                "high_threshold": tp["high"],
                "low_threshold": tp["low"],
                "min_area": winner_a["min_area"],
                "max_instances": winner_a["max_instances"],
                "pq": ov["pq"],
                "sq": ov["sq"],
                "rq": ov["rq"],
                "mean_dice": ov["mean_dice"],
                "tp": ov["tp"],
                "fp": ov["fp"],
                "fn": ov["fn"],
                "fragmented_gt_count": ov["fragmented_gt_count"],
                "over_merged_pred_count": ov["over_merged_pred_count"],
                "missed_gt_count": ov["missed_gt_count"],
                "spurious_pred_count": ov["spurious_pred_count"],
                "delta_tp_vs_baseline": ov["tp"] - base_ov["tp"],
                "delta_fp_vs_baseline": ov["fp"] - base_ov["fp"],
                "delta_pq_vs_baseline": ov["pq"] - base_ov["pq"],
                "obs_hitting_cap12_count": len(obs_hit_cap12),
                "obs_hitting_cap12_ids": obs_hit_cap12,
                "per_observation": rep["per_observation"],
                "eval_seconds": round(t_eval, 3),
            }
            stage_b_results.append(res_entry)
            print(f"  [{b_idx}/4] {tp['id']:<26} | PQ: {ov['pq']:.4f} (dPQ: {res_entry['delta_pq_vs_baseline']:+.4f}) | TP: {ov['tp']} (dTP: {res_entry['delta_tp_vs_baseline']:+d}) | FP: {ov['fp']} (dFP: {res_entry['delta_fp_vs_baseline']:+d}) | Dice: {ov['mean_dice']:.4f} ({t_eval:.2f}s)")
        
        with open(stage_b_cache_path, "w", encoding="utf-8") as f:
            json.dump(stage_b_results, f, indent=2)

    # Combine all 10 settings
    all_10_settings = stage_a_results + stage_b_results

    # Overall selection rule:
    # 1. highest PQ
    # 2. lower FP
    # 3. baseline preference if tied with baseline
    # 4. lexicographically first (high, low, area, cap with None ordered last)
    def overall_sort_key(x):
        is_baseline = (
            x["high_threshold"] == 0.85
            and x["low_threshold"] == 0.60
            and x["min_area"] == 400
            and x["max_instances"] == 12
        )
        cap_val = float("inf") if x["max_instances"] is None else x["max_instances"]
        return (
            -x["pq"],
            x["fp"],
            0 if is_baseline else 1,
            x["high_threshold"],
            x["low_threshold"],
            x["min_area"],
            cap_val,
        )

    all_sorted = sorted(all_10_settings, key=overall_sort_key)
    overall_winner = all_sorted[0]

    print("\n=======================================================")
    print("ALL 10 SETTINGS RANKING (TUNING PARTITION):")
    for rank, s in enumerate(all_sorted, start=1):
        cap_disp = str(s["max_instances"]) if s["max_instances"] is not None else "None"
        print(f"  Rank {rank:2d}: {s['setting_id']:<28} | high={s['high_threshold']:.2f}, low={s['low_threshold']:.2f}, area={s['min_area']}, cap={cap_disp:<4} | PQ: {s['pq']:.4f} (dPQ: {s['delta_pq_vs_baseline']:+.4f}) | TP: {s['tp']}, FP: {s['fp']}, FN: {s['fn']}")
    print(f"\nOVERALL WINNING SETTING: {overall_winner['setting_id']}")
    print(f"  high={overall_winner['high_threshold']}, low={overall_winner['low_threshold']}, area={overall_winner['min_area']}, cap={overall_winner['max_instances']}")
    print(f"  Tuning PQ: {overall_winner['pq']:.6f} (Baseline: {base_ov['pq']:.6f}, Delta: {overall_winner['delta_pq_vs_baseline']:+.6f})")
    print("=======================================================\n")

    # =========================================================================
    # STEP 4: Freeze Machine-Readable Tuning Selection Record BEFORE Comparison
    # =========================================================================
    winning_resolved_inf_cfg = resolve_inference_config(
        training_config,
        overrides={
            "method": "connected_components",
            "high_threshold": overall_winner["high_threshold"],
            "low_threshold": overall_winner["low_threshold"],
            "min_area": overall_winner["min_area"],
            "max_instances": overall_winner["max_instances"],
            "tile_size": 512,
            "stride": 256,
            "norm_mode": "imagenet",
            "precision": "float32",
        },
    )
    winning_resolved_cfg_sha = hashlib.sha256(json.dumps(winning_resolved_inf_cfg, sort_keys=True).encode("utf-8")).hexdigest()

    # Exact annotator entry IDs
    from src.data.folds import load_frozen_folds_manifest
    from src.data.annotations import load_coco_annotations
    train_json = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = load_coco_annotations(str(train_json))
    fold_assignments, _ = load_frozen_folds_manifest("artifacts/folds_manifest.json")
    with open(partitions_path, "r", encoding="utf-8") as f:
        migrated_parts = json.load(f)
    tuning_physical_ids = migrated_parts["tuning"]["canonical_observation_ids"]
    tuning_entry_ids = []
    for entry in ann_index.by_annotator_image.values():
        c_obs = canonical_observation_id(entry.observation_id)
        if c_obs in tuning_physical_ids:
            tuning_entry_ids.append(entry.annotator_image_id)
    tuning_entry_ids = sorted(list(set(tuning_entry_ids)))

    selection_record = {
        "record_version": "2.0.0",
        "description": "Iteration 2 declared filter calibration selection record frozen BEFORE comparison scoring",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "epoch": 6,
            "filename": ckpt_path.name,
            "path": str(ckpt_path),
            "file_sha256": ckpt_file_sha,
            "model_state_dict_sha256": sd_sha,
            "training_config_sha256": training_cfg_sha,
        },
        "partitions": {
            "partitions_file": partitions_path.name,
            "partitions_sha256": partitions_sha,
            "tuning_physical_ids": tuning_physical_ids,
            "tuning_entry_ids": tuning_entry_ids,
            "tuning_entries_count": len(tuning_entry_ids),
        },
        "selection_rule": (
            "Cartesian Stage A (min_area in {200, 400}, max_instances in {12, 20, None}), "
            "then Stage B (threshold pairs {(0.80,0.60), (0.90,0.60), (0.85,0.50), (0.85,0.70)} "
            "with winning Stage A area/cap). Rank by highest strict tuning PQ, then lower FP, "
            "then baseline preference if tied, then lexicographically first."
        ),
        "baseline_tuning": {
            "pq": base_ov["pq"],
            "sq": base_ov["sq"],
            "rq": base_ov["rq"],
            "mean_dice": base_ov["mean_dice"],
            "tp": base_ov["tp"],
            "fp": base_ov["fp"],
            "fn": base_ov["fn"],
        },
        "stage_a_results": stage_a_results,
        "stage_b_results": stage_b_results,
        "all_ranked_settings": all_sorted,
        "winning_setting": {
            "setting_id": overall_winner["setting_id"],
            "high_threshold": overall_winner["high_threshold"],
            "low_threshold": overall_winner["low_threshold"],
            "min_area": overall_winner["min_area"],
            "max_instances": overall_winner["max_instances"],
            "pq": overall_winner["pq"],
            "sq": overall_winner["sq"],
            "rq": overall_winner["rq"],
            "mean_dice": overall_winner["mean_dice"],
            "tp": overall_winner["tp"],
            "fp": overall_winner["fp"],
            "fn": overall_winner["fn"],
            "fragmented_gt_count": overall_winner["fragmented_gt_count"],
            "over_merged_pred_count": overall_winner["over_merged_pred_count"],
            "missed_gt_count": overall_winner["missed_gt_count"],
            "spurious_pred_count": overall_winner["spurious_pred_count"],
            "delta_pq_vs_baseline": overall_winner["delta_pq_vs_baseline"],
            "delta_tp_vs_baseline": overall_winner["delta_tp_vs_baseline"],
            "delta_fp_vs_baseline": overall_winner["delta_fp_vs_baseline"],
        },
        "winning_resolved_inference_config": winning_resolved_inf_cfg,
        "winning_resolved_inference_config_sha256": winning_resolved_cfg_sha,
    }

    selection_record_path = Path("artifacts/reports/iteration2_tuning_selection.json")
    with open(selection_record_path, "w", encoding="utf-8") as f:
        json.dump(selection_record, f, indent=2)
    print(f"Frozen iteration 2 selection record written to: {selection_record_path}")

    # =========================================================================
    # STEP 5: Comparison Scope Evaluation (9 unique physical groups, 15 entries)
    # =========================================================================
    print("\n--- STEP 5: Evaluate Frozen Winning Setting on CONFIRMATION (9 unique physical groups) ---")
    t0_conf = time.time()
    conf_report = evaluate_oof(
        checkpoint_path=str(ckpt_path),
        fold=0,
        split="confirmation",
        method="connected_components",
        high_threshold=overall_winner["high_threshold"],
        low_threshold=overall_winner["low_threshold"],
        min_area=overall_winner["min_area"],
        max_instances=overall_winner["max_instances"],
        tile_size=512,
        stride=256,
        cache_dir=str(cache_dir),
        device_str=str(device),
    )
    t_conf = time.time() - t0_conf
    conf_ov = conf_report["overall"]

    baseline_conf_pq = 0.2993563566276471
    baseline_conf_tp = 57
    baseline_conf_fp = 68
    baseline_conf_fn = 73
    baseline_conf_dice = 0.5947294166436193

    print(f"Confirmation Evaluation ({t_conf:.1f}s):")
    print(f"  PQ:   {conf_ov['pq']:.16f} (Baseline: {baseline_conf_pq:.16f}, Delta: {conf_ov['pq'] - baseline_conf_pq:+.6f})")
    print(f"  SQ:   {conf_ov['sq']:.16f}, RQ: {conf_ov['rq']:.16f}")
    print(f"  Dice: {conf_ov['mean_dice']:.10f} (Baseline: {baseline_conf_dice:.10f})")
    print(f"  TP: {conf_ov['tp']} (Baseline: {baseline_conf_tp}), FP: {conf_ov['fp']} (Baseline: {baseline_conf_fp}), FN: {conf_ov['fn']} (Baseline: {baseline_conf_fn})")
    print(f"  Frag: {conf_ov['fragmented_gt_count']}, Miss: {conf_ov['missed_gt_count']}, Spurious: {conf_ov['spurious_pred_count']}")

    conf_summary_path = Path("artifacts/reports/iteration2_comparison_evaluation.json")
    conf_summary = {
        "scope_disclosure": "previously inspected confirmation partition with 9 unique physical groups and 15 annotator entries",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "winning_setting": overall_winner["setting_id"],
        "resolved_inference_config": winning_resolved_inf_cfg,
        "metrics": {
            "pq": conf_ov["pq"],
            "sq": conf_ov["sq"],
            "rq": conf_ov["rq"],
            "mean_dice": conf_ov["mean_dice"],
            "tp": conf_ov["tp"],
            "fp": conf_ov["fp"],
            "fn": conf_ov["fn"],
            "fragmented_gt_count": conf_ov["fragmented_gt_count"],
            "over_merged_pred_count": conf_ov["over_merged_pred_count"],
            "missed_gt_count": conf_ov["missed_gt_count"],
            "spurious_pred_count": conf_ov["spurious_pred_count"],
        },
        "baseline_comparison": {
            "candidate1_pq": baseline_conf_pq,
            "candidate1_dice": baseline_conf_dice,
            "candidate1_tp": baseline_conf_tp,
            "candidate1_fp": baseline_conf_fp,
            "candidate1_fn": baseline_conf_fn,
            "heuristic_baseline_pq": 0.1073,
            "delta_pq_vs_candidate1": conf_ov["pq"] - baseline_conf_pq,
            "delta_tp_vs_candidate1": conf_ov["tp"] - baseline_conf_tp,
            "delta_fp_vs_candidate1": conf_ov["fp"] - baseline_conf_fp,
        },
        "per_observation": conf_report["per_observation"],
    }
    with open(conf_summary_path, "w", encoding="utf-8") as f:
        json.dump(conf_summary, f, indent=2)
    print(f"Confirmation evaluation report written to: {conf_summary_path}")

    # =========================================================================
    # STEP 6: Pre-declared Iteration 2 Gate Evaluation
    # =========================================================================
    print("\n--- STEP 6: Pre-declared Iteration 2 Gate Evaluation ---")
    gate_1_threshold = base_ov["pq"] + 0.001
    gate_1_passed = bool(overall_winner["pq"] >= gate_1_threshold)
    gate_1_delta = overall_winner["pq"] - base_ov["pq"]

    gate_2_threshold = baseline_conf_pq - 0.01
    gate_2_passed = bool(conf_ov["pq"] >= gate_2_threshold)
    gate_2_delta = conf_ov["pq"] - baseline_conf_pq

    print(f"Gate 1 (Tuning PQ >= baseline + 0.001):")
    print(f"  Actual: {overall_winner['pq']:.6f} vs Threshold: {gate_1_threshold:.6f} (Delta: {gate_1_delta:+.6f}) -> {'PASSED' if gate_1_passed else 'FAILED'}")

    print(f"Gate 2 (Comparison PQ >= candidate1 - 0.01):")
    print(f"  Actual: {conf_ov['pq']:.6f} vs Threshold: {gate_2_threshold:.6f} (Delta: {gate_2_delta:+.6f}) -> {'PASSED' if gate_2_passed else 'FAILED'}")

    all_gates_passed = gate_1_passed and gate_2_passed
    print(f"\nOverall Pre-Declared Gates Status: {'ALL GATES PASSED - Proceeding to Candidate 2 Generation' if all_gates_passed else 'GATES FAILED - Stopping without Candidate 2 generation'}")

    if not all_gates_passed:
        print("\n[STOP] Pre-declared calibration gates failed. As directed by Codex, reporting metrics and stopping.")
        return {
            "status": "gates_failed",
            "gate_1_passed": gate_1_passed,
            "gate_2_passed": gate_2_passed,
            "winning_setting": overall_winner,
            "confirmation_metrics": conf_ov,
        }

    # =========================================================================
    # STEP 7: Candidate 2 Test Inference & Audit
    # =========================================================================
    print("\n--- STEP 7: Generating Candidate 2 Submission (180 Test Observations) ---")
    test_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    output_candidate2_csv = Path("artifacts/submission_candidate_2.csv")
    output_candidate2_manifest = Path("artifacts/submission_candidate_2.manifest.json")
    output_candidate2_selection = Path("artifacts/submission_candidate_2.selection_config.json")

    image_paths = sorted(
        list(test_dir.glob("*.jpeg"))
        + list(test_dir.glob("*.jpg"))
        + list(test_dir.glob("*.png"))
    )
    obs_to_img = {}
    for p in image_paths:
        obs_id = canonical_observation_id(p.stem)
        if obs_id not in obs_to_img:
            obs_to_img[obs_id] = p
    assert len(obs_to_img) == 180, f"Expected 180 test observations, found {len(obs_to_img)}"

    # Load model on device
    model = build_model(training_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    csv_rows = []
    manifest_obs = []
    
    t_test_start = time.time()
    for idx, (obs_id, img_path) in enumerate(sorted(obs_to_img.items()), start=1):
        t0_img = time.time()
        img_rgb = load_solar_image(img_path)
        fg_map, _, _, _ = predict_full_observation(
            model=model,
            image_rgb=img_rgb,
            device=device,
            tile_size=512,
            stride=256,
            tile_batch_size=16,
            norm_mode="imagenet",
            include_aux=False,
        )

        instances = extract_instances_from_maps(
            foreground_prob=fg_map,
            obs_id=obs_id,
            high_threshold=overall_winner["high_threshold"],
            low_threshold=overall_winner["low_threshold"],
            min_area=overall_winner["min_area"],
            method="connected_components",
            max_instances=overall_winner["max_instances"],
            shape=NATIVE_IMAGE_SHAPE,
        )
        elapsed = time.time() - t0_img

        if len(instances) > 0:
            manifest_obs.append({
                "observation_id": obs_id,
                "status": "processed",
                "instance_count": len(instances),
            })
            for inst in instances:
                csv_rows.append({
                    "filament_id": inst.filament_id,
                    "segmentation_rle": inst.rle_counts,
                })
        else:
            manifest_obs.append({
                "observation_id": obs_id,
                "status": "abstained",
                "instance_count": 0,
            })

        if idx % 20 == 0 or idx == len(obs_to_img):
            print(f"  [{idx:03d}/180] {obs_id}: {len(instances)} instances ({elapsed:.2f}s, total: {len(csv_rows)})", flush=True)

    # Write Candidate 2 CSV
    output_candidate2_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_candidate2_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(csv_rows)

    c2_csv_sha = compute_file_sha256(output_candidate2_csv)
    c1_csv_sha = compute_file_sha256("artifacts/submission_candidate_1_verified.csv")

    print(f"\nCandidate 2 CSV SHA256: {c2_csv_sha}")
    print(f"Candidate 1 CSV SHA256: {c1_csv_sha}")
    assert c2_csv_sha != c1_csv_sha, "Gate 3 failed: Candidate 2 CSV SHA256 must differ from Candidate 1!"
    print("Gate 3 (Distinct candidate content) PASSED: SHA256 hashes differ.")

    proc_count = sum(1 for o in manifest_obs if o["status"] == "processed")
    abst_count = sum(1 for o in manifest_obs if o["status"] == "abstained")
    total_inst = len(csv_rows)

    # Write Manifest
    manifest_data = {
        "manifest_version": "2.0.0",
        "csv_file": output_candidate2_csv.name,
        "csv_sha256": c2_csv_sha,
        "candidate_id": "candidate_2",
        "total_test_observations": len(obs_to_img),
        "total_instances": total_inst,
        "detected_observations_count": proc_count,
        "abstained_observations_count": abst_count,
        "checkpoint_provenance": {
            "epoch": 6,
            "path": str(ckpt_path),
            "file_sha256": ckpt_file_sha,
            "model_state_dict_sha256": sd_sha,
            "training_config_sha256": training_cfg_sha,
            "fold_provenance": "verified",
            "fold": 0,
        },
        "partitions_provenance": {
            "partitions_file": partitions_path.name,
            "partitions_sha256": partitions_sha,
            "tuning_observations_count": 10,
            "confirmation_observations_count": 9,
        },
        "inference_configuration": {
            "resolved_config": winning_resolved_inf_cfg,
            "resolved_config_sha256": winning_resolved_cfg_sha,
            "tile_size": 512,
            "stride": 256,
            "norm_mode": "imagenet",
            "precision": "float32",
            "tile_batch_size": 16,
            "cache_version": "v3_fp32",
        },
        "validation_evidence": {
            "tuning": {
                "candidate2_metrics": {
                    "pq": overall_winner["pq"],
                    "sq": overall_winner["sq"],
                    "rq": overall_winner["rq"],
                    "mean_dice": overall_winner["mean_dice"],
                    "tp": overall_winner["tp"],
                    "fp": overall_winner["fp"],
                    "fn": overall_winner["fn"],
                },
                "candidate1_baseline": {
                    "pq": base_ov["pq"],
                    "tp": base_ov["tp"],
                    "fp": base_ov["fp"],
                    "fn": base_ov["fn"],
                },
                "delta_pq_over_candidate1": overall_winner["delta_pq_vs_baseline"],
            },
            "confirmation": {
                "candidate2_metrics": {
                    "pq": conf_ov["pq"],
                    "sq": conf_ov["sq"],
                    "rq": conf_ov["rq"],
                    "mean_dice": conf_ov["mean_dice"],
                    "tp": conf_ov["tp"],
                    "fp": conf_ov["fp"],
                    "fn": conf_ov["fn"],
                },
                "candidate1_baseline": {
                    "pq": baseline_conf_pq,
                    "tp": baseline_conf_tp,
                    "fp": baseline_conf_fp,
                    "fn": baseline_conf_fn,
                },
                "delta_pq_over_candidate1": conf_ov["pq"] - baseline_conf_pq,
                "history_disclosure": "previously inspected validation partition with 9 unique physical observations",
            },
        },
        "observations": manifest_obs,
    }

    with open(output_candidate2_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Write Selection Config
    selection_config_data = {
        "candidate_id": "candidate_2",
        "description": "Iteration 2 calibrated filter package (tested 10 settings, frozen on tuning, verified on confirmation)",
        "checkpoint_file": str(ckpt_path),
        "checkpoint_sha256": ckpt_file_sha,
        "model_state_dict_sha256": sd_sha,
        "resolved_inference_config": winning_resolved_inf_cfg,
        "reproduction_commands": {
            "calibration_and_evaluation": "python scripts/iteration2_experiment.py",
            "test_inference": (
                f"python inference.py --checkpoint {ckpt_path} "
                f"--output_csv {output_candidate2_csv} "
                f"--output_manifest {output_candidate2_manifest} "
                f"--method connected_components "
                f"--high-threshold {winning_resolved_inf_cfg['high_threshold']} "
                f"--low-threshold {winning_resolved_inf_cfg['low_threshold']} "
                f"--min-area {winning_resolved_inf_cfg['min_area']} "
                f"--tile-size 512 --stride 256"
            ),
        },
    }
    with open(output_candidate2_selection, "w", encoding="utf-8") as f:
        json.dump(selection_config_data, f, indent=2)

    # Native RLE audit
    print("\nRunning native audit on Candidate 2 CSV and Manifest...")
    audit = audit_submission_and_manifest(
        csv_path=output_candidate2_csv,
        manifest_path=output_candidate2_manifest,
        expected_observation_ids=set(obs_to_img.keys()),
    )
    assert audit["is_valid"], f"Audit failed: {audit}"
    print(f"Native audit PASSED: {total_inst} instances across {proc_count} observations ({abst_count} abstentions).")

    total_time = time.time() - start_total_time
    print(f"\n=======================================================")
    print(f"Iteration 2 Completed Successfully in {total_time:.1f}s!")
    print(f"  Candidate 2 CSV:      {output_candidate2_csv} (SHA256: {c2_csv_sha})")
    print(f"  Candidate 2 Manifest: {output_candidate2_manifest}")
    print(f"  Selection Record:     {selection_record_path}")
    print(f"  Comparison Report:    {conf_summary_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    run_iteration2()
