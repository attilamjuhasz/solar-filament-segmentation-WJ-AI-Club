from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.dataset import SolarFilamentDataset
from src.data.folds import load_frozen_folds_manifest
from src.evaluation.competition_adapter import evaluate_entry_pq
from src.inference.engine import compute_file_sha256, load_cached_prediction
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import decode_instance
from evaluate import compute_instance_diagnostics


def run_sweep(
    checkpoint_path: str = "checkpoints/resnet34_unet_fold0_latest.pt",
    cache_dir: str = "artifacts/cache",
    calibration_ratio: float = 0.50,
):
    ckpt_path = Path(checkpoint_path)
    ckpt_sha256 = compute_file_sha256(ckpt_path)
    root_dir = Path(__file__).resolve().parent.parent

    # Load fold 0 validation dataset
    manifest_path = root_dir / "artifacts" / "folds_manifest.json"
    fold_assignments, _ = load_frozen_folds_manifest(manifest_path)
    train_json = root_dir / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_images = train_json.parent / "train_images"

    val_ds = SolarFilamentDataset(
        images_dir=train_images,
        annotation_index=val_ds_index if "val_ds_index" in locals() else None,
        annotations_json=train_json,
        fold_assignments=fold_assignments,
        target_fold=0,
        is_train=False,
        patch_size=(2048, 2048),
    )

    eval_obs_list = val_ds.observations[:5]
    n_calib = max(1, int(len(eval_obs_list) * calibration_ratio))
    calib_obs = set(eval_obs_list[:n_calib])
    confirm_obs = set(eval_obs_list[n_calib:])

    print(f"Loaded {len(eval_obs_list)} validation observations: Calib={len(calib_obs)}, Confirm={len(confirm_obs)}")

    # Preload cached maps and GT masks for the 5 observations
    preloaded = []
    for obs_id in eval_obs_list:
        cached = load_cached_prediction(cache_dir, ckpt_sha256, obs_id)
        if cached is None:
            raise FileNotFoundError(f"Missing cached prediction for obs {obs_id} and ckpt {ckpt_sha256[:8]}")
        variants = val_ds.get_observation_annotations(obs_id)
        var_gt_masks = []
        for var in variants:
            gt_masks = [inst.get_mask(NATIVE_IMAGE_SHAPE) for inst in var.instances if inst.area > 0]
            gt_masks = [m for m in gt_masks if (m > 0).any()]
            var_gt_masks.append(gt_masks)
        preloaded.append((obs_id, cached, var_gt_masks))

    # Predeclared parameter grid
    grid = [
        # Connected components variations
        {"method": "connected_components", "high": 0.50, "low": 0.35, "ctr": 0.35, "bnd_w": 0.5, "min_dist": 7, "min_area": 16},
        {"method": "connected_components", "high": 0.52, "low": 0.38, "ctr": 0.35, "bnd_w": 0.5, "min_dist": 7, "min_area": 32},
        {"method": "connected_components", "high": 0.55, "low": 0.40, "ctr": 0.35, "bnd_w": 0.5, "min_dist": 7, "min_area": 32},
        {"method": "connected_components", "high": 0.58, "low": 0.42, "ctr": 0.35, "bnd_w": 0.5, "min_dist": 7, "min_area": 50},
        {"method": "connected_components", "high": 0.62, "low": 0.45, "ctr": 0.35, "bnd_w": 0.5, "min_dist": 7, "min_area": 64},
        # Watershed variations with center proposals
        {"method": "watershed", "high": 0.50, "low": 0.35, "ctr": 0.45, "bnd_w": 0.5, "min_dist": 7, "min_area": 16},
        {"method": "watershed", "high": 0.52, "low": 0.38, "ctr": 0.50, "bnd_w": 0.5, "min_dist": 7, "min_area": 32},
        {"method": "watershed", "high": 0.55, "low": 0.38, "ctr": 0.55, "bnd_w": 0.5, "min_dist": 7, "min_area": 32},
        {"method": "watershed", "high": 0.55, "low": 0.40, "ctr": 0.60, "bnd_w": 0.5, "min_dist": 7, "min_area": 50},
        {"method": "watershed", "high": 0.58, "low": 0.42, "ctr": 0.65, "bnd_w": 0.5, "min_dist": 7, "min_area": 50},
        {"method": "watershed", "high": 0.58, "low": 0.42, "ctr": 0.70, "bnd_w": 0.7, "min_dist": 7, "min_area": 50},
        {"method": "watershed", "high": 0.60, "low": 0.45, "ctr": 0.75, "bnd_w": 0.5, "min_dist": 10, "min_area": 64},
    ]

    sweep_results = []

    for cfg in grid:
        # Accumulate metrics
        acc_calib = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "dice_scores": [], "frag": 0, "merge": 0, "miss": 0, "n_gt": 0, "n_pred": 0}
        acc_confirm = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "dice_scores": [], "frag": 0, "merge": 0, "miss": 0, "n_gt": 0, "n_pred": 0}

        for obs_id, (full_fg, full_ctr, full_bnd, full_off), var_gt_masks in preloaded:
            is_calib = obs_id in calib_obs
            target_acc = acc_calib if is_calib else acc_confirm

            pred_instances = extract_instances_from_maps(
                foreground_prob=full_fg,
                center_prob=full_ctr,
                boundary_prob=full_bnd,
                offset_field=full_off,
                obs_id=obs_id,
                high_threshold=cfg["high"],
                low_threshold=cfg["low"],
                min_area=cfg["min_area"],
                method=cfg["method"],
                center_threshold=cfg["ctr"],
                boundary_weight=cfg["bnd_w"],
                marker_min_distance=cfg["min_dist"],
                max_peaks=200,
                max_instances=200,
                shape=NATIVE_IMAGE_SHAPE,
            )
            pred_masks = [decode_instance(p.rle_counts, shape=NATIVE_IMAGE_SHAPE) for p in pred_instances]

            for gt_masks in var_gt_masks:
                iou_sum, tp, fp, fn, _, _ = evaluate_entry_pq(gt_masks, pred_masks)
                target_acc["tp"] += tp
                target_acc["fp"] += fp
                target_acc["fn"] += fn
                target_acc["iou_sum"] += iou_sum

                diag = compute_instance_diagnostics(gt_masks, pred_masks)
                target_acc["frag"] += diag["fragmented_gt_count"]
                target_acc["merge"] += diag["over_merged_pred_count"]
                target_acc["miss"] += diag["missed_gt_count"]
                target_acc["n_gt"] += diag["n_gt"]
                target_acc["n_pred"] += diag["n_pred"]

                gt_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
                for gm in gt_masks:
                    gt_union |= (gm > 0)
                pred_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
                for pm in pred_masks:
                    pred_union |= (pm > 0)
                inter = 2.0 * np.logical_and(gt_union, pred_union).sum()
                union = gt_union.sum() + pred_union.sum()
                target_acc["dice_scores"].append(float((inter + 1e-6) / (union + 1e-6)))

        def summarize(acc):
            tp = acc["tp"]
            fp = acc["fp"]
            fn = acc["fn"]
            denom = tp + 0.5 * fp + 0.5 * fn
            rq = tp / denom if denom > 0 else 0.0
            sq = acc["iou_sum"] / tp if tp > 0 else 0.0
            pq = sq * rq
            mean_dice = float(np.mean(acc["dice_scores"])) if acc["dice_scores"] else 0.0
            n_gt = max(1, acc["n_gt"])
            return {
                "pq": pq,
                "sq": sq,
                "rq": rq,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_dice": mean_dice,
                "miss_rate": acc["miss"] / n_gt,
                "frag_rate": acc["frag"] / n_gt,
                "merge_rate": acc["merge"] / max(1, acc["n_pred"]),
                "n_pred": acc["n_pred"],
                "n_gt": acc["n_gt"],
            }

        calib_stats = summarize(acc_calib)
        confirm_stats = summarize(acc_confirm)

        sweep_results.append({
            "config": cfg,
            "calib": calib_stats,
            "confirm": confirm_stats,
        })

    # Sort by calibration PQ then calibration Dice
    sweep_results.sort(key=lambda r: (r["calib"]["pq"], r["calib"]["mean_dice"], -r["calib"]["fp"]), reverse=True)

    print("\n" + "=" * 80)
    print("CALIBRATION SWEEP RESULTS (Sorted by Calib PQ -> Calib Dice):")
    print(f"{'Method':<20} {'High':<5} {'Low':<5} {'Ctr':<5} {'MinA':<5} | {'Calib PQ':<9} {'Calib Dice':<10} {'Calib FP':<9} | {'Confirm PQ':<11} {'Confirm Dice':<12}")
    print("-" * 80)
    for r in sweep_results:
        c = r["config"]
        cal = r["calib"]
        cnf = r["confirm"]
        print(f"{c['method']:<20} {c['high']:<5.2f} {c['low']:<5.2f} {c['ctr']:<5.2f} {c['min_area']:<5} | {cal['pq']:<9.4f} {cal['mean_dice']:<10.4f} {cal['fp']:<9} | {cnf['pq']:<11.4f} {cnf['mean_dice']:<12.4f}")
    print("=" * 80)

    # Persist report
    out_file = root_dir / "artifacts" / "reports" / f"sweep_{ckpt_sha256[:8]}_fold0.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Full sweep saved to {out_file}")
    return sweep_results


if __name__ == "__main__":
    run_sweep()
