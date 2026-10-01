import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.annotations import load_coco_annotations
from src.data.folds import get_deterministic_fold_partitions, load_frozen_folds_manifest
from src.evaluation.competition_adapter import evaluate_entry_pq
from src.inference.engine import load_cached_prediction
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import decode_instance


def sweep_epoch2():
    manifest_path = Path("artifacts/folds_manifest.json")
    assignments, _ = load_frozen_folds_manifest(manifest_path)
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ann_index = load_coco_annotations(str(train_json))
    parts = get_deterministic_fold_partitions(assignments, target_fold=0)
    tuning_obs = parts["tuning"]

    ckpt_hash = "5d63b582f5743eb9"
    cache_dir = Path("artifacts/cache")

    preloaded = []
    for obs_id in tuning_obs:
        cached = load_cached_prediction(
            cache_dir=cache_dir,
            ckpt_hash=ckpt_hash,
            obs_id=obs_id,
            expected_norm_mode="imagenet",
        )
        if cached is None:
            print(f"[Warning] Missing cache for {obs_id}")
            continue
        fg, ctr, bnd, off = cached
        variants = ann_index.get_observation_variants(obs_id)
        gt_entries = []
        for v in variants:
            gt_masks = [inst.get_mask(NATIVE_IMAGE_SHAPE) for inst in v.instances if inst.area > 0]
            gt_masks = [m for m in gt_masks if (m > 0).any()]
            gt_entries.append((v.annotator_image_id, gt_masks))
        preloaded.append((obs_id, fg, ctr, bnd, off, gt_entries))

    print(f"Loaded {len(preloaded)} tuning observations for checkpoint {ckpt_hash}")

    grid = [
        # (method, high, low, min_area, max_inst)
        ("connected_components", 0.90, 0.70, 500, 10),
        ("connected_components", 0.92, 0.75, 500, 10),
        ("connected_components", 0.94, 0.80, 500, 10),
        ("connected_components", 0.95, 0.85, 500, 10),
        ("connected_components", 0.96, 0.88, 500, 10),
        ("connected_components", 0.97, 0.90, 500, 10),
        ("connected_components", 0.98, 0.92, 500, 10),
        ("connected_components", 0.95, 0.85, 300, 15),
        ("connected_components", 0.96, 0.88, 300, 15),
        ("connected_components", 0.95, 0.85, 800, 10),
        ("connected_components", 0.96, 0.88, 800, 10),
        ("connected_components", 0.94, 0.80, 400, 12),
        ("connected_components", 0.95, 0.85, 400, 12),
        ("connected_components", 0.96, 0.88, 400, 12),
    ]

    results = []
    for method, high, low, min_area, max_inst in grid:
        total_iou = 0.0
        tp_tot = 0
        fp_tot = 0
        fn_tot = 0
        dice_scores = []
        total_preds = 0

        for obs_id, fg, ctr, bnd, off, gt_entries in preloaded:
            insts = extract_instances_from_maps(
                foreground_prob=fg,
                center_prob=ctr,
                boundary_prob=bnd,
                offset_field=off,
                obs_id=obs_id,
                high_threshold=high,
                low_threshold=low,
                min_area=min_area,
                method=method,
                max_instances=max_inst,
            )
            pred_masks = [decode_instance(p.rle_counts, shape=NATIVE_IMAGE_SHAPE) for p in insts]
            total_preds += len(pred_masks)

            pred_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
            for pm in pred_masks:
                pred_union |= (pm > 0)

            for _, gt_masks in gt_entries:
                iou_sum, tp, fp, fn, _, _ = evaluate_entry_pq(gt_masks, pred_masks)
                total_iou += iou_sum
                tp_tot += tp
                fp_tot += fp
                fn_tot += fn

                gt_union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=bool)
                for gm in gt_masks:
                    gt_union |= (gm > 0)
                inter = int(np.logical_and(gt_union, pred_union).sum())
                union = int(gt_union.sum()) + int(pred_union.sum())
                d = (2.0 * inter / union) if union > 0 else (1.0 if int(gt_union.sum()) == 0 else 0.0)
                dice_scores.append(d)

        denom = tp_tot + 0.5 * (fp_tot + fn_tot)
        pq = total_iou / denom if denom > 0 else 0.0
        sq = total_iou / tp_tot if tp_tot > 0 else 0.0
        rq = tp_tot / denom if denom > 0 else 0.0
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

        rec = {
            "method": method,
            "high": high,
            "low": low,
            "min_area": min_area,
            "max_inst": max_inst,
            "pq": round(pq, 5),
            "sq": round(sq, 5),
            "rq": round(rq, 5),
            "mean_dice": round(mean_dice, 5),
            "tp": tp_tot,
            "fp": fp_tot,
            "fn": fn_tot,
            "total_preds": total_preds,
        }
        results.append(rec)
        print(f"high={high:.2f} low={low:.2f} min_area={min_area:4d} max_inst={max_inst:2d} | PQ={pq:.4f} SQ={sq:.4f} RQ={rq:.4f} Dice={mean_dice:.4f} | TP={tp_tot:2d} FP={fp_tot:3d} FN={fn_tot:3d} Preds={total_preds}")

    # Rank by primary: PQ, secondary: Mean Dice, tertiary: RQ
    results.sort(key=lambda x: (x["pq"], x["mean_dice"], x["rq"]), reverse=True)
    print("\nTop tuning configuration for Epoch 2:")
    print(results[0])

    sweep_path = Path("artifacts/reports/sweep_tuning_epoch2_selected.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": "tuning",
            "checkpoint_sha256": ckpt_hash,
            "top_configuration": results[0],
            "all_results": results,
        }, f, indent=2)
    print(f"Saved tuning sweep report to: {sweep_path}")


if __name__ == "__main__":
    sweep_epoch2()
