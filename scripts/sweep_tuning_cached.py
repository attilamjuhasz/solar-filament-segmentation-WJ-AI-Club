import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from evaluate import compute_instance_diagnostics
from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.annotations import load_coco_annotations
from src.data.folds import get_deterministic_fold_partitions, load_frozen_folds_manifest
from src.evaluation.competition_adapter import evaluate_entry_pq
from src.inference.engine import load_cached_prediction
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import decode_instance


def test_grid():
    manifest_path = Path("artifacts/folds_manifest.json")
    assignments, _ = load_frozen_folds_manifest(manifest_path)
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ann_index = load_coco_annotations(str(train_json))
    parts = get_deterministic_fold_partitions(assignments, target_fold=0)
    tuning_obs = parts["tuning"]

    ckpt_hash = "94a4cbd785a08bb6"
    cache_dir = Path("artifacts/cache")

    # Preload all 10 tuning probability maps and GT masks
    preloaded = []
    for obs_id in tuning_obs:
        cached = load_cached_prediction(
            cache_dir=cache_dir,
            ckpt_hash=ckpt_hash,
            obs_id=obs_id,
            expected_norm_mode="imagenet",
        )
        if cached is None:
            print(f"Missing cached map for {obs_id}")
            continue
        fg, ctr, bnd, off = cached
        variants = ann_index.get_observation_variants(obs_id)
        gt_entries = []
        for v in variants:
            gt_masks = [inst.get_mask(NATIVE_IMAGE_SHAPE) for inst in v.instances if inst.area > 0]
            gt_masks = [m for m in gt_masks if (m > 0).any()]
            gt_entries.append((v.annotator_image_id, gt_masks))
        preloaded.append((obs_id, fg, ctr, bnd, off, gt_entries))

    print(f"Loaded {len(preloaded)} tuning observations with cached maps.")

    grid = [
        # (method, high, low, min_area, max_inst)
        ("connected_components", 0.50, 0.30, 50, 100),
        ("connected_components", 0.60, 0.35, 100, 50),
        ("connected_components", 0.65, 0.40, 200, 30),
        ("connected_components", 0.70, 0.45, 300, 20),
        ("connected_components", 0.75, 0.50, 400, 15),
        ("connected_components", 0.80, 0.50, 500, 10),
        ("connected_components", 0.85, 0.60, 500, 10),
        ("connected_components", 0.90, 0.70, 500, 10),
    ]

    best_pq = -1.0
    best_cfg = None

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

        print(f"Cfg: {method} high={high:.2f} low={low:.2f} min_area={min_area} max_inst={max_inst:2d} | "
              f"PQ={pq:.4f} SQ={sq:.4f} RQ={rq:.4f} Dice={mean_dice:.4f} | TP={tp_tot:2d} FP={fp_tot:3d} FN={fn_tot:3d} TotalPreds={total_preds}")

        if pq > best_pq:
            best_pq = pq
            best_cfg = (method, high, low, min_area, max_inst, pq, sq, rq, mean_dice, tp_tot, fp_tot, fn_tot)

    print("\nBest Tuning Configuration:")
    print(f"  {best_cfg}")


if __name__ == "__main__":
    test_grid()
