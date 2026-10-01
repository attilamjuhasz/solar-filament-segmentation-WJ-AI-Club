from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import torch

from src.contracts import InstancePrediction, NATIVE_IMAGE_SHAPE
from src.data.annotations import AnnotationIndex, decode_coco_segmentation, load_coco_annotations
from src.data.dataset import load_solar_image
from src.data.folds import canonical_observation_id
from src.inference.engine import (
    compute_bytes_sha256,
    compute_file_sha256,
    compute_state_dict_sha256,
    predict_full_observation,
)
from src.inference.instances import extract_instances_from_maps
from src.models import build_model


def select_mining_physical_ids(
    training_observations: Sequence[str],
    seed_prefix: str = "2026",
    count: int = 128,
) -> List[str]:
    """Deterministically select 128 physical training observations sorted by SHA256 of '2026|<canonical_id>'."""
    unique_canons = sorted(list(set(canonical_observation_id(o) for o in training_observations)))
    if len(unique_canons) < count:
        raise ValueError(f"Requested {count} observations but only {len(unique_canons)} unique training observations available.")

    def sort_key(cid: str) -> str:
        return hashlib.sha256(f"{seed_prefix}|{cid}".encode("utf-8")).hexdigest()

    sorted_canons = sorted(unique_canons, key=sort_key)
    return sorted_canons[:count]


def compute_component_metrics(
    component_mask: np.ndarray,
    parent_fg_map: np.ndarray,
    annotator_variants: List[Any],
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE,
) -> Dict[str, Any]:
    """Compute IoU against all variant instances and overlap with foreground union."""
    comp_pixels = component_mask > 0
    comp_area = int(comp_pixels.sum())
    if comp_area == 0:
        return {
            "eligible": False,
            "max_iou": 0.0,
            "union_overlap_ratio": 0.0,
            "mean_prob": 0.0,
            "area": 0,
        }

    mean_prob = float(parent_fg_map[comp_pixels].mean())

    # Build union mask and compute instance IoUs
    all_gt_union = np.zeros(shape, dtype=bool)
    max_iou = 0.0

    for var in annotator_variants:
        for inst in var.instances:
            gt_mask = (inst.mask > 0)
            all_gt_union |= gt_mask

            inter = int(np.logical_and(comp_pixels, gt_mask).sum())
            if inter > 0:
                union = int(np.logical_or(comp_pixels, gt_mask).sum())
                iou = inter / union if union > 0 else 0.0
                if iou > max_iou:
                    max_iou = iou

    union_inter = int(np.logical_and(comp_pixels, all_gt_union).sum())
    union_overlap_ratio = float(union_inter / comp_area)

    # Hard-negative eligibility condition:
    # max IoU < 0.1 against every GT instance AND union overlap <= 0.05
    is_eligible = bool(max_iou < 0.10 and union_overlap_ratio <= 0.05)

    # Bounding box of component
    y_indices, x_indices = np.where(comp_pixels)
    min_y, max_y = int(y_indices.min()), int(y_indices.max())
    min_x, max_x = int(x_indices.min()), int(x_indices.max())

    return {
        "eligible": is_eligible,
        "max_iou": round(max_iou, 6),
        "union_overlap_ratio": round(union_overlap_ratio, 6),
        "mean_prob": round(mean_prob, 6),
        "area": comp_area,
        "bbox": [min_y, min_x, max_y, max_x],
    }


def find_non_overlapping_crop_boxes(
    candidates: List[Dict[str, Any]],
    box_size: int = 512,
    max_boxes: int = 2,
    canvas_shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE,
) -> List[Dict[str, Any]]:
    """Select up to max_boxes non-overlapping box_size x box_size crops from ranked candidates.
    
    Candidates are sorted by:
    1. Descending mean parent foreground probability
    2. Descending component area
    3. Lexicographic top-left of component bounding box
    """
    def rank_key(c: Dict[str, Any]) -> Tuple[float, int, int, int]:
        bbox = c["bbox"]
        return (-c["mean_prob"], -c["area"], bbox[0], bbox[1])

    sorted_candidates = sorted(candidates, key=rank_key)

    H, W = canvas_shape
    selected_boxes: List[Dict[str, Any]] = []

    def boxes_overlap(b1: List[int], b2: List[int]) -> bool:
        # b = [y0, x0, y1, x1]
        return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])

    for cand in sorted_candidates:
        if len(selected_boxes) >= max_boxes:
            break

        min_y, min_x, max_y, max_x = cand["bbox"]
        cy = (min_y + max_y) // 2
        cx = (min_x + max_x) // 2

        y0 = max(0, min(H - box_size, cy - box_size // 2))
        x0 = max(0, min(W - box_size, cx - box_size // 2))
        y1 = y0 + box_size
        x1 = x0 + box_size
        new_box = [y0, x0, y1, x1]

        # Check non-overlap with previously selected boxes
        overlap = any(boxes_overlap(new_box, sb["crop_box"]) for sb in selected_boxes)
        if not overlap:
            selected_boxes.append({
                "crop_box": new_box,
                "anchor_bbox": cand["bbox"],
                "mean_prob": cand["mean_prob"],
                "area": cand["area"],
                "max_iou": cand.get("max_iou", 0.0),
                "union_overlap_ratio": cand.get("union_overlap_ratio", 0.0),
            })

    return selected_boxes


def build_and_freeze_mining_bank(
    parent_checkpoint_path: Union[str, Path],
    train_images_dir: Union[str, Path],
    train_json_path: Union[str, Path],
    folds_manifest_path: Union[str, Path] = "artifacts/folds_manifest.json",
    partitions_path: Union[str, Path] = "artifacts/partitions_migrated_v1.json",
    output_bank_path: Union[str, Path] = "artifacts/reports/mining_bank_v1.json",
    device_str: Optional[str] = None,
    count: int = 128,
) -> Dict[str, Any]:
    """Execute parent inference across 128 deterministic training observations and freeze hard-negative mining bank."""
    from datetime import datetime, timezone
    from src.inference.rle import decode_instance
    from src.data.folds import load_frozen_folds_manifest

    parent_path = Path(parent_checkpoint_path)
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent checkpoint not found: {parent_path}")

    ckpt_file_sha = compute_file_sha256(parent_path)
    is_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_name = device_str if device_str else ("cuda" if is_cuda else "cpu")
    device = torch.device(device_name)

    ckpt = torch.load(str(parent_path), map_location=device, weights_only=False)
    sd_sha = compute_state_dict_sha256(ckpt["model_state_dict"])
    training_cfg = ckpt.get("config", {})
    cfg_sha = hashlib.sha256(json.dumps(training_cfg, sort_keys=True).encode("utf-8")).hexdigest()

    # Load annotations and manifest
    ann_index = load_coco_annotations(str(train_json_path))
    fold_assignments, folds_sha = load_frozen_folds_manifest(folds_manifest_path)

    # 1. Deterministically select the 128 physical training observations
    train_obs_recorded = ckpt.get("train_observations", [])
    mining_ids = select_mining_physical_ids(train_obs_recorded, count=count)

    # 2. Strict leakage isolation audit
    fold0_canons = set(canonical_observation_id(k) for k, f in fold_assignments.items() if f == 0)
    overlap_fold0 = set(mining_ids).intersection(fold0_canons)
    if overlap_fold0:
        raise ValueError(f"Strict isolation failure! Mining IDs overlap with fold 0: {sorted(list(overlap_fold0))}")

    with open(partitions_path, "r", encoding="utf-8") as f:
        migrated_parts = json.load(f)
    eval_partition_ids = set(
        migrated_parts["tuning"]["canonical_observation_ids"]
        + migrated_parts["confirmation"]["canonical_observation_ids"]
        + migrated_parts["explored"]["canonical_observation_ids"]
    )
    overlap_eval = set(mining_ids).intersection(eval_partition_ids)
    if overlap_eval:
        raise ValueError(f"Strict isolation failure! Mining IDs overlap with evaluation partitions: {sorted(list(overlap_eval))}")

    print(f"[Mining] Verified {len(mining_ids)} physical training observations with ZERO fold-0 / evaluation overlap.")

    # 3. Load model in eval mode
    model = build_model(training_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Map training observations to image files
    train_dir = Path(train_images_dir)
    obs_to_path: Dict[str, Path] = {}
    for p in list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")):
        c_id = canonical_observation_id(p.stem)
        if c_id in mining_ids and c_id not in obs_to_path:
            obs_to_path[c_id] = p

    missing_paths = [m for m in mining_ids if m not in obs_to_path]
    if missing_paths:
        raise FileNotFoundError(f"Could not locate image files for {len(missing_paths)} mining observations.")

    obs_records = []
    total_eligible_boxes = 0
    obs_with_boxes = 0

    print(f"[Mining] Screening {len(mining_ids)} training observations with fixed Candidate 2 inference config...")
    for idx, obs_id in enumerate(mining_ids, start=1):
        img_path = obs_to_path[obs_id]
        img_rgb = load_solar_image(img_path)
        img_bytes_sha = compute_bytes_sha256(img_rgb.tobytes())

        # Predict full native observation
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

        # Extract components with Candidate 2 fixed config
        instances: List[InstancePrediction] = extract_instances_from_maps(
            foreground_prob=fg_map,
            obs_id=obs_id,
            high_threshold=0.85,
            low_threshold=0.70,
            min_area=400,
            method="connected_components",
            max_instances=20,
            shape=NATIVE_IMAGE_SHAPE,
        )

        variants = ann_index.get_observation_variants(obs_id)

        # Filter candidate components
        candidate_metrics = []
        for inst in instances:
            comp_mask = decode_instance(inst.rle_counts, shape=NATIVE_IMAGE_SHAPE)
            metrics = compute_component_metrics(
                component_mask=comp_mask,
                parent_fg_map=fg_map,
                annotator_variants=variants,
                shape=NATIVE_IMAGE_SHAPE,
            )
            if metrics["eligible"]:
                candidate_metrics.append(metrics)

        # Select at most 2 non-overlapping 512x512 boxes
        selected_boxes = find_non_overlapping_crop_boxes(
            candidate_metrics,
            box_size=512,
            max_boxes=2,
            canvas_shape=NATIVE_IMAGE_SHAPE,
        )

        if len(selected_boxes) > 0:
            obs_with_boxes += 1
            total_eligible_boxes += len(selected_boxes)

        obs_entry = {
            "physical_id": obs_id,
            "image_sha256": img_bytes_sha,
            "annotator_variants_count": len(variants),
            "predicted_components_count": len(instances),
            "eligible_candidates_count": len(candidate_metrics),
            "selected_crop_boxes_count": len(selected_boxes),
            "crop_boxes": selected_boxes,
        }
        obs_records.append(obs_entry)

        if idx % 20 == 0 or idx == len(mining_ids):
            print(f"  [{idx:03d}/{len(mining_ids)}] {obs_id}: {len(selected_boxes)} eligible boxes (total boxes so far: {total_eligible_boxes})", flush=True)

    # 5. Build frozen mining bank artifact
    bank_data = {
        "bank_version": "1.0.0",
        "description": "Iteration 3 training-only hard-negative crop bank frozen before fine-tuning",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": {
            "path": str(parent_path),
            "file_sha256": ckpt_file_sha,
            "model_state_sha256": sd_sha,
            "training_config_sha256": cfg_sha,
            "parent_epoch": ckpt.get("epoch", 6),
        },
        "mining_config": {
            "selection_prefix": "2026",
            "selected_observations_count": len(mining_ids),
            "high_threshold": 0.85,
            "low_threshold": 0.70,
            "min_area": 400,
            "max_instances": 20,
            "tile_size": 512,
            "stride": 256,
            "tile_batch_size": 16,
            "max_iou_threshold": 0.10,
            "max_union_overlap_ratio": 0.05,
            "max_crop_boxes_per_observation": 2,
            "crop_box_size": 512,
        },
        "summary": {
            "total_selected_observations": len(mining_ids),
            "observations_with_eligible_crops": obs_with_boxes,
            "total_eligible_crop_boxes": total_eligible_boxes,
        },
        "mining_boxes_by_image": {
            obs["physical_id"]: [b["crop_box"] for b in obs["crop_boxes"]]
            for obs in obs_records if obs["selected_crop_boxes_count"] > 0
        },
        "observations": obs_records,
    }

    out_path = Path(output_bank_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bank_data, f, indent=2)

    bank_sha = compute_file_sha256(out_path)
    print(f"\n[Mining] Mining bank successfully frozen to {out_path} (SHA256: {bank_sha})")
    print(f"  {obs_with_boxes}/{len(mining_ids)} observations produced {total_eligible_boxes} non-overlapping hard-negative crop boxes.")

def verify_mining_bank_provenance(
    bank_path: Union[str, Path],
    expected_parent_sha: Optional[str] = None,
    expected_folds_manifest_sha: Optional[str] = None,
    expected_partitions_sha: Optional[str] = None,
    folds_manifest_path: Union[str, Path] = "artifacts/folds_manifest.json",
    partitions_path: Union[str, Path] = "artifacts/partitions_migrated_v1.json",
) -> Dict[str, Any]:
    """Verify that the frozen mining bank artifact is cryptographically and structurally valid."""
    bank_file = Path(bank_path)
    if not bank_file.is_file():
        raise FileNotFoundError(f"Mining bank not found: {bank_file}")

    bank_sha = compute_file_sha256(bank_file)
    with open(bank_file, "r", encoding="utf-8") as f:
        bank_data = json.load(f)

    # 1. Parent checkpoint provenance
    parent_meta = bank_data.get("parent_checkpoint", {})
    parent_file_sha = parent_meta.get("file_sha256")
    if expected_parent_sha and parent_file_sha != expected_parent_sha:
        raise ValueError(f"Mining bank parent file SHA mismatch: {parent_file_sha} != {expected_parent_sha}")

    # 2. Strict isolation checks against fold 0 and evaluation partitions
    if Path(folds_manifest_path).is_file():
        from src.data.folds import load_frozen_folds_manifest
        fold_assignments, folds_sha = load_frozen_folds_manifest(folds_manifest_path)
        if expected_folds_manifest_sha and folds_sha != expected_folds_manifest_sha:
            raise ValueError(f"Folds manifest SHA mismatch during bank verification: {folds_sha}")
        fold0_canons = set(canonical_observation_id(k) for k, f in fold_assignments.items() if f == 0)
    else:
        fold0_canons = set()

    if Path(partitions_path).is_file():
        with open(partitions_path, "r", encoding="utf-8") as f:
            migrated_parts = json.load(f)
        eval_canons = set(
            migrated_parts["tuning"]["canonical_observation_ids"]
            + migrated_parts["confirmation"]["canonical_observation_ids"]
            + migrated_parts["explored"]["canonical_observation_ids"]
        )
    else:
        eval_canons = set()

    mining_boxes_by_image = bank_data.get("mining_boxes_by_image", {})
    all_obs_ids = set(canonical_observation_id(x["physical_id"]) for x in bank_data.get("observations", []))

    overlap_fold0 = all_obs_ids.intersection(fold0_canons)
    if overlap_fold0:
        raise ValueError(f"Mining bank contains fold 0 observations: {sorted(list(overlap_fold0))}")

    overlap_eval = all_obs_ids.intersection(eval_canons)
    if overlap_eval:
        raise ValueError(f"Mining bank contains evaluation partition observations: {sorted(list(overlap_eval))}")

    # 3. Structural box checks: (512, 512), within (2048, 2048), non-overlapping per obs
    total_boxes = 0
    for obs_id, boxes in mining_boxes_by_image.items():
        if len(boxes) > 2:
            raise ValueError(f"Observation {obs_id} has {len(boxes)} > 2 crop boxes")
        for i, box in enumerate(boxes):
            if len(box) != 4:
                raise ValueError(f"Malformed crop box in {obs_id}: {box}")
            y0, x0, y1, x1 = box
            if (y1 - y0 != 512) or (x1 - x0 != 512):
                raise ValueError(f"Crop box in {obs_id} is not 512x512: {box}")
            if y0 < 0 or x0 < 0 or y1 > 2048 or x1 > 2048:
                raise ValueError(f"Crop box in {obs_id} exceeds image bounds: {box}")
            total_boxes += 1
        if len(boxes) == 2:
            b1, b2 = boxes[0], boxes[1]
            overlap_y = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            overlap_x = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            if overlap_y * overlap_x > 0:
                raise ValueError(f"Mining boxes in {obs_id} overlap!")

    return {
        "bank_path": str(bank_file),
        "bank_file_sha256": bank_sha,
        "parent_file_sha256": parent_file_sha,
        "parent_model_state_sha256": parent_meta.get("model_state_sha256"),
        "total_selected_observations": len(all_obs_ids),
        "observations_with_crops": len(mining_boxes_by_image),
        "total_crop_boxes": total_boxes,
        "native_maps_cached": False,
        "is_valid": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and freeze hard-negative mining bank for Iteration 3")
    parser.add_argument(
        "--parent-checkpoint",
        type=str,
        default="artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt",
        help="Path to immutable parent epoch 6 checkpoint",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        default="data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images",
        help="Path to train images directory",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default="data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        help="Path to train annotations JSON",
    )
    parser.add_argument(
        "--folds-manifest",
        type=str,
        default="artifacts/folds_manifest.json",
        help="Path to folds manifest",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default="artifacts/partitions_migrated_v1.json",
        help="Path to migrated partitions JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/mining_bank_v1.json",
        help="Output path for frozen mining bank",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=128,
        help="Number of deterministic training observations to screen",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on",
    )
    args = parser.parse_args()

    build_and_freeze_mining_bank(
        parent_checkpoint_path=args.parent_checkpoint,
        train_images_dir=args.train_images,
        train_json_path=args.train_json,
        folds_manifest_path=args.folds_manifest,
        partitions_path=args.partitions,
        output_bank_path=args.output,
        device_str=args.device,
        count=args.count,
    )

