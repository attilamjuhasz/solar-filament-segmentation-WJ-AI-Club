from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import yaml

from src.contracts import InstancePrediction, NATIVE_IMAGE_SHAPE
from src.data.dataset import extract_solar_disk_mask, load_solar_image
from src.data.manifest import canonical_observation_id
from src.inference.config import UNSET, resolve_inference_config
from src.inference.engine import compute_file_sha256, predict_full_observation
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import audit_submission_and_manifest
from src.models import build_model


def run_inference(
    checkpoint_path: str,
    test_images_dir: str,
    output_csv: str,
    output_manifest: str,
    tile_size: Optional[int] = None,
    stride: Optional[int] = None,
    method: Optional[str] = None,
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
    center_threshold: Optional[float] = None,
    boundary_weight: Optional[float] = None,
    marker_min_distance: Optional[int] = None,
    marker_cap_per_component: Optional[int] = UNSET,
    max_peaks: Optional[int] = None,
    max_instances: Optional[int] = UNSET,
    min_area: Optional[int] = None,
    device_str: Optional[str] = None,
    tile_batch_size: int = 16,
    inference_config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full test inference pipeline and generate audited submission files."""
    is_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_name = device_str if device_str else ("cuda" if is_cuda else "cpu")
    device = torch.device(device_name)
    print(f"[Inference] Using device: {device}")

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_sha256 = compute_file_sha256(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = ckpt.get("config", {"model": {"name": "resnet34_unet"}})
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Resolve unified immutable inference configuration
    overrides: Dict[str, Any] = {}
    if method is not None: overrides["method"] = method
    if high_threshold is not None: overrides["high_threshold"] = high_threshold
    if low_threshold is not None: overrides["low_threshold"] = low_threshold
    if center_threshold is not None: overrides["center_threshold"] = center_threshold
    if boundary_weight is not None: overrides["boundary_weight"] = boundary_weight
    if marker_min_distance is not None: overrides["marker_min_distance"] = marker_min_distance
    if marker_cap_per_component is not UNSET: overrides["marker_cap_per_component"] = marker_cap_per_component
    if max_peaks is not None: overrides["max_peaks"] = max_peaks
    if max_instances is not UNSET: overrides["max_instances"] = max_instances
    if min_area is not None: overrides["min_area"] = min_area
    if tile_size is not None: overrides["tile_size"] = tile_size
    if stride is not None: overrides["stride"] = stride
    if inference_config_overrides is not None:
        overrides.update(inference_config_overrides)

    resolved_inf_cfg = resolve_inference_config(config, overrides=overrides)
    act_tile_size = resolved_inf_cfg["tile_size"]
    act_stride = resolved_inf_cfg["stride"]
    act_norm_mode = resolved_inf_cfg["norm_mode"]
    include_aux = (resolved_inf_cfg["method"] == "watershed")

    test_dir = Path(test_images_dir)
    image_paths = sorted(
        list(test_dir.glob("*.jpeg"))
        + list(test_dir.glob("*.jpg"))
        + list(test_dir.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    # Deduplicate image files by canonical observation ID
    obs_to_img: Dict[str, Path] = {}
    for p in image_paths:
        obs_id = canonical_observation_id(p.stem)
        if obs_id not in obs_to_img:
            obs_to_img[obs_id] = p

    expected_obs_ids = set(obs_to_img.keys())
    print(f"[Inference] Discovered {len(expected_obs_ids)} unique test observations.")

    csv_rows = []
    manifest_obs = []

    for idx, (obs_id, img_path) in enumerate(sorted(obs_to_img.items()), start=1):
        if idx % 10 == 0 or idx == len(obs_to_img):
            print(f"[Inference] Processing {idx}/{len(obs_to_img)} ({obs_id})...", flush=True)
        img_rgb = load_solar_image(img_path)
        fg_map, ctr_map, bnd_map, off_map = predict_full_observation(
            model=model,
            image_rgb=img_rgb,
            device=device,
            tile_size=act_tile_size,
            stride=act_stride,
            tile_batch_size=tile_batch_size,
            norm_mode=act_norm_mode,
            include_aux=include_aux,
        )

        instances: List[InstancePrediction] = extract_instances_from_maps(
            foreground_prob=fg_map,
            center_prob=ctr_map if include_aux else None,
            boundary_prob=bnd_map if include_aux else None,
            offset_field=off_map if include_aux else None,
            obs_id=obs_id,
            high_threshold=resolved_inf_cfg["high_threshold"],
            low_threshold=resolved_inf_cfg["low_threshold"],
            min_area=resolved_inf_cfg["min_area"],
            method=resolved_inf_cfg["method"],
            center_threshold=resolved_inf_cfg["center_threshold"],
            boundary_weight=resolved_inf_cfg["boundary_weight"],
            marker_min_distance=resolved_inf_cfg["marker_min_distance"],
            marker_cap_per_component=resolved_inf_cfg["marker_cap_per_component"],
            max_peaks=resolved_inf_cfg["max_peaks"],
            max_instances=resolved_inf_cfg["max_instances"],
            shape=NATIVE_IMAGE_SHAPE,
        )

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

    # Write CSV with exact competition header
    out_csv_path = Path(output_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Compute SHA256 binding
    csv_sha256 = compute_file_sha256(out_csv_path)

    # Write manifest
    total_inst = len(csv_rows)
    proc_count = sum(1 for o in manifest_obs if o["status"] == "processed")
    abst_count = sum(1 for o in manifest_obs if o["status"] == "abstained")

    manifest_data = {
        "csv_sha256": csv_sha256,
        "total_test_observations": len(expected_obs_ids),
        "total_instances": total_inst,
        "detected_observations_count": proc_count,
        "abstained_observations_count": abst_count,
        "checkpoint_sha256": ckpt_sha256,
        "postprocess_params": {**resolved_inf_cfg, "tile_batch_size": tile_batch_size},
        "observations": manifest_obs,
    }

    out_man_path = Path(output_manifest)
    out_man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_man_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Audit submission through shared audit engine
    audit_report = audit_submission_and_manifest(
        csv_path=out_csv_path,
        manifest_path=out_man_path,
        expected_observation_ids=expected_obs_ids,
    )

    print(f"[Inference] Submission generated and audited successfully!")
    print(f"  CSV: {out_csv_path} ({total_inst} instances)")
    print(f"  Manifest: {out_man_path} (SHA256: {csv_sha256})")
    print(f"  Processed: {proc_count}, Abstained: {abst_count}")

    return audit_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full tiled inference and generate audited submission")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--test_images", type=str, default="data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images", help="Test images directory")
    parser.add_argument("--output_csv", type=str, default="artifacts/submission_resnet34.csv", help="Output submission CSV")
    parser.add_argument("--output_manifest", type=str, default="artifacts/submission_resnet34.manifest.json", help="Output submission manifest")
    parser.add_argument("--method", type=str, default=None, choices=["watershed", "connected_components"])
    parser.add_argument("--high-threshold", type=float, default=None)
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument("--center-threshold", type=float, default=None)
    parser.add_argument("--boundary-weight", type=float, default=None)
    parser.add_argument("--tile-size", type=int, default=None, help="Tiling window size")
    parser.add_argument("--stride", type=int, default=None, help="Tiling stride step")
    parser.add_argument("--tile-batch-size", type=int, default=16, help="Tile batch size for inference")
    parser.add_argument("--min-distance", type=int, default=None)
    parser.add_argument("--max-peaks", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        test_images_dir=args.test_images,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
        tile_size=args.tile_size,
        stride=args.stride,
        tile_batch_size=args.tile_batch_size,
        method=args.method,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        center_threshold=args.center_threshold,
        boundary_weight=args.boundary_weight,
        marker_min_distance=args.min_distance,
        max_peaks=args.max_peaks,
        max_instances=args.max_instances,
        min_area=args.min_area,
        device_str=args.device,
    )
