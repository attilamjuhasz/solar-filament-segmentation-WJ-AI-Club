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

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.dataset import load_solar_image
from src.data.folds import canonical_observation_id
from src.inference.rle import audit_submission_and_manifest
from src.inference.config import resolve_inference_config
from src.inference.engine import compute_file_sha256, predict_full_observation
from src.inference.instances import extract_instances_from_maps, InstancePrediction
from src.models import build_model


def compute_state_dict_sha256(state_dict: dict) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def main():
    start_time = time.time()
    ckpt_path = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    test_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    output_csv = Path("artifacts/submission_candidate_1_verified.csv")
    output_manifest = Path("artifacts/submission_candidate_1_verified.manifest.json")
    output_selection = Path("artifacts/submission_candidate_1_verified.selection_config.json")
    partitions_path = Path("artifacts/partitions_migrated_v1.json")

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not partitions_path.is_file():
        raise FileNotFoundError(f"Partitions file not found: {partitions_path}")

    # Checkpoint & Partition hashes
    ckpt_file_sha = compute_file_sha256(ckpt_path)
    partitions_sha = compute_file_sha256(partitions_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Generate] Loading checkpoint: {ckpt_path.name} on {device}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    ckpt_sd = ckpt["model_state_dict"]
    sd_sha = compute_state_dict_sha256(ckpt_sd)
    training_config = ckpt.get("config", {})
    training_cfg_sha = hashlib.sha256(json.dumps(training_config, sort_keys=True).encode("utf-8")).hexdigest()

    model = build_model(training_config).to(device)
    model.load_state_dict(ckpt_sd)
    model.eval()

    # Frozen candidate 1 configuration
    overrides = {
        "method": "connected_components",
        "high_threshold": 0.85,
        "low_threshold": 0.60,
        "min_area": 400,
        "max_instances": 12,
        "tile_size": 512,
        "stride": 256,
    }
    resolved_cfg = resolve_inference_config(training_config, overrides=overrides)
    resolved_cfg_sha = hashlib.sha256(json.dumps(resolved_cfg, sort_keys=True).encode("utf-8")).hexdigest()

    act_tile_size = resolved_cfg["tile_size"]
    act_stride = resolved_cfg["stride"]
    act_norm_mode = resolved_cfg["norm_mode"]
    include_aux = (resolved_cfg["method"] == "watershed")  # False for CC

    # Discover and deduplicate test images
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

    expected_obs_ids = set(obs_to_img.keys())
    print(f"[Generate] Discovered {len(expected_obs_ids)} unique test observations.")
    assert len(expected_obs_ids) == 180, f"Expected exactly 180 test observations, found {len(expected_obs_ids)}"

    csv_rows = []
    manifest_obs = []
    cache_dir = Path("artifacts/.test_cache_candidate_1_verified")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Generate] Running test inference (tile={act_tile_size}, stride={act_stride}, norm={act_norm_mode}, batch=16)...")
    for idx, (obs_id, img_path) in enumerate(sorted(obs_to_img.items()), start=1):
        cache_item = cache_dir / f"{obs_id}.json"
        if cache_item.is_file():
            with open(cache_item, "r", encoding="utf-8") as f:
                inst_data = json.load(f)
            elapsed = 0.0
        else:
            t0 = time.time()
            img_rgb = load_solar_image(img_path)
            fg_map, ctr_map, bnd_map, off_map = predict_full_observation(
                model=model,
                image_rgb=img_rgb,
                device=device,
                tile_size=act_tile_size,
                stride=act_stride,
                tile_batch_size=16,
                norm_mode=act_norm_mode,
                include_aux=include_aux,
            )

            instances = extract_instances_from_maps(
                foreground_prob=fg_map,
                center_prob=ctr_map if include_aux else None,
                boundary_prob=bnd_map if include_aux else None,
                offset_field=off_map if include_aux else None,
                obs_id=obs_id,
                high_threshold=resolved_cfg["high_threshold"],
                low_threshold=resolved_cfg["low_threshold"],
                min_area=resolved_cfg["min_area"],
                method=resolved_cfg["method"],
                center_threshold=resolved_cfg["center_threshold"],
                boundary_weight=resolved_cfg["boundary_weight"],
                marker_min_distance=resolved_cfg["marker_min_distance"],
                marker_cap_per_component=resolved_cfg["marker_cap_per_component"],
                max_peaks=resolved_cfg["max_peaks"],
                max_instances=resolved_cfg["max_instances"],
                shape=NATIVE_IMAGE_SHAPE,
            )

            inst_data = [{"filament_id": inst.filament_id, "segmentation_rle": inst.rle_counts} for inst in instances]
            with open(cache_item, "w", encoding="utf-8") as f:
                json.dump(inst_data, f)
            elapsed = time.time() - t0

        if len(inst_data) > 0:
            manifest_obs.append({
                "observation_id": obs_id,
                "status": "processed",
                "instance_count": len(inst_data),
            })
            for inst in inst_data:
                csv_rows.append({
                    "filament_id": inst["filament_id"],
                    "segmentation_rle": inst["segmentation_rle"],
                })
        else:
            manifest_obs.append({
                "observation_id": obs_id,
                "status": "abstained",
                "instance_count": 0,
            })

        if idx % 15 == 0 or idx == len(obs_to_img):
            print(f"  [{idx:03d}/{len(obs_to_img)}] {obs_id}: {len(inst_data)} instances ({elapsed:.2f}s, total insts: {len(csv_rows)})", flush=True)

    # Write submission CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(csv_rows)

    csv_sha = compute_file_sha256(output_csv)

    proc_count = sum(1 for o in manifest_obs if o["status"] == "processed")
    abst_count = sum(1 for o in manifest_obs if o["status"] == "abstained")
    total_inst = len(csv_rows)

    # Load validation reports
    tuning_rescore_path = Path("artifacts/reports/immutable_epochs_tuning_rescore.json")
    conf_report_path = Path("artifacts/reports/eval_selected_epoch06_confirmation.json")
    with open(tuning_rescore_path, "r", encoding="utf-8") as f:
        tuning_data = json.load(f)
    with open(conf_report_path, "r", encoding="utf-8") as f:
        conf_data = json.load(f)

    # Construct comprehensive manifest
    manifest_data = {
        "manifest_version": "1.0.0",
        "csv_file": output_csv.name,
        "csv_sha256": csv_sha,
        "total_test_observations": len(expected_obs_ids),
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
            "resolved_config": resolved_cfg,
            "resolved_config_sha256": resolved_cfg_sha,
            "tile_size": act_tile_size,
            "stride": act_stride,
            "norm_mode": act_norm_mode,
            "precision": "float32_inference",
        },
        "validation_evidence": {
            "tuning": {
                "neural_epoch6": {
                    "pq": tuning_data["selected_best_epoch"]["pq"],
                    "sq": tuning_data["selected_best_epoch"]["sq"],
                    "rq": tuning_data["selected_best_epoch"]["rq"],
                    "mean_dice": tuning_data["selected_best_epoch"]["mean_dice"],
                    "tp": tuning_data["selected_best_epoch"]["tp"],
                    "fp": tuning_data["selected_best_epoch"]["fp"],
                    "fn": tuning_data["selected_best_epoch"]["fn"],
                },
                "heuristic_baseline": {
                    "pq": 0.0368,
                    "mean_dice": 0.0615,
                    "tp": 12,
                    "fp": 192,
                    "fn": 179,
                },
                "margin_over_heuristic": tuning_data["selected_best_epoch"]["pq"] - 0.0368,
            },
            "confirmation": {
                "neural_epoch6": {
                    "pq": conf_data["overall"]["pq"],
                    "sq": conf_data["overall"]["sq"],
                    "rq": conf_data["overall"]["rq"],
                    "mean_dice": conf_data["overall"]["mean_dice"],
                    "tp": conf_data["overall"]["tp"],
                    "fp": conf_data["overall"]["fp"],
                    "fn": conf_data["overall"]["fn"],
                },
                "heuristic_baseline": {
                    "pq": 0.1073,
                    "mean_dice": 0.0399,
                    "tp": 20,
                    "fp": 81,
                    "fn": 110,
                },
                "margin_over_heuristic": conf_data["overall"]["pq"] - 0.1073,
                "history_disclosure": "previously inspected validation partition with 9 unique physical observations",
            },
        },
        "observations": manifest_obs,
    }

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Save selection config & reproduction command
    selection_config = {
        "candidate_id": "candidate_1_verified",
        "description": "Verified replacement package for Candidate 1 based on frozen CC settings and highest strict tuning PQ epoch",
        "checkpoint_file": str(ckpt_path),
        "checkpoint_sha256": ckpt_file_sha,
        "model_state_dict_sha256": sd_sha,
        "resolved_inference_config": resolved_cfg,
        "reproduction_commands": {
            "evaluation_tuning": "python scripts/rescore_immutable_epochs.py",
            "evaluation_confirmation": "python scripts/eval_epoch6_confirmation.py",
            "test_inference": (
                "python inference.py --checkpoint artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt "
                "--output_csv artifacts/submission_candidate_1_verified.csv "
                "--output_manifest artifacts/submission_candidate_1_verified.manifest.json "
                "--method connected_components --high-threshold 0.85 --low-threshold 0.60 "
                "--min-area 400 --max-instances 12 --tile-size 512 --stride 256"
            ),
        },
    }
    with open(output_selection, "w", encoding="utf-8") as f:
        json.dump(selection_config, f, indent=2)

    print("\n[Generate] Running strict audit against all 180 expected test observations...")
    audit = audit_submission_and_manifest(
        csv_path=output_csv,
        manifest_path=output_manifest,
        expected_observation_ids=expected_obs_ids,
    )

    total_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"Candidate 1 Replacement Generation & Audit Complete!")
    print(f"  Time Elapsed: {total_time:.1f}s")
    print(f"  CSV Path: {output_csv} (SHA256: {csv_sha})")
    print(f"  Manifest Path: {output_manifest}")
    print(f"  Selection Config: {output_selection}")
    print(f"  Total Instances: {total_inst} across {proc_count} observations ({abst_count} abstentions)")
    print(f"  Strict Native Audit: {'PASSED' if audit['is_valid'] else 'FAILED'}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
