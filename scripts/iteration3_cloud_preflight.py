from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.contracts import NATIVE_IMAGE_SHAPE
from src.inference.config import resolve_inference_config
from src.inference.engine import compute_file_sha256, compute_state_dict_sha256
from src.models import build_model
from src.losses import CompoundTopologyLoss
from src.data.dataset import SolarFilamentDataset
from src.data.annotations import load_coco_annotations
from src.data.folds import load_frozen_folds_manifest
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
    verify_immutable_inputs,
)
import yaml


def run_cloud_preflight(
    parent_checkpoint: str = "artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt",
    folds_manifest: str = "artifacts/folds_manifest.json",
    partitions: str = "artifacts/partitions_migrated_v1.json",
    mining_bank: str = "artifacts/reports/mining_bank_v1.json",
    config_path: str = "configs/iteration3_resnet34.yaml",
    train_images: str = "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    train_json: str = "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    run_benchmark: bool = False,
) -> Dict[str, Any]:
    """Execute complete read-only cloud preflight check and optional short benchmark."""
    root_dir = Path(__file__).resolve().parent.parent
    parent_p = root_dir / parent_checkpoint
    folds_p = root_dir / folds_manifest
    parts_p = root_dir / partitions
    bank_p = root_dir / mining_bank
    cfg_p = root_dir / config_path
    train_img_p = root_dir / train_images
    train_json_p = root_dir / train_json

    print("==================================================================")
    print("=== ITERATION 3 CLOUD READINESS PREFLIGHT & HARDWARE AUDIT ===")
    print("==================================================================")

    # 1. Hardware & Environment Inspection
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    device_count = torch.cuda.device_count() if cuda_avail else 0
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0

    print(f"[Hardware] CUDA Available: {cuda_avail} | Devices: {device_count} | Device 0: {device_name} ({vram_gb:.2f} GB VRAM)")
    print(f"[Platform] Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | NumPy: {np.__version__}")

    # 2. Immutable Cryptographic Input Hashes
    input_hashes = verify_immutable_inputs(
        parent_path=parent_p,
        folds_manifest_path=folds_p,
        partitions_path=parts_p,
        mining_bank_path=bank_p if bank_p.is_file() else None,
    )

    # 3. Inference Configuration Validation
    with open(cfg_p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolved_inf_cfg = resolve_inference_config(config)
    expected_inf = {
        "method": FIXED_INF_METHOD,
        "high_threshold": FIXED_HIGH_THRESH,
        "low_threshold": FIXED_LOW_THRESH,
        "min_area": FIXED_MIN_AREA,
        "max_instances": FIXED_MAX_INSTANCES,
    }
    for k, v in expected_inf.items():
        if resolved_inf_cfg.get(k) != v:
            raise ValueError(f"Inference config mismatch: {k}={resolved_inf_cfg.get(k)}, expected {v}")
    print(f"[Config] Resolved inference config verified: {resolved_inf_cfg['method']} (high={resolved_inf_cfg['high_threshold']}, low={resolved_inf_cfg['low_threshold']}, area={resolved_inf_cfg['min_area']}, cap={resolved_inf_cfg['max_instances']})")

    # 4. Dataset Shapes & Pipeline Integrity
    fold_assignments, _ = load_frozen_folds_manifest(folds_p)
    annotation_index = load_coco_annotations(train_json_p)

    train_ds = SolarFilamentDataset(
        images_dir=train_img_p,
        annotation_index=annotation_index,
        fold_assignments=fold_assignments,
        target_fold=0,
        is_train=True,
        patch_size=(512, 512),
        norm_mode="imagenet",
        max_observations=10,
        use_geometric_augmentation=True,
    )
    sample = train_ds[0]
    assert tuple(sample["image"].shape) == (3, 512, 512), f"Train sample image shape mismatch: {sample['image'].shape}"
    assert tuple(sample["target_fg"].shape) in ((512, 512), (1, 512, 512)), f"Train sample fg shape mismatch: {sample['target_fg'].shape}"
    print(f"[Dataset] Verified training crop shape: {tuple(sample['image'].shape)}, target: {tuple(sample['target_fg'].shape)}")

    # 5. Optional Benchmark (disposable subset)
    benchmark_results: Optional[Dict[str, Any]] = None
    if run_benchmark:
        print("\n=== RUNNING DISPOSABLE TRAINING BENCHMARK (4 Steps) ===")
        device = torch.device("cuda" if cuda_avail else "cpu")
        loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
        model = build_model(config).to(device)
        criterion = CompoundTopologyLoss(alpha=1.0, beta=1.0, gamma=0.0, w_bnd=0.0, w_ctr=0.0, w_off=0.0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

        if cuda_avail:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        step_times = []
        input_wait_times = []
        compute_times = []

        t_last = time.time()
        successful_updates = 0

        for b_idx, batch in enumerate(loader):
            if b_idx >= 4:
                break
            t_data = time.time() - t_last
            input_wait_times.append(t_data)

            t_compute_start = time.time()
            imgs = batch["image"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "image"}

            preds = model(imgs)
            loss, stats = criterion(
                fg_logits=preds["fg_logits"].float(),
                bnd_logits=preds["bnd_logits"].float(),
                ctr_logits=preds["ctr_logits"].float(),
                off_pred=preds["off_pred"].float(),
                target_fg=targets["target_fg"].float(),
                target_skel=targets["target_skel"].float(),
                target_bnd=targets["target_bnd"].float(),
                target_ctr=targets["target_ctr"].float(),
                target_off=targets["target_off"].float(),
                valid_mask=targets.get("valid_mask"),
            )
            loss.backward()

            if (b_idx + 1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()
                successful_updates += 1

            if cuda_avail:
                torch.cuda.synchronize()
            t_comp = time.time() - t_compute_start
            compute_times.append(t_comp)
            step_times.append(t_data + t_comp)
            t_last = time.time()

        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if cuda_avail else 0.0
        avg_wait = float(np.mean(input_wait_times))
        avg_compute = float(np.mean(compute_times))
        avg_total = float(np.mean(step_times))
        samples_per_sec = float((2 * len(step_times)) / sum(step_times)) if sum(step_times) > 0 else 0.0

        benchmark_results = {
            "num_steps": len(step_times),
            "avg_input_wait_seconds": round(avg_wait, 4),
            "avg_compute_seconds": round(avg_compute, 4),
            "avg_step_seconds": round(avg_total, 4),
            "samples_per_sec": round(samples_per_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "successful_updates": successful_updates,
            "disposable_notice": "Benchmark run was disposable; weights discarded and ineligible for selection.",
        }
        print(f"[Benchmark] Peak VRAM: {peak_vram_mb:.1f} MB | Samples/sec: {samples_per_sec:.2f} | Avg Step: {avg_total*1000:.1f}ms (Wait: {avg_wait*1000:.1f}ms, Compute: {avg_compute*1000:.1f}ms)")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cuda_available": cuda_avail,
            "device_name": device_name,
            "device_count": device_count,
            "vram_gb": round(vram_gb, 2),
            "pytorch_version": torch.__version__,
        },
        "input_hashes": input_hashes,
        "resolved_inference_config": resolved_inf_cfg,
        "preflight_status": "READY",
        "benchmark": benchmark_results,
    }

    out_p = root_dir / "artifacts" / "reports" / "cloud_preflight_report.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Preflight Complete] Machine-readable report saved to {out_p}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only Cloud Readiness Preflight Check and Hardware Audit")
    parser.add_argument("--benchmark", action="store_true", help="Run short disposable 4-step benchmark")
    args = parser.parse_args()

    run_cloud_preflight(run_benchmark=args.benchmark)
