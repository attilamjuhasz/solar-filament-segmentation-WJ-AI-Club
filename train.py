from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.annotations import load_coco_annotations
from src.data.dataset import SolarFilamentDataset
from src.data.folds import (
    ObservationRecord,
    assign_stratified_group_folds,
    consolidate_canonical_records,
    freeze_folds_manifest,
    load_frozen_folds_manifest,
    save_folds_manifest,
    verify_fold_isolation,
)
from src.data.manifest import canonical_observation_id
from src.losses import CompoundTopologyLoss
from src.models import build_model
from src.training.finetune import setup_finetune_state, validate_finetune_parent


def set_seed(seed: int = 2026) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_fold_records(annotation_index) -> List[ObservationRecord]:
    """Convert annotation index into fold ObservationRecords."""
    raw_records = []
    for obs_id, variants in annotation_index.by_observation.items():
        all_var_ids = [v.annotator_image_id for v in variants]
        inst_counts = [len(v.instances) for v in variants]
        areas = [sum(inst.area for inst in v.instances) for v in variants]
        raw_records.append(
            ObservationRecord(
                observation_id=obs_id,
                annotator_variants=all_var_ids,
                median_instance_count=float(np.median(inst_counts)) if inst_counts else 0.0,
                median_mask_area=float(np.median(areas)) if areas else 0.0,
            )
        )
    return consolidate_canonical_records(raw_records)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompoundTopologyLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    grad_accum_steps: int = 1,
    max_steps: Optional[int] = None,
    deadline_time: Optional[float] = None,
) -> Dict[str, float]:
    """Execute one training epoch with verified partial gradient accumulation and finite gradient safety."""
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive integer, got {grad_accum_steps}")
    if max_steps is not None and max_steps <= 0:
        raise ValueError(f"max_steps must be positive integer, got {max_steps}")

    model.train()
    optimizer.zero_grad()

    epoch_stats = {
        "loss_total": 0.0,
        "loss_seg": 0.0,
        "loss_bce": 0.0,
        "loss_dice": 0.0,
        "loss_cldice": 0.0,
        "loss_boundary": 0.0,
        "loss_center": 0.0,
        "loss_offset": 0.0,
    }
    num_batches = 0
    successful_updates = 0
    skipped_updates = 0
    total_loader_batches = len(loader)
    effective_total_batches = min(total_loader_batches, max_steps) if (max_steps is not None and max_steps > 0) else total_loader_batches

    for step, batch in enumerate(loader):
        if deadline_time is not None and time.time() >= deadline_time:
            raise TimeoutError(f"Budget deadline {deadline_time} exceeded during training at step {step}.")

        if max_steps is not None and step >= max_steps:
            break

        img = batch["image"].to(device, non_blocking=True)
        tgt_fg = batch["target_fg"].to(device, non_blocking=True)
        tgt_bnd = batch["target_bnd"].to(device, non_blocking=True)
        tgt_skel = batch["target_skel"].to(device, non_blocking=True)
        tgt_ctr = batch["target_ctr"].to(device, non_blocking=True)
        tgt_off = batch["target_off"].to(device, non_blocking=True)
        val_mask = batch["valid_mask"].to(device, non_blocking=True)

        use_amp = (device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            preds = model(img)

        # Multi-task loss in FP32 outside autocast for total topological and reduction safety
        loss, stats = criterion(
            fg_logits=preds["fg_logits"].float(),
            bnd_logits=preds["bnd_logits"].float(),
            ctr_logits=preds["ctr_logits"].float(),
            off_pred=preds["off_pred"].float(),
            target_fg=tgt_fg,
            target_skel=tgt_skel,
            target_bnd=tgt_bnd,
            target_ctr=tgt_ctr,
            target_off=tgt_off,
            valid_mask=val_mask,
        )

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered at step {step}: {stats}")

        # Correct normalization for partial accumulation groups across effective batch length
        current_window = min(grad_accum_steps, effective_total_batches - (step // grad_accum_steps) * grad_accum_steps)
        scaled_loss = loss / current_window

        is_accum_boundary = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == effective_total_batches)

        if scaler is not None and use_amp and (amp_dtype == torch.float16):
            scaler.scale(scaled_loss).backward()
            if is_accum_boundary:
                scaler.unscale_(optimizer)
                grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if grads_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    successful_updates += 1
                else:
                    print(f"[Warning] Non-finite gradients at step {step}, skipping update.")
                    skipped_updates += 1
                scaler.update()
                optimizer.zero_grad()
                if deadline_time is not None and time.time() >= deadline_time:
                    raise TimeoutError(f"Budget deadline {deadline_time} exceeded during training at update boundary (step {step}).")
        else:
            scaled_loss.backward()
            if is_accum_boundary:
                grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if grads_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    successful_updates += 1
                else:
                    print(f"[Warning] Non-finite gradients at step {step}, skipping update.")
                    skipped_updates += 1
                optimizer.zero_grad()
                if deadline_time is not None and time.time() >= deadline_time:
                    raise TimeoutError(f"Budget deadline {deadline_time} exceeded during training at update boundary (step {step}).")

        for k, v in stats.items():
            epoch_stats[k] += v
        num_batches += 1

    if num_batches > 0:
        for k in epoch_stats:
            epoch_stats[k] /= num_batches

    epoch_stats["successful_updates"] = float(successful_updates)
    epoch_stats["skipped_updates"] = float(skipped_updates)
    return epoch_stats


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    fold: int = 0,
    best_metric_name: str = "val_pq",
    best_metric: float = 0.0,
    train_observations: Optional[List[str]] = None,
    val_observations: Optional[List[str]] = None,
    folds_manifest_sha256: str = "",
    global_step: int = 0,
    val_stats: Optional[Dict[str, Any]] = None,
    dataset_rng_state: Optional[Any] = None,
    dataset_aug_rng_state: Optional[Any] = None,
    successful_updates: int = 0,
    skipped_updates: int = 0,
    run_id: str = "",
    runtime_config: Optional[Dict[str, Any]] = None,
    finetune_parent: Optional[Dict[str, Any]] = None,
    mining_bank_sha256: Optional[str] = None,
    total_epochs: Optional[int] = None,
    eligible_for_promotion: bool = True,
) -> None:
    """Save full training checkpoint with all weights, optimizer states, RNGs, and provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "random_rng_state": random.getstate(),
        "dataset_rng_state": dataset_rng_state,
        "dataset_aug_rng_state": dataset_aug_rng_state,
        "successful_updates": successful_updates,
        "skipped_updates": skipped_updates,
        "config": config or {},
        "runtime_config": runtime_config or {},
        "run_id": run_id,
        "fold": fold,
        "metric_name": best_metric_name,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
        "val_stats": val_stats,
        "train_observations": train_observations if train_observations is not None else [],
        "val_observations": val_observations if val_observations is not None else [],
        "folds_manifest_sha256": folds_manifest_sha256,
        "finetune_parent": finetune_parent,
        "mining_bank_sha256": mining_bank_sha256,
        "total_epochs": total_epochs,
        "eligible_for_promotion": eligible_for_promotion,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, str(path))


def train(
    config_path: str = "configs/b0_resnet34.yaml",
    fold: int = 0,
    smoke: bool = False,
    max_steps: Optional[int] = None,
    max_observations: Optional[int] = None,
    max_val_observations: Optional[int] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    device_str: Optional[str] = None,
    resume_path: Optional[str] = None,
    finetune_from: Optional[str] = None,
    mining_bank_path: Optional[str] = None,
    augment_flips: bool = False,
    deadline_time: Optional[float] = None,
) -> Tuple[Path, Path]:
    """Config-driven training loop with validation PQ checkpoint promotion, resumption, and fine-tuning."""
    if resume_path and finetune_from:
        raise ValueError(
            "Cannot specify both resume_path and finetune_from. "
            "Resume restores prior optimizer/scheduler states; fine-tuning initializes fresh optimizer/schedule from parent weights."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 2026)
    set_seed(seed)

    device_name = device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    # Locate dataset paths
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    train_images = data_dir / "train" / "train_images"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    if not train_json.is_file():
        raise FileNotFoundError(f"Annotations file not found at {train_json}")

    print(f"[Train] Loading annotations from {train_json}...")
    annotation_index = load_coco_annotations(str(train_json))
    records = build_fold_records(annotation_index)

    # Load or freeze reproducible fold split manifest
    manifest_file = root_dir / "artifacts" / "folds_manifest.json"
    if manifest_file.is_file():
        fold_assignments, folds_manifest_sha256 = load_frozen_folds_manifest(manifest_file, verify_annotations_path=train_json)
    else:
        fold_assignments, folds_manifest_sha256 = freeze_folds_manifest(
            records, train_json, root_dir / "artifacts", n_splits=5, seed=seed
        )

    patch_size = tuple(config.get("data", {}).get("training_size", [512, 512]))
    max_train_obs = 10 if (smoke and max_observations is None) else max_observations

    norm_mode = config.get("data", {}).get("norm_mode", "imagenet")
    fg_crop_prob = float(config.get("data", {}).get("fg_crop_prob", 0.70))

    # Optional hard negative mining bank
    mining_boxes = None
    crop_policy = "standard"
    active_bank_sha = None
    if mining_bank_path:
        mining_bank_file = Path(mining_bank_path)
        if not mining_bank_file.is_file():
            raise FileNotFoundError(f"Mining bank file not found: {mining_bank_file}")

        from src.inference.engine import compute_file_sha256
        active_bank_sha = compute_file_sha256(mining_bank_file)

        # Independently verify mining bank provenance before loading
        from src.training.mining import verify_mining_bank_provenance
        bank_meta = verify_mining_bank_provenance(
            bank_path=mining_bank_file,
            expected_parent_sha="9580632d5de717999bb1a60ee940e3f14ee716e853d87d1220be456db62d344a",
            expected_folds_manifest_sha=folds_manifest_sha256,
            folds_manifest_path=root_dir / "artifacts" / "folds_manifest.json",
            partitions_path=root_dir / "artifacts" / "partitions_migrated_v1.json",
        )
        print(f"[Train] Verified mining bank provenance: {bank_meta['total_crop_boxes']} boxes across {bank_meta['observations_with_crops']} observations.")

        with open(mining_bank_file, "r", encoding="utf-8") as f:
            mining_bank_json = json.load(f)
        mining_boxes = mining_bank_json.get("mining_boxes_by_image", mining_bank_json)
        crop_policy = "mining_bank"
        print(f"[Train] Loaded verified mining bank from {mining_bank_file} ({len(mining_boxes)} images with mining boxes)")

    train_ds = SolarFilamentDataset(
        images_dir=train_images,
        annotation_index=annotation_index,
        fold_assignments=fold_assignments,
        target_fold=fold,
        is_train=True,
        patch_size=patch_size,
        norm_mode=norm_mode,
        fg_crop_prob=fg_crop_prob,
        seed=seed,
        max_observations=max_train_obs,
        mining_bank=mining_boxes,
        crop_policy=crop_policy,
        use_geometric_augmentation=augment_flips,
    )

    val_limit = 5 if (smoke and max_val_observations is None) else max_val_observations
    val_ds = SolarFilamentDataset(
        images_dir=train_images,
        annotation_index=annotation_index,
        fold_assignments=fold_assignments,
        target_fold=fold,
        is_train=False,
        patch_size=(2048, 2048),
        norm_mode=norm_mode,
        seed=seed,
        max_observations=val_limit,
    )

    batch_size = batch_size_override if batch_size_override else (2 if smoke else config.get("training", {}).get("batch_size", 4))
    effective_grad_accum_steps = 2 if (device.type == "cuda" and batch_size < 4) else 1
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Build model
    model = build_model(config).to(device)

    # Loss configuration with explicit zeroes for disabled terms
    loss_cfg = config.get("loss", {})
    criterion = CompoundTopologyLoss(
        alpha=float(loss_cfg.get("bce_weight", 0.35)),
        beta=float(loss_cfg.get("dice_weight", 0.45)),
        gamma=float(loss_cfg.get("cldice_weight", 0.20)),
        w_bnd=float(loss_cfg.get("boundary_weight", 0.10)),
        w_ctr=float(loss_cfg.get("center_weight", 0.05)),
        w_off=float(loss_cfg.get("offset_weight", 0.05)),
        cldice_iters=5 if smoke else 15,
    ).to(device)

    finetune_parent_summary: Optional[Dict[str, Any]] = None
    is_diag = smoke or (max_train_obs is not None) or (val_limit is not None)

    if finetune_from:
        print(f"[Train] Setting up fine-tuning from parent checkpoint {finetune_from} (diagnostic={is_diag})...")
        parent_meta = validate_finetune_parent(
            checkpoint_path=Path(finetune_from),
            expected_fold=fold,
            expected_manifest_sha=folds_manifest_sha256,
            train_observations=train_ds.observations,
            val_observations=val_ds.observations,
            is_diagnostic=is_diag,
        )
        finetune_parent_summary = {
            "path": parent_meta["path"],
            "file_sha256": parent_meta["file_sha256"],
            "model_state_sha256": parent_meta["model_state_sha256"],
            "training_config_sha256": parent_meta["training_config_sha256"],
            "parent_epoch": parent_meta["parent_epoch"],
            "fold": parent_meta["fold"],
            "folds_manifest_sha256": parent_meta["folds_manifest_sha256"],
            "is_diagnostic": is_diag,
        }
        finetune_epochs = epochs_override if epochs_override is not None else 3
        lr = float(config.get("finetune", {}).get("lr", 5e-5))
        weight_decay = float(config.get("finetune", {}).get("weight_decay", 1e-4))
        eta_min = float(config.get("finetune", {}).get("eta_min", 1e-6))
        optimizer, scheduler, scaler = setup_finetune_state(
            model=model,
            parent_state_dict=parent_meta["model_state_dict"],
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            total_epochs=finetune_epochs,
            eta_min=eta_min,
        )
        total_epochs = finetune_epochs
        start_epoch = 1
        global_step = 0
        best_pq = -1.0
        print(f"[Train] Fine-tuning initialized: AdamW lr={lr}, CosineAnnealingLR T_max={total_epochs}, eta_min={eta_min}")
    else:
        lr = float(config.get("training", {}).get("lr", 3e-4))
        weight_decay = float(config.get("training", {}).get("weight_decay", 1e-4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_epochs = epochs_override if epochs_override is not None else (1 if smoke else int(config.get("training", {}).get("epochs", 10)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda") else None
        start_epoch = 1
        global_step = 0
        best_pq = -1.0
    best_metric_name = "diagnostic_val_pq" if val_limit is not None else "val_pq"

    total_successful_updates = 0
    total_skipped_updates = 0

    # Resume from checkpoint if requested
    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
        print(f"[Train] Resuming training state from {resume_file}...")
        ckpt = torch.load(str(resume_file), map_location=device, weights_only=False)

        ckpt_fold = ckpt.get("fold")
        if ckpt_fold is None or ckpt_fold != fold:
            raise ValueError(
                f"Resume fold mismatch: checkpoint was trained on fold {ckpt_fold}, requested fold {fold}"
            )
        ckpt_man = ckpt.get("folds_manifest_sha256")
        if not ckpt_man or ckpt_man != folds_manifest_sha256:
            raise ValueError(
                f"Resume manifest mismatch: checkpoint used manifest {ckpt_man}, active is {folds_manifest_sha256}"
            )

        # 1. Exact Canonical Train and Validation Observation Membership Verification
        ckpt_train_obs = ckpt.get("train_observations")
        if not ckpt_train_obs or sorted(ckpt_train_obs) != sorted(train_ds.observations):
            raise ValueError(
                f"Resume train observations mismatch or missing from checkpoint: ckpt has {len(ckpt_train_obs) if ckpt_train_obs else 0}, active has {len(train_ds.observations)}"
            )
        ckpt_val_obs = ckpt.get("val_observations")
        if not ckpt_val_obs or sorted(ckpt_val_obs) != sorted(val_ds.observations):
            raise ValueError(
                f"Resume val observations mismatch or missing from checkpoint: ckpt has {len(ckpt_val_obs) if ckpt_val_obs else 0}, active has {len(val_ds.observations)}"
            )

        # 2. Mining Bank SHA Verification (Exact Match or Both None)
        ckpt_bank_sha = ckpt.get("mining_bank_sha256")
        if ckpt_bank_sha is None and "runtime_config" in ckpt:
            ckpt_bank_sha = ckpt["runtime_config"].get("mining_bank_sha256")
        if active_bank_sha != ckpt_bank_sha:
            raise ValueError(
                f"Resume mining bank SHA mismatch: ckpt has '{ckpt_bank_sha}', active has '{active_bank_sha}'"
            )

        # 3. Runtime Configuration Verification
        rc = ckpt.get("runtime_config", {})
        if rc.get("augment_flips") != augment_flips:
            raise ValueError(
                f"Resume augmentation mismatch: ckpt has augment_flips={rc.get('augment_flips')}, active is {augment_flips}"
            )
        if rc.get("crop_policy") != crop_policy:
            raise ValueError(
                f"Resume crop policy mismatch: ckpt has crop_policy={rc.get('crop_policy')}, active is {crop_policy}"
            )
        ckpt_total_epochs = rc.get("total_epochs") or ckpt.get("total_epochs")
        if ckpt_total_epochs is None:
            raise ValueError("Resume checkpoint missing schedule horizon (total_epochs)")
        if ckpt_total_epochs != total_epochs:
            raise ValueError(
                f"Resume schedule horizon mismatch: ckpt total_epochs={ckpt_total_epochs}, requested {total_epochs}"
            )

        # Batch Size and Accumulation Verification
        ckpt_batch_size = rc.get("batch_size") or ckpt.get("batch_size") or (ckpt_cfg.get("training", {}).get("batch_size") if ckpt_cfg else None)
        if ckpt_batch_size is not None and ckpt_batch_size != batch_size:
            raise ValueError(
                f"Resume batch size mismatch: ckpt has batch_size={ckpt_batch_size}, active is {batch_size}"
            )

        effective_grad_accum_steps = 2 if (device.type == "cuda" and batch_size < 4) else 1
        ckpt_accum = rc.get("grad_accum_steps") or ckpt.get("grad_accum_steps")
        if ckpt_accum is not None and ckpt_accum != effective_grad_accum_steps:
            raise ValueError(
                f"Resume gradient accumulation mismatch: ckpt has grad_accum_steps={ckpt_accum}, active is {effective_grad_accum_steps}"
            )

        # 4. Recipe / Config Identity
        ckpt_cfg = ckpt.get("config", {})
        if ckpt_cfg:
            if ckpt_cfg.get("model", {}).get("name") != config.get("model", {}).get("name"):
                raise ValueError(
                    f"Resume model mismatch: ckpt has {ckpt_cfg.get('model', {}).get('name')}, active has {config.get('model', {}).get('name')}"
                )
            cfg_loss = config.get("loss", {})
            ckpt_loss = ckpt_cfg.get("loss", {})
            for loss_k in ["bce_weight", "dice_weight", "cldice_weight", "boundary_weight", "center_weight", "offset_weight"]:
                if float(cfg_loss.get(loss_k, 0.0)) != float(ckpt_loss.get(loss_k, 0.0)):
                    raise ValueError(
                        f"Resume loss config mismatch for {loss_k}: ckpt has {ckpt_loss.get(loss_k)}, active has {cfg_loss.get(loss_k)}"
                    )

            # Verify optimizer recipe from config
            active_opt_cfg = config.get("finetune" if finetune_from else "training", {})
            ckpt_opt_cfg = ckpt_cfg.get("finetune" if finetune_from else "training", {})
            if ckpt_opt_cfg:
                for opt_k in ["optimizer", "lr", "weight_decay"]:
                    if opt_k in ckpt_opt_cfg and opt_k in active_opt_cfg:
                        v_ckpt = ckpt_opt_cfg[opt_k]
                        v_curr = active_opt_cfg[opt_k]
                        if isinstance(v_curr, (int, float)):
                            if abs(float(v_ckpt) - float(v_curr)) > 1e-9:
                                raise ValueError(
                                    f"Resume optimizer recipe mismatch for {opt_k}: ckpt has {v_ckpt}, active has {v_curr}"
                                )
                        elif str(v_ckpt) != str(v_curr):
                            raise ValueError(
                                f"Resume optimizer recipe mismatch for {opt_k}: ckpt has {v_ckpt}, active has {v_curr}"
                            )

        # Verify optimizer state completeness and parameter group hyperparams
        if "optimizer_state_dict" not in ckpt or not ckpt["optimizer_state_dict"]:
            raise ValueError("Resume checkpoint missing valid optimizer_state_dict")

        ckpt_opt_state = ckpt["optimizer_state_dict"]
        if "param_groups" in ckpt_opt_state:
            ckpt_groups = ckpt_opt_state["param_groups"]
            curr_groups = optimizer.param_groups
            if len(ckpt_groups) != len(curr_groups):
                raise ValueError(
                    f"Resume optimizer param groups mismatch: ckpt has {len(ckpt_groups)}, active has {len(curr_groups)}"
                )
            for g_idx, (g_ckpt, g_curr) in enumerate(zip(ckpt_groups, curr_groups)):
                for hparam in ["lr", "weight_decay"]:
                    if hparam in g_ckpt and hparam in g_curr:
                        val_c = g_ckpt[hparam]
                        val_a = g_curr[hparam]
                        if abs(float(val_c) - float(val_a)) > 1e-9:
                            raise ValueError(
                                f"Resume optimizer recipe mismatch in param group {g_idx} for {hparam}: ckpt has {val_c}, active has {val_a}"
                            )

        # 5. Updates and Finiteness
        if "successful_updates" not in ckpt or not isinstance(ckpt["successful_updates"], int):
            raise ValueError("Resume checkpoint missing valid successful_updates metadata")
        total_successful_updates = int(ckpt["successful_updates"])
        total_skipped_updates = int(ckpt.get("skipped_updates", 0))
        if total_skipped_updates > 0:
            raise ValueError(
                f"Cannot resume from checkpoint with {total_skipped_updates} cumulative skipped updates"
            )
        all_finite = all(torch.isfinite(val).all().item() for val in ckpt["model_state_dict"].values())
        if not all_finite:
            raise ValueError("Cannot resume from checkpoint with non-finite model weights")

        # 6. RNG State Completeness
        for rng_key in ["torch_rng_state", "numpy_rng_state", "random_rng_state"]:
            if rng_key not in ckpt or ckpt[rng_key] is None:
                raise ValueError(f"Resume checkpoint missing RNG state: {rng_key}")
        if hasattr(train_ds, "rng") and ckpt.get("dataset_rng_state") is None:
            raise ValueError("Resume checkpoint missing dataset_rng_state")
        if augment_flips and hasattr(train_ds, "aug_rng") and ckpt.get("dataset_aug_rng_state") is None:
            raise ValueError("Resume checkpoint missing dataset_aug_rng_state")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "torch_rng_state" in ckpt:
            rng_s = ckpt["torch_rng_state"]
            if isinstance(rng_s, torch.Tensor):
                rng_s = rng_s.cpu()
            torch.set_rng_state(rng_s)
        if "numpy_rng_state" in ckpt:
            np_s = ckpt["numpy_rng_state"]
            if isinstance(np_s, list):
                np_s = tuple(np_s)
            np.random.set_state(np_s)
        if "random_rng_state" in ckpt:
            py_s = ckpt["random_rng_state"]
            if isinstance(py_s, list):
                py_s = tuple(py_s)
            random.setstate(py_s)
        if "dataset_rng_state" in ckpt and hasattr(train_ds, "rng") and ckpt["dataset_rng_state"] is not None:
            train_ds.rng.set_state(ckpt["dataset_rng_state"])
        if "dataset_aug_rng_state" in ckpt and hasattr(train_ds, "aug_rng") and ckpt["dataset_aug_rng_state"] is not None and train_ds.aug_rng is not None:
            train_ds.aug_rng.set_state(ckpt["dataset_aug_rng_state"])
        if torch.cuda.is_available() and device.type == "cuda" and "cuda_rng_state" in ckpt:
            cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in ckpt["cuda_rng_state"]]
            torch.cuda.set_rng_state_all(cuda_states)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", (start_epoch - 1) * len(train_loader))
        best_pq = ckpt.get("best_metric", -1.0)
        print(f"[Train] Resumed from epoch {ckpt['epoch']}, updates: {total_successful_updates} successful, {total_skipped_updates} skipped, best metric: {best_pq:.4f}")

    checkpoint_dir = root_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.get("model", {}).get("name", "resnet34_unet")
    latest_ckpt = checkpoint_dir / f"{model_name}_fold{fold}_latest.pt"
    best_ckpt = checkpoint_dir / f"{model_name}_fold{fold}_best.pt"

    # Create immutable versioned run directory
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = root_dir / "artifacts" / "runs" / run_id
    run_ckpt_dir = run_dir / "checkpoints"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = {
        "run_id": run_id,
        "config_path": str(config_path),
        "fold": fold,
        "total_epochs": total_epochs,
        "batch_size": batch_size,
        "grad_accum_steps": effective_grad_accum_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "smoke": smoke,
        "is_diagnostic": is_diag,
        "eligible_for_promotion": (not is_diag),
        "max_steps": max_steps,
        "max_observations": max_observations,
        "max_val_observations": max_val_observations,
        "device": str(device),
        "seed": seed,
        "finetune_from": finetune_from,
        "mining_bank_path": mining_bank_path,
        "mining_bank_sha256": active_bank_sha,
        "augment_flips": augment_flips,
        "crop_policy": crop_policy,
    }
    with open(run_dir / "runtime_config.json", "w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2)

    effective_max_steps = 2 if (smoke and max_steps is None) else max_steps

    print(f"[Train] Starting training run {run_id} on {device} (Fold {fold}, Obs: {len(train_ds)}, Epochs {start_epoch}-{total_epochs})...")
    for epoch in range(start_epoch, total_epochs + 1):
        if deadline_time is not None and time.time() >= deadline_time:
            raise TimeoutError(f"Budget deadline {deadline_time} exceeded before epoch {epoch}.")

        stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=2 if (device.type == "cuda" and batch_size < 4) else 1,
            max_steps=effective_max_steps,
            deadline_time=deadline_time,
        )
        scheduler.step()
        global_step += len(train_loader) if effective_max_steps is None else min(effective_max_steps, len(train_loader))

        epoch_success = int(stats.get("successful_updates", 0))
        epoch_skip = int(stats.get("skipped_updates", 0))
        total_successful_updates += epoch_success
        total_skipped_updates += epoch_skip

        print(f"[Epoch {epoch:02d}] Train Loss: {stats['loss_total']:.4f} | Seg: {stats['loss_seg']:.4f} (BCE: {stats['loss_bce']:.4f}, Dice: {stats['loss_dice']:.4f}, clDice: {stats['loss_cldice']:.4f}) | Bnd: {stats['loss_boundary']:.4f} | Ctr: {stats['loss_center']:.4f} | Off: {stats['loss_offset']:.4f} | Updates: {epoch_success} ok, {epoch_skip} skipped", flush=True)

        is_eligible = (not is_diag) and (total_skipped_updates == 0)

        # Temporary save to enable evaluation
        save_checkpoint(
            path=latest_ckpt,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            fold=fold,
            best_metric_name=best_metric_name,
            best_metric=best_pq,
            train_observations=train_ds.observations,
            val_observations=val_ds.observations,
            folds_manifest_sha256=folds_manifest_sha256,
            dataset_rng_state=train_ds.rng.get_state() if hasattr(train_ds, "rng") else None,
            dataset_aug_rng_state=train_ds.aug_rng.get_state() if (hasattr(train_ds, "aug_rng") and train_ds.aug_rng is not None) else None,
            successful_updates=total_successful_updates,
            skipped_updates=total_skipped_updates,
            run_id=run_id,
            runtime_config=runtime_config,
            finetune_parent=finetune_parent_summary,
            mining_bank_sha256=active_bank_sha,
            total_epochs=total_epochs,
            eligible_for_promotion=is_eligible,
        )

        # Evaluate validation PQ via evaluate_oof
        from evaluate import evaluate_oof
        eval_report = evaluate_oof(
            checkpoint_path=str(latest_ckpt),
            fold=fold,
            split="tuning",
            limit=val_limit,
            allow_unverified_provenance=False,
            device_str=device_name,
            method="connected_components" if finetune_from else None,
            high_threshold=0.85 if finetune_from else None,
            low_threshold=0.70 if finetune_from else None,
            min_area=400 if finetune_from else None,
            max_instances=20 if finetune_from else None,
        )

        # Assert exact Candidate 2 inference configuration in evaluation report
        act_cfg = eval_report.get("resolved_inference_config", {})
        if finetune_from:
            expected_inf_cfg = {
                "method": "connected_components",
                "high_threshold": 0.85,
                "low_threshold": 0.70,
                "min_area": 400,
                "max_instances": 20,
            }
            for k, exp_v in expected_inf_cfg.items():
                if act_cfg.get(k) != exp_v:
                    raise ValueError(f"Inference config mismatch during evaluation: {k}={act_cfg.get(k)}, expected {exp_v}")

        curr_val_pq = eval_report["overall"]["pq"]
        print(f"[Epoch {epoch:02d}] Validation PQ: {curr_val_pq:.4f} (SQ: {eval_report['overall']['sq']:.4f}, RQ: {eval_report['overall']['rq']:.4f}, Dice: {eval_report['overall']['mean_dice']:.4f})", flush=True)

        # Check promotion condition
        is_promoted = False
        if total_skipped_updates > 0:
            print(f"[Warning] Run has {total_skipped_updates} cumulative skipped updates due to non-finite gradients; refusing checkpoint promotion.", flush=True)
        elif is_diag:
            print(f"[Train] Diagnostic / smoke run (is_diag=True); checkpoint is ineligible for promotion.", flush=True)
        elif curr_val_pq > 0.0 and curr_val_pq > best_pq:
            best_pq = curr_val_pq
            is_promoted = True

        # Update latest checkpoint with verified val_stats and promoted best_pq
        save_checkpoint(
            path=latest_ckpt,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            fold=fold,
            best_metric_name=best_metric_name,
            best_metric=best_pq,
            train_observations=train_ds.observations,
            val_observations=val_ds.observations,
            folds_manifest_sha256=folds_manifest_sha256,
            val_stats=eval_report["overall"],
            dataset_rng_state=train_ds.rng.get_state() if hasattr(train_ds, "rng") else None,
            dataset_aug_rng_state=train_ds.aug_rng.get_state() if (hasattr(train_ds, "aug_rng") and train_ds.aug_rng is not None) else None,
            successful_updates=total_successful_updates,
            skipped_updates=total_skipped_updates,
            run_id=run_id,
            runtime_config=runtime_config,
            finetune_parent=finetune_parent_summary,
            mining_bank_sha256=active_bank_sha,
            total_epochs=total_epochs,
            eligible_for_promotion=is_eligible,
        )

        # Save immutable epoch checkpoint in run directory
        save_checkpoint(
            path=run_ckpt_dir / f"epoch_{epoch:03d}.pt",
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            fold=fold,
            best_metric_name=best_metric_name,
            best_metric=best_pq,
            train_observations=train_ds.observations,
            val_observations=val_ds.observations,
            folds_manifest_sha256=folds_manifest_sha256,
            val_stats=eval_report["overall"],
            dataset_rng_state=train_ds.rng.get_state() if hasattr(train_ds, "rng") else None,
            dataset_aug_rng_state=train_ds.aug_rng.get_state() if (hasattr(train_ds, "aug_rng") and train_ds.aug_rng is not None) else None,
            successful_updates=total_successful_updates,
            skipped_updates=total_skipped_updates,
            run_id=run_id,
            runtime_config=runtime_config,
            finetune_parent=finetune_parent_summary,
            mining_bank_sha256=active_bank_sha,
            total_epochs=total_epochs,
            eligible_for_promotion=is_eligible,
        )

        if is_promoted:
            save_checkpoint(
                path=best_ckpt,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                fold=fold,
                best_metric_name=best_metric_name,
                best_metric=best_pq,
                train_observations=train_ds.observations,
                val_observations=val_ds.observations,
                folds_manifest_sha256=folds_manifest_sha256,
                val_stats=eval_report["overall"],
                dataset_rng_state=train_ds.rng.get_state() if hasattr(train_ds, "rng") else None,
                dataset_aug_rng_state=train_ds.aug_rng.get_state() if (hasattr(train_ds, "aug_rng") and train_ds.aug_rng is not None) else None,
                successful_updates=total_successful_updates,
                skipped_updates=total_skipped_updates,
                run_id=run_id,
                runtime_config=runtime_config,
                finetune_parent=finetune_parent_summary,
                mining_bank_sha256=active_bank_sha,
                total_epochs=total_epochs,
                eligible_for_promotion=is_eligible,
            )
            print(f"[Train] Promoted new best checkpoint! ({best_metric_name}: {best_pq:.4f}) -> {best_ckpt}", flush=True)

    print(f"[Train] Training complete. Latest: {latest_ckpt}, Best: {best_ckpt} ({best_metric_name}={best_pq:.4f})", flush=True)
    return latest_ckpt, best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-34 U-Net for Solar Filament Segmentation")
    parser.add_argument("--config", type=str, default="configs/b0_resnet34.yaml", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold ID [0-4]")
    parser.add_argument("--smoke", action="store_true", help="Run bounded smoke test")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch")
    parser.add_argument("--max-observations", type=int, default=None, help="Max training observations")
    parser.add_argument("--max-val-observations", type=int, default=None, help="Max validation observations")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--finetune-from", type=str, default=None, help="Path to parent checkpoint for fine-tuning")
    parser.add_argument("--mining-bank", type=str, default=None, help="Path to frozen mining bank JSON")
    parser.add_argument("--augment-flips", action="store_true", help="Enable geometric horizontal/vertical flip augmentation")
    parser.add_argument("--deadline-timestamp", type=float, default=None, help="Unix timestamp deadline after which training halts safely")
    args = parser.parse_args()

    train(
        config_path=args.config,
        fold=args.fold,
        smoke=args.smoke,
        max_steps=args.max_steps,
        max_observations=args.max_observations,
        max_val_observations=args.max_val_observations,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        device_str=args.device,
        resume_path=args.resume,
        finetune_from=args.finetune_from,
        mining_bank_path=args.mining_bank,
        augment_flips=args.augment_flips,
        deadline_time=args.deadline_timestamp,
    )
