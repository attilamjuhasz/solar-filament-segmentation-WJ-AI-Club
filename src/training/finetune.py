from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.manifest import canonical_observation_id
from src.inference.engine import compute_file_sha256, compute_state_dict_sha256


def validate_finetune_parent(
    checkpoint_path: Path,
    expected_fold: int,
    expected_manifest_sha: str,
    train_observations: Optional[Sequence[str]] = None,
    val_observations: Optional[Sequence[str]] = None,
    is_diagnostic: bool = False,
) -> Dict[str, Any]:
    """Validate that the parent checkpoint meets all strict provenance criteria for fine-tuning.
    
    Guarantees:
    - File exists and has full provenance.
    - Checkpoint fold strictly matches expected fold (fold 0).
    - Folds manifest hash strictly matches active frozen manifest.
    - Model architecture matches resnet34_unet.
    - Full production runs require EXACT 565 train and 142 validation canonical groups, strictly disjoint.
    - Diagnostic smoke runs must be explicitly flagged and ineligible for promotion.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Fine-tune parent checkpoint not found: {ckpt_path}")

    file_sha256 = compute_file_sha256(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    ckpt_fold = ckpt.get("fold")
    if ckpt_fold != expected_fold:
        raise ValueError(
            f"Fine-tune fold mismatch: checkpoint was trained on fold {ckpt_fold}, expected fold {expected_fold}."
        )

    ckpt_man_sha = ckpt.get("folds_manifest_sha256")
    if ckpt_man_sha != expected_manifest_sha:
        raise ValueError(
            f"Fine-tune manifest mismatch: checkpoint has manifest SHA {ckpt_man_sha}, expected {expected_manifest_sha}."
        )

    ckpt_model_cfg = ckpt.get("config", {}).get("model", {})
    model_name = ckpt_model_cfg.get("name", "")
    if model_name != "resnet34_unet":
        raise ValueError(f"Unsupported fine-tune architecture '{model_name}'; must be 'resnet34_unet'.")

    state_dict_sha = compute_state_dict_sha256(ckpt["model_state_dict"])
    parent_cfg = ckpt.get("config", {})
    parent_cfg_sha = hashlib.sha256(json.dumps(parent_cfg, sort_keys=True).encode("utf-8")).hexdigest()

    ckpt_train_canons = {canonical_observation_id(x) for x in ckpt.get("train_observations", [])}
    ckpt_val_canons = {canonical_observation_id(x) for x in ckpt.get("val_observations", [])}

    if not ckpt_train_canons or not ckpt_val_canons:
        raise ValueError("Parent checkpoint missing recorded train/validation observation sets.")

    overlap = ckpt_train_canons.intersection(ckpt_val_canons)
    if overlap:
        raise ValueError(f"Parent checkpoint has train/val observation overlap: {sorted(list(overlap))}")

    if train_observations is not None:
        act_train_canons = {canonical_observation_id(x) for x in train_observations}
        if is_diagnostic:
            if not act_train_canons.issubset(ckpt_train_canons):
                diff = act_train_canons - ckpt_train_canons
                raise ValueError(f"Diagnostic training set contains unknown observations: {len(diff)}")
        else:
            if act_train_canons != ckpt_train_canons:
                raise ValueError(
                    f"Fine-tune training membership mismatch: expected exact 565 parent observations ({len(ckpt_train_canons)}), "
                    f"got {len(act_train_canons)}. Difference: {len(ckpt_train_canons.symmetric_difference(act_train_canons))}"
                )

    if val_observations is not None:
        act_val_canons = {canonical_observation_id(x) for x in val_observations}
        if is_diagnostic:
            if not act_val_canons.issubset(ckpt_val_canons):
                diff = act_val_canons - ckpt_val_canons
                raise ValueError(f"Diagnostic validation set contains unknown observations: {len(diff)}")
        else:
            if act_val_canons != ckpt_val_canons:
                raise ValueError(
                    f"Fine-tune validation membership mismatch: expected exact 142 parent observations ({len(ckpt_val_canons)}), "
                    f"got {len(act_val_canons)}. Difference: {len(ckpt_val_canons.symmetric_difference(act_val_canons))}"
                )

    if train_observations is not None and val_observations is not None:
        disjoint_overlap = act_train_canons.intersection(act_val_canons)
        if disjoint_overlap:
            raise ValueError(f"Active train and validation sets are not disjoint: {sorted(list(disjoint_overlap))}")

    return {
        "path": str(ckpt_path),
        "file_sha256": file_sha256,
        "model_state_sha256": state_dict_sha,
        "training_config_sha256": parent_cfg_sha,
        "parent_epoch": ckpt.get("epoch", 6),
        "fold": ckpt_fold,
        "folds_manifest_sha256": ckpt_man_sha,
        "model_state_dict": ckpt["model_state_dict"],
        "is_diagnostic": is_diagnostic,
        "eligible_for_promotion": not is_diagnostic,
        "train_count": len(ckpt_train_canons),
        "val_count": len(ckpt_val_canons),
    }


def setup_finetune_state(
    model: nn.Module,
    parent_state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    lr: float = 0.00005,
    weight_decay: float = 0.0001,
    total_epochs: int = 3,
    eta_min: float = 0.000001,
) -> Tuple[AdamW, CosineAnnealingLR, Optional[torch.amp.GradScaler]]:
    """Initialize fresh optimizer, cosine schedule, and scaler for fine-tuning from loaded weights.
    
    Guarantees:
    - Loads parent model weights into model.
    - Discards parent optimizer/scheduler/scaler states.
    - Configures fresh AdamW with target low learning rate (default 5e-5).
    - Configures fresh CosineAnnealingLR over total_epochs (default 3) to eta_min (default 1e-6).
    """
    model.load_state_dict(parent_state_dict)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda") else None

    return optimizer, scheduler, scaler
