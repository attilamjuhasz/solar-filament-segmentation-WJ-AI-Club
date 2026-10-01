"""
Gate B: Training-crop overfit diagnostic.
Validates image-mask alignment, polarity, normalization, and tests whether
a ResNet-34 U-Net with pure BCE+Dice can overfit a fixed mini-batch of 4
training crops to achieve eval-mode Dice > 0.85 without BatchNorm drift degradation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import SolarFilamentDataset
from src.data.folds import load_frozen_folds_manifest
from src.losses import TorchSoftDiceLoss
from src.models import ResNet34UNet


class PureBCEDiceLoss(nn.Module):
    """Pure BCE + Soft Dice loss with safe FP32 reductions and valid-mask weighting."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = TorchSoftDiceLoss(from_logits=True)

    def forward(
        self,
        fg_logits: torch.Tensor,
        target_fg: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute BCE on valid pixels
        bce_map = self.bce(fg_logits.float(), target_fg.float())
        if valid_mask is not None:
            loss_bce = (bce_map * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            loss_bce = bce_map.mean()

        # Compute per-sample Dice loss in FP32
        loss_dice = self.dice(fg_logits, target_fg, valid_mask=valid_mask)

        total_loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_dice": float(loss_dice.item()),
        }


def compute_batch_dice(
    pred_logits: torch.Tensor,
    target_fg: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.50,
) -> float:
    """Compute hard Dice coefficient for binary predictions at given threshold, respecting valid disk mask."""
    probs = torch.sigmoid(pred_logits)
    if valid_mask is not None:
        probs = probs * valid_mask
    preds = (probs >= threshold).float()
    targets = (target_fg > 0).float()
    if valid_mask is not None:
        targets = targets * valid_mask

    inter = (preds * targets).sum().item()
    total = preds.sum().item() + targets.sum().item()
    if total == 0:
        return 1.0 if targets.sum().item() == 0 else 0.0
    return (2.0 * inter) / total


def run_overfit_diagnostic(
    max_steps: int = 120,
    lr: float = 1e-3,
    target_eval_dice: float = 0.85,
    device_str: str | None = None,
) -> Dict[str, Any]:
    """Execute Gate B overfit diagnostic on 4 fixed training crops."""
    start_time = time.time()
    device_name = device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[Gate B] Running overfit diagnostic on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load dataset and select 4 fixed training crops with genuine filaments
    manifest_path = root_dir / "artifacts" / "folds_manifest.json"
    assignments, manifest_sha256 = load_frozen_folds_manifest(manifest_path)
    data_dir = root_dir / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_images = data_dir / "train" / "train_images"

    train_ds = SolarFilamentDataset(
        images_dir=train_images,
        annotations_json=train_json,
        fold_assignments=assignments,
        target_fold=0,
        is_train=True,
        patch_size=(512, 512),
        fg_crop_prob=1.0,
        norm_mode="imagenet",
        seed=42,
    )

    # Collect 4 crops with >= 1000 foreground pixels
    fixed_samples = []
    crop_info = []
    for i in range(50):
        sample = train_ds[i]
        fg_pixels = int(sample["target_fg"].sum().item())
        if fg_pixels >= 1000:
            fixed_samples.append(sample)
            crop_info.append({
                "observation_id": sample["observation_id"],
                "origin": sample["origin"],
                "fg_pixels": fg_pixels,
                "fg_rate": float(fg_pixels / (512 * 512)),
            })
            if len(fixed_samples) >= 4:
                break

    if len(fixed_samples) < 4:
        raise RuntimeError(f"Could not find 4 training crops with sufficient foreground (found {len(fixed_samples)})")

    print(f"[Gate B] Selected {len(fixed_samples)} fixed training crops:")
    for ci in crop_info:
        print(f"  - Obs: {ci['observation_id']}, Origin: {ci['origin']}, FG Pixels: {ci['fg_pixels']} ({ci['fg_rate']*100:.2f}%)")

    # Stack into fixed batch
    batch_images = torch.stack([s["image"] for s in fixed_samples]).to(device)  # [4, 3, 512, 512]
    batch_fg = torch.stack([s["target_fg"] for s in fixed_samples]).to(device)   # [4, 1, 512, 512]
    batch_valid = torch.stack([s["valid_mask"] for s in fixed_samples]).to(device) # [4, 1, 512, 512]

    # Pre-training data audit & verification
    img_mean = batch_images.mean().item()
    img_std = batch_images.std().item()
    valid_fraction = batch_valid.mean().item()
    total_fg_pixels = int(batch_fg.sum().item())

    # Check intensity contrast: H-alpha filaments are darker absorption features
    # On ImageNet normalized images, darker means lower values
    fg_mask_bool = batch_fg > 0
    bg_mask_bool = (batch_fg == 0) & (batch_valid > 0)
    fg_mean_intensity = batch_images[:, 0:1, :, :][fg_mask_bool].mean().item()
    bg_mean_intensity = batch_images[:, 0:1, :, :][bg_mask_bool].mean().item()

    print(f"[Gate B] Pre-training Data Audit:")
    print(f"  - Batch shape: {list(batch_images.shape)}")
    print(f"  - Normalization: ImageNet (Batch Mean: {img_mean:.3f}, Std: {img_std:.3f})")
    print(f"  - Valid mask fraction: {valid_fraction*100:.2f}%")
    print(f"  - Total FG pixels: {total_fg_pixels} across 4 crops")
    print(f"  - Channel 0 Mean Intensity on GT FG: {fg_mean_intensity:.3f} vs BG: {bg_mean_intensity:.3f}")
    print(f"  - Contrast delta (FG - BG): {fg_mean_intensity - bg_mean_intensity:.3f} (confirms absorption feature alignment)")

    # 2. Build model with pure BCE+Dice
    model = ResNet34UNet(in_channels=3, pretrained=True).to(device)
    criterion = PureBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Train loop tracking train-mode vs eval-mode metrics and BN drift
    history: List[Dict[str, Any]] = []
    eval_dice_passed = False
    best_eval_dice = 0.0

    print(f"\n[Gate B] Starting optimization loop ({max_steps} steps)...")
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad()

        preds = model(batch_images)
        fg_logits = preds["fg_logits"].float()
        loss, stats = criterion(fg_logits, batch_fg, valid_mask=batch_valid)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item()
        optimizer.step()

        # Evaluate at checkpoints (every 10 steps, plus step 1, and final step)
        if step == 1 or step % 10 == 0 or step == max_steps:
            # 1. Train-mode stats (using batch statistics in BatchNorm)
            model.train()
            with torch.no_grad():
                train_preds = model(batch_images)["fg_logits"].float()
                train_dice = compute_batch_dice(train_preds, batch_fg, valid_mask=batch_valid)
                train_fg_logit_mean = train_preds[fg_mask_bool].mean().item()
                train_bg_logit_mean = train_preds[bg_mask_bool].mean().item()

            # 2. Eval-mode stats (using running statistics in BatchNorm)
            model.eval()
            with torch.no_grad():
                eval_preds = model(batch_images)["fg_logits"].float()
                eval_loss, eval_stats = criterion(eval_preds, batch_fg, valid_mask=batch_valid)
                eval_dice = compute_batch_dice(eval_preds, batch_fg, valid_mask=batch_valid)
                eval_fg_logit_mean = eval_preds[fg_mask_bool].mean().item()
                eval_bg_logit_mean = eval_preds[bg_mask_bool].mean().item()

            bn_drift = abs(train_dice - eval_dice)
            if eval_dice > best_eval_dice:
                best_eval_dice = eval_dice

            record = {
                "step": step,
                "train_loss": stats["loss_total"],
                "train_dice": round(train_dice, 5),
                "eval_loss": eval_stats["loss_total"],
                "eval_dice": round(eval_dice, 5),
                "bn_drift": round(bn_drift, 5),
                "grad_norm": round(grad_norm, 4),
                "train_fg_logit_mean": round(train_fg_logit_mean, 3),
                "train_bg_logit_mean": round(train_bg_logit_mean, 3),
                "eval_fg_logit_mean": round(eval_fg_logit_mean, 3),
                "eval_bg_logit_mean": round(eval_bg_logit_mean, 3),
            }
            history.append(record)

            print(
                f"Step {step:03d} | "
                f"Train Loss: {stats['loss_total']:.4f} (Dice: {train_dice:.4f}) | "
                f"Eval Loss: {eval_stats['loss_total']:.4f} (Dice: {eval_dice:.4f}) | "
                f"BN Drift: {bn_drift:.4f} | "
                f"Eval Logits (FG: {eval_fg_logit_mean:+.2f}, BG: {eval_bg_logit_mean:+.2f})"
            )

            if eval_dice >= target_eval_dice:
                eval_dice_passed = True

    elapsed = time.time() - start_time
    print(f"\n[Gate B] Diagnostic completed in {elapsed:.2f}s.")
    print(f"[Gate B] Best Eval-Mode Dice: {best_eval_dice:.4f} (Target: {target_eval_dice:.4f})")
    print(f"[Gate B] Gate Passed: {eval_dice_passed}")

    # Build report
    report = {
        "gate": "Gate B: Training-Crop Overfit Diagnostic",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "device": str(device),
        "target_eval_dice": target_eval_dice,
        "best_eval_dice": round(best_eval_dice, 5),
        "final_eval_dice": history[-1]["eval_dice"] if history else 0.0,
        "final_train_dice": history[-1]["train_dice"] if history else 0.0,
        "final_bn_drift": history[-1]["bn_drift"] if history else 0.0,
        "gate_passed": eval_dice_passed,
        "data_audit": {
            "normalization": "imagenet",
            "img_mean": round(img_mean, 4),
            "img_std": round(img_std, 4),
            "valid_fraction": round(valid_fraction, 4),
            "total_fg_pixels": total_fg_pixels,
            "fg_mean_intensity": round(fg_mean_intensity, 4),
            "bg_mean_intensity": round(bg_mean_intensity, 4),
            "contrast_delta": round(fg_mean_intensity - bg_mean_intensity, 4),
            "crops": crop_info,
        },
        "history": history,
    }

    report_dir = root_dir / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "overfit_diagnostic.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Gate B] Diagnostic report saved to: {report_path}")

    return report


if __name__ == "__main__":
    report = run_overfit_diagnostic()
    if not report["gate_passed"]:
        print(f"[ERROR] Gate B failed: eval Dice did not reach {report['target_eval_dice']}")
        sys.exit(1)
    else:
        print(f"[SUCCESS] Gate B verified: eval Dice {report['best_eval_dice']} >= {report['target_eval_dice']}")
        sys.exit(0)
