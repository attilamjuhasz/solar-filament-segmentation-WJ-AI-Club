from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy import ndimage

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = object
    F = None
    TORCH_AVAILABLE = False


# ============================================================================
# 1. NumPy / SciPy Numerical Reference Implementations (Bug-free & Safe)
# ============================================================================

def binary_cross_entropy(
    prob: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pos_weight: float = 1.0,
    eps: float = 1e-7
) -> float:
    """Class-balanced Binary Cross Entropy with valid-mask filtering."""
    p = np.clip(prob, eps, 1.0 - eps)
    t = target.astype(np.float32)

    loss_map = -(pos_weight * t * np.log(p) + (1.0 - t) * np.log(1.0 - p))
    if valid_mask is not None:
        valid = valid_mask.astype(bool)
        if not valid.any():
            return 0.0
        return float(loss_map[valid].mean())
    return float(loss_map.mean())


def dice_loss(
    prob: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-6
) -> float:
    """Per-sample soft Dice loss."""
    p = prob.astype(np.float32)
    t = target.astype(np.float32)

    if valid_mask is not None:
        v = valid_mask.astype(np.float32)
        if not (v > 0).any():
            return 0.0
        p = p * v
        t = t * v

    intersection = 2.0 * np.sum(p * t) + eps
    union = np.sum(p) + np.sum(t) + eps
    return float(1.0 - (intersection / union))


def soft_erode(p: np.ndarray) -> np.ndarray:
    """Soft morphological erosion using min-pooling (1 - max-pooling(1 - p))."""
    inv_p = 1.0 - p
    max_pooled = ndimage.maximum_filter(inv_p, size=3, mode="reflect")
    return 1.0 - max_pooled


def soft_open(p: np.ndarray) -> np.ndarray:
    """Soft opening: soft erosion followed by soft dilation (max-pooling)."""
    eroded = soft_erode(p)
    return ndimage.maximum_filter(eroded, size=3, mode="reflect")


def soft_skeletonize(p: np.ndarray, iters: int = 15) -> np.ndarray:
    """Iterative soft morphological skeletonization: S(p) = sum_k (p_k - open(p_k))."""
    current_p = p.astype(np.float32)
    skel = np.zeros_like(current_p)

    for _ in range(iters):
        opened = soft_open(current_p)
        top_hat = np.maximum(0.0, current_p - opened)
        skel = np.maximum(skel, top_hat)
        current_p = soft_erode(current_p)
        if not (current_p > 0.01).any():
            break

    return np.clip(skel, 0.0, 1.0)


def cldice_loss(
    prob: np.ndarray,
    target_skeleton: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    iters: int = 15,
    eps: float = 1e-6
) -> float:
    """Topology-preserving Centerline Dice (clDice) loss."""
    if not (target_mask > 0).any():
        return 0.0

    p = prob.astype(np.float32)
    s_p = soft_skeletonize(p, iters=iters)
    s_g = target_skeleton.astype(np.float32)
    g = target_mask.astype(np.float32)

    if valid_mask is not None:
        v = valid_mask.astype(np.float32)
        if not (v > 0).any():
            return 0.0
        s_p *= v
        p *= v
        s_g *= v
        g *= v

    t_prec = (np.sum(s_p * g) + eps) / (np.sum(s_p) + eps)
    t_sens = (np.sum(s_g * p) + eps) / (np.sum(s_g) + eps)

    cldice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + eps)
    return float(1.0 - cldice)


def compute_compound_loss(
    fg_prob: np.ndarray,
    boundary_prob: np.ndarray,
    center_prob: np.ndarray,
    offset_pred: np.ndarray,
    target_fg: np.ndarray,
    target_skel: np.ndarray,
    target_boundary: np.ndarray,
    target_center: np.ndarray,
    target_offset: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    alpha: float = 0.35,
    beta: float = 0.45,
    gamma: float = 0.20,
    w_bnd: float = 0.10,
    w_ctr: float = 0.05,
    w_off: float = 0.05,
    cldice_iters: int = 15
) -> Dict[str, float]:
    """Compute total compound multi-task topology loss (NumPy reference).
    
    Guarantees:
    - Safe reduction when valid_mask is completely empty (no NaNs).
    - Configurable cldice_iters passed through.
    - Configurable auxiliary weights.
    """
    # Check if valid_mask is completely empty
    if valid_mask is not None and not valid_mask.astype(bool).any():
        return {
            "loss_total": 0.0,
            "loss_seg": 0.0,
            "loss_bce": 0.0,
            "loss_dice": 0.0,
            "loss_cldice": 0.0,
            "loss_boundary": 0.0,
            "loss_center": 0.0,
            "loss_offset": 0.0
        }

    # 1. Segmentation objective
    bce = binary_cross_entropy(fg_prob, target_fg, valid_mask=valid_mask)
    dice = dice_loss(fg_prob, target_fg, valid_mask=valid_mask)
    cld = cldice_loss(fg_prob, target_skel, target_fg, valid_mask=valid_mask, iters=cldice_iters)
    l_seg = alpha * bce + beta * dice + gamma * cld

    # 2. Auxiliary boundaries
    bnd_bce = binary_cross_entropy(boundary_prob, target_boundary, valid_mask=valid_mask)
    bnd_dice = dice_loss(boundary_prob, target_boundary, valid_mask=valid_mask)
    l_boundary = 0.5 * (bnd_bce + bnd_dice)

    # 3. Center heatmap MSE (safe against empty mask)
    center_err = (center_prob - target_center) ** 2
    if valid_mask is not None:
        v_bool = valid_mask.astype(bool)
        l_center = float(center_err[v_bool].mean()) if v_bool.any() else 0.0
    else:
        l_center = float(center_err.mean())

    # 4. Foreground-masked offset Smooth L1
    fg_bool = target_fg.astype(bool)
    if valid_mask is not None:
        fg_bool &= valid_mask.astype(bool)

    if fg_bool.any():
        diff = np.abs(offset_pred[:, fg_bool] - target_offset[:, fg_bool])
        smooth_l1 = np.where(diff < 1.0, 0.5 * (diff ** 2), diff - 0.5)
        l_offset = float(smooth_l1.mean())
    else:
        l_offset = 0.0

    l_total = l_seg + w_bnd * l_boundary + w_ctr * l_center + w_off * l_offset

    return {
        "loss_total": float(l_total),
        "loss_seg": float(l_seg),
        "loss_bce": float(bce),
        "loss_dice": float(dice),
        "loss_cldice": float(cld),
        "loss_boundary": float(l_boundary),
        "loss_center": float(l_center),
        "loss_offset": float(l_offset)
    }


# ============================================================================
# 2. Differentiable PyTorch Autograd Loss Classes
# ============================================================================

if TORCH_AVAILABLE:

    def canonicalize_scalar_tensor(
        tensor: torch.Tensor,
        expected_b: int,
        expected_h: int,
        expected_w: int,
        name: str
    ) -> torch.Tensor:
        """Validate and canonicalize a single-channel 2D spatial tensor strictly to [B, 1, H, W].
        
        Permits [B, 1, H, W] or [B, H, W] and ensures batch and spatial dimensions match
        exactly to prevent silent broadcasting bugs.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 4:
            if tensor.shape[1] != 1:
                raise ValueError(f"{name} must have channel size 1, got shape {tuple(tensor.shape)}")
        else:
            raise ValueError(f"{name} must have 3 or 4 dimensions, got shape {tuple(tensor.shape)}")

        if tensor.shape[0] != expected_b or tensor.shape[2] != expected_h or tensor.shape[3] != expected_w:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match expected (B={expected_b}, 1, H={expected_h}, W={expected_w})"
            )
        return tensor

    def validate_offset_tensor(
        tensor: torch.Tensor,
        expected_b: int,
        expected_h: int,
        expected_w: int,
        name: str
    ) -> torch.Tensor:
        """Validate an offset vector field tensor strictly to [B, 2, H, W]."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        if tensor.ndim != 4 or tensor.shape[1] != 2:
            raise ValueError(f"{name} must have shape [B, 2, H, W], got shape {tuple(tensor.shape)}")
        if tensor.shape[0] != expected_b or tensor.shape[2] != expected_h or tensor.shape[3] != expected_w:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match expected (B={expected_b}, 2, H={expected_h}, W={expected_w})"
            )
        return tensor

    def canonicalize_mask(
        valid_mask: Optional[torch.Tensor],
        expected_b: int,
        expected_h: int,
        expected_w: int,
        name: str = "valid_mask"
    ) -> Optional[torch.Tensor]:
        """Validate and canonicalize a mask tensor strictly to [B, 1, H, W] in float32."""
        if valid_mask is None:
            return None
        return canonicalize_scalar_tensor(valid_mask.float(), expected_b, expected_h, expected_w, name=name)

    class TorchSoftDiceLoss(nn.Module):
        """Differentiable soft Dice loss per-sample from logits or probabilities with FP32 reduction safety."""
        def __init__(self, eps: float = 1e-6, from_logits: bool = True):
            super().__init__()
            self.eps = eps
            self.from_logits = from_logits

        def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            if pred.ndim not in (3, 4):
                raise ValueError(f"pred must have 3 or 4 dimensions, got shape {tuple(pred.shape)}")
            if pred.ndim == 4 and pred.shape[1] != 1:
                raise ValueError(f"pred must have channel size 1, got shape {tuple(pred.shape)}")

            B = pred.shape[0]
            H = pred.shape[-2]
            W = pred.shape[-1]
            p_in = canonicalize_scalar_tensor(pred, B, H, W, name="pred")
            t_in = canonicalize_scalar_tensor(target, B, H, W, name="target")
            v = canonicalize_mask(valid_mask, B, H, W, name="valid_mask")

            # Force FP32 calculation to prevent FP16 half-precision overflow on large spatial grids
            p = torch.sigmoid(p_in.float()) if self.from_logits else p_in.float()
            t = t_in.float()

            # Sample eligibility: samples that have valid pixels (or all samples if no mask)
            if v is not None:
                sample_valid = (v.view(B, -1).sum(dim=1) > 0)
                if not sample_valid.any():
                    return 0.0 * pred.float().sum()
                p = p * v
                t = t * v
            else:
                sample_valid = torch.ones(B, dtype=torch.bool, device=pred.device)

            # Flatten spatial dimensions per batch sample: [B, -1]
            p_flat = p.view(B, -1)
            t_flat = t.view(B, -1)

            intersection = 2.0 * torch.sum(p_flat * t_flat, dim=1) + self.eps
            union = torch.sum(p_flat, dim=1) + torch.sum(t_flat, dim=1) + self.eps
            dice_per_sample = 1.0 - (intersection / union)

            # Exclude wholly invalid samples from per-sample Dice averages
            return dice_per_sample[sample_valid].mean()

    class TorchSoftSkeletonize(nn.Module):
        """Differentiable 2D soft morphological skeletonization using spatial max pooling in FP32."""
        def __init__(self, iters: int = 15):
            super().__init__()
            self.iters = iters

        def soft_erode(self, p: torch.Tensor) -> torch.Tensor:
            # Erosion = 1 - max_pool(1 - p)
            return 1.0 - F.max_pool2d(1.0 - p, kernel_size=3, stride=1, padding=1)

        def soft_open(self, p: torch.Tensor) -> torch.Tensor:
            eroded = self.soft_erode(p)
            return F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)

        def forward(self, p: torch.Tensor) -> torch.Tensor:
            current_p = p.float()
            skel = torch.zeros_like(current_p)
            for _ in range(self.iters):
                opened = self.soft_open(current_p)
                top_hat = F.relu(current_p - opened)
                skel = torch.maximum(skel, top_hat)
                current_p = self.soft_erode(current_p)
            return torch.clamp(skel, 0.0, 1.0)

    class TorchSoftclDiceLoss(nn.Module):
        """Differentiable soft clDice loss per-sample with FP32 reduction safety.
        
        Preserves topology along thin filament spines by penalizing disconnects.
        """
        def __init__(self, iters: int = 15, eps: float = 1e-6, from_logits: bool = True):
            super().__init__()
            self.eps = eps
            self.from_logits = from_logits
            self.skeletonizer = TorchSoftSkeletonize(iters=iters)

        def forward(
            self,
            pred: torch.Tensor,
            target_skel: torch.Tensor,
            target_mask: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            if pred.ndim not in (3, 4):
                raise ValueError(f"pred must have 3 or 4 dimensions, got shape {tuple(pred.shape)}")
            if pred.ndim == 4 and pred.shape[1] != 1:
                raise ValueError(f"pred must have channel size 1, got shape {tuple(pred.shape)}")

            B = pred.shape[0]
            H = pred.shape[-2]
            W = pred.shape[-1]
            p_in = canonicalize_scalar_tensor(pred, B, H, W, name="pred")
            s_g_in = canonicalize_scalar_tensor(target_skel, B, H, W, name="target_skel")
            g_in = canonicalize_scalar_tensor(target_mask, B, H, W, name="target_mask")
            v = canonicalize_mask(valid_mask, B, H, W, name="valid_mask")

            # Force FP32 calculation to prevent FP16 half-precision overflow on large spatial grids
            p = torch.sigmoid(p_in.float()) if self.from_logits else p_in.float()
            s_g = s_g_in.float()
            g = g_in.float()

            # Determine eligible samples: only samples with non-empty valid foreground
            # (GT foreground intersect valid region)
            if v is not None:
                valid_g = g * v
                eligible = (valid_g.view(B, -1).sum(dim=1) > 0)
            else:
                eligible = (g.view(B, -1).sum(dim=1) > 0)

            if not eligible.any():
                return 0.0 * pred.float().sum()

            s_p = self.skeletonizer(p)

            if v is not None:
                s_p = s_p * v
                p = p * v
                s_g = s_g * v
                g = g * v

            s_p_flat = s_p.view(B, -1)
            p_flat = p.view(B, -1)
            s_g_flat = s_g.view(B, -1)
            g_flat = g.view(B, -1)

            t_prec = (torch.sum(s_p_flat * g_flat, dim=1) + self.eps) / (torch.sum(s_p_flat, dim=1) + self.eps)
            t_sens = (torch.sum(s_g_flat * p_flat, dim=1) + self.eps) / (torch.sum(s_g_flat, dim=1) + self.eps)

            cldice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + self.eps)
            cldice_per_sample = 1.0 - cldice
            # Reduce ONLY over eligible (non-empty valid GT) samples so negative patches don't dilute the loss
            return cldice_per_sample[eligible].mean()

    class CompoundTopologyLoss(nn.Module):
        """Complete differentiable multi-task loss module for Option B1 architecture.
        
        Preserves autograd gradients on:
        - Foreground semantic head (BCEWithLogits + SoftDice + SoftclDice)
        - Boundary detection head (BCEWithLogits + SoftDice)
        - Center heatmap proposal head (MSE / Focal)
        - Continuous offset vector field head (Masked Smooth L1)
        """
        def __init__(
            self,
            alpha: float = 0.35,
            beta: float = 0.45,
            gamma: float = 0.20,
            w_bnd: float = 0.10,
            w_ctr: float = 0.05,
            w_off: float = 0.05,
            cldice_iters: int = 15
        ):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.w_bnd = w_bnd
            self.w_ctr = w_ctr
            self.w_off = w_off

            self.bce = nn.BCEWithLogitsLoss(reduction="none")
            self.dice = TorchSoftDiceLoss(from_logits=True)
            self.cldice = TorchSoftclDiceLoss(iters=cldice_iters, from_logits=True)
            self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

        def forward(
            self,
            fg_logits: torch.Tensor,
            bnd_logits: torch.Tensor,
            ctr_logits: torch.Tensor,
            off_pred: torch.Tensor,
            target_fg: torch.Tensor,
            target_skel: torch.Tensor,
            target_bnd: torch.Tensor,
            target_ctr: torch.Tensor,
            target_off: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            if fg_logits.ndim not in (3, 4):
                raise ValueError(f"fg_logits must have 3 or 4 dimensions, got shape {tuple(fg_logits.shape)}")
            if fg_logits.ndim == 4 and fg_logits.shape[1] != 1:
                raise ValueError(f"fg_logits must have channel size 1, got shape {tuple(fg_logits.shape)}")

            B = fg_logits.shape[0]
            H = fg_logits.shape[-2]
            W = fg_logits.shape[-1]

            # Feedback #3 [P2] #2: Validate all prediction heads and targets at entry, BEFORE invalid-only branch
            fg_logits = canonicalize_scalar_tensor(fg_logits, B, H, W, name="fg_logits")
            bnd_logits = canonicalize_scalar_tensor(bnd_logits, B, H, W, name="bnd_logits")
            ctr_logits = canonicalize_scalar_tensor(ctr_logits, B, H, W, name="ctr_logits")
            off_pred = validate_offset_tensor(off_pred, B, H, W, name="off_pred")

            target_fg = canonicalize_scalar_tensor(target_fg, B, H, W, name="target_fg")
            target_skel = canonicalize_scalar_tensor(target_skel, B, H, W, name="target_skel")
            target_bnd = canonicalize_scalar_tensor(target_bnd, B, H, W, name="target_bnd")
            target_ctr = canonicalize_scalar_tensor(target_ctr, B, H, W, name="target_ctr")
            target_off = validate_offset_tensor(target_off, B, H, W, name="target_off")

            # Canonicalize valid_mask strictly to [B, 1, H, W]
            v = canonicalize_mask(valid_mask, B, H, W, name="valid_mask")

            # Ensure graph-connected zero on entirely invalid batches with safe FP32 promotion
            if v is not None and v.sum() == 0:
                zero = 0.0 * (fg_logits.float().sum() + bnd_logits.float().sum() + ctr_logits.float().sum() + off_pred.float().sum())
                return zero, {k: 0.0 for k in ["loss_total", "loss_seg", "loss_bce", "loss_dice", "loss_cldice", "loss_boundary", "loss_center", "loss_offset"]}

            # 1. Foreground segmentation (FP32 safe)
            bce_map = self.bce(fg_logits.float(), target_fg.float())
            if v is not None:
                loss_bce = (bce_map * v).sum() / (v.sum() + 1e-6)
            else:
                loss_bce = bce_map.mean()

            loss_dice = self.dice(fg_logits, target_fg, valid_mask=v)
            loss_cld = self.cldice(fg_logits, target_skel, target_fg, valid_mask=v)
            loss_seg = self.alpha * loss_bce + self.beta * loss_dice + self.gamma * loss_cld

            # 2. Boundary supervision
            bnd_bce_map = self.bce(bnd_logits.float(), target_bnd.float())
            if v is not None:
                loss_bnd_bce = (bnd_bce_map * v).sum() / (v.sum() + 1e-6)
            else:
                loss_bnd_bce = bnd_bce_map.mean()
            loss_bnd_dice = self.dice(bnd_logits, target_bnd, valid_mask=v)
            loss_bnd = 0.5 * (loss_bnd_bce + loss_bnd_dice)

            # 3. Center proposal heatmap (MSE on sigmoid probabilities)
            ctr_prob = torch.sigmoid(ctr_logits.float())
            ctr_err = (ctr_prob - target_ctr.float()) ** 2
            if v is not None:
                loss_ctr = (ctr_err * v).sum() / (v.sum() + 1e-6)
            else:
                loss_ctr = ctr_err.mean()

            # 4. Continuous offset field (foreground masked Smooth L1)
            fg_mask = (target_fg > 0).float()
            if v is not None:
                fg_mask = fg_mask * v

            off_err = self.smooth_l1(off_pred.float(), target_off.float())  # [B, 2, H, W]
            masked_off_err = off_err * fg_mask  # Broadcasts along channel dim 1 without sample mixing
            off_denom = fg_mask.sum() * 2.0 + 1e-6
            loss_off = masked_off_err.sum() / off_denom

            loss_total = loss_seg + self.w_bnd * loss_bnd + self.w_ctr * loss_ctr + self.w_off * loss_off

            stats = {
                "loss_total": float(loss_total.detach().cpu().item()),
                "loss_seg": float(loss_seg.detach().cpu().item()),
                "loss_bce": float(loss_bce.detach().cpu().item()),
                "loss_dice": float(loss_dice.detach().cpu().item()),
                "loss_cldice": float(loss_cld.detach().cpu().item()),
                "loss_boundary": float(loss_bnd.detach().cpu().item()),
                "loss_center": float(loss_ctr.detach().cpu().item()),
                "loss_offset": float(loss_off.detach().cpu().item())
            }

            return loss_total, stats
