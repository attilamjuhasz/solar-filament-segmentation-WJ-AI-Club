import numpy as np
import pytest

from src.losses import (
    binary_cross_entropy,
    dice_loss,
    cldice_loss,
    compute_compound_loss,
    soft_skeletonize,
    TORCH_AVAILABLE,
)
if TORCH_AVAILABLE:
    import torch
    from src.losses import CompoundTopologyLoss


def test_losses_on_perfect_prediction():
    target = np.zeros((100, 100), dtype=np.uint8)
    target[30:70, 30:70] = 1

    prob = target.astype(np.float32)

    bce = binary_cross_entropy(prob, target)
    assert bce < 1e-4

    dice = dice_loss(prob, target)
    assert dice < 1e-4

    skel = soft_skeletonize(prob)
    cld = cldice_loss(prob, skel, target)
    assert cld < 1e-3


def test_cldice_penalizes_omitted_thin_structures():
    """Verify that clDice severely penalizes missing thin filaments where standard Dice barely reacts."""
    # Target: 400-pixel thick blob + 60-pixel thin 1-pixel filament
    target = np.zeros((100, 100), dtype=np.uint8)
    target[20:40, 20:40] = 1   # 400 pixels
    target[50, 20:80] = 1      # 60 pixels
    
    skel = np.zeros((100, 100), dtype=np.uint8)
    skel[30, 20:40] = 1
    skel[50, 20:80] = 1

    # Prediction captures the 400-pixel blob, but COMPLETELY OMITS the thin filament
    pred_omit = np.zeros((100, 100), dtype=np.float32)
    pred_omit[20:40, 20:40] = 1.0

    d_loss = dice_loss(pred_omit, target)
    cld_loss = cldice_loss(pred_omit, skel, target)

    # Standard Dice loss is small (~0.07) because area is dominated by the blob
    assert d_loss < 0.10
    # But clDice loss is severe (> 0.50) because the skeleton sensitivity dropped
    assert cld_loss > 0.50
    assert cld_loss > 5.0 * d_loss


def test_all_zero_valid_mask_no_nan():
    """Codex P2 finding: All-zero valid_mask must not produce NaNs or crash."""
    H, W = 64, 64
    target = np.zeros((H, W), dtype=np.uint8)
    target[20:40, 20:40] = 1
    skel = target.copy()
    bnd = target.copy()
    ctr = target.astype(np.float32)
    off = np.zeros((2, H, W), dtype=np.float32)

    pred = np.full((H, W), 0.5, dtype=np.float32)
    pred_off = np.zeros((2, H, W), dtype=np.float32)
    empty_valid = np.zeros((H, W), dtype=np.uint8)

    losses = compute_compound_loss(
        fg_prob=pred,
        boundary_prob=pred,
        center_prob=pred,
        offset_pred=pred_off,
        target_fg=target,
        target_skel=skel,
        target_boundary=bnd,
        target_center=ctr,
        target_offset=off,
        valid_mask=empty_valid
    )

    for k, v in losses.items():
        assert not np.isnan(v), f"Loss key {k} is NaN on all-zero valid_mask!"
        assert v == 0.0


def test_compound_loss_keys_and_cldice_iters():
    H, W = 64, 64
    target = np.zeros((H, W), dtype=np.uint8)
    target[20:40, 20:40] = 1
    skel = target.copy()
    bnd = target.copy()
    ctr = target.astype(np.float32)
    off = np.zeros((2, H, W), dtype=np.float32)

    pred = np.full((H, W), 0.5, dtype=np.float32)
    pred_off = np.zeros((2, H, W), dtype=np.float32)

    losses = compute_compound_loss(
        fg_prob=pred,
        boundary_prob=pred,
        center_prob=pred,
        offset_pred=pred_off,
        target_fg=target,
        target_skel=skel,
        target_boundary=bnd,
        target_center=ctr,
        target_offset=off,
        cldice_iters=20
    )

    required_keys = [
        "loss_total", "loss_seg", "loss_bce", "loss_dice",
        "loss_cldice", "loss_boundary", "loss_center", "loss_offset"
    ]
    for k in required_keys:
        assert k in losses
        assert isinstance(losses[k], float)
        assert losses[k] >= 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pytorch_differentiable_autograd_loss():
    """Codex P1 finding #1: PyTorch loss must preserve backward autograd gradients on every prediction head."""
    B, H, W = 2, 64, 64

    # Model logits with requires_grad=True
    fg_logits = torch.randn(B, 1, H, W, requires_grad=True)
    bnd_logits = torch.randn(B, 1, H, W, requires_grad=True)
    ctr_logits = torch.randn(B, 1, H, W, requires_grad=True)
    off_pred = torch.randn(B, 2, H, W, requires_grad=True)

    # Targets
    target_fg = torch.zeros(B, 1, H, W)
    target_fg[:, :, 20:40, 20:40] = 1.0
    target_skel = target_fg.clone()
    target_bnd = target_fg.clone()
    target_ctr = target_fg.clone()
    target_off = torch.zeros(B, 2, H, W)

    criterion = CompoundTopologyLoss(cldice_iters=5)
    loss, stats = criterion(
        fg_logits=fg_logits,
        bnd_logits=bnd_logits,
        ctr_logits=ctr_logits,
        off_pred=off_pred,
        target_fg=target_fg,
        target_skel=target_skel,
        target_bnd=target_bnd,
        target_ctr=target_ctr,
        target_off=target_off
    )

    assert loss.requires_grad
    loss.backward()

    # Check finite non-zero gradients on all 4 heads
    assert fg_logits.grad is not None and torch.isfinite(fg_logits.grad).all() and fg_logits.grad.abs().sum() > 0
    assert bnd_logits.grad is not None and torch.isfinite(bnd_logits.grad).all() and bnd_logits.grad.abs().sum() > 0
    assert ctr_logits.grad is not None and torch.isfinite(ctr_logits.grad).all() and ctr_logits.grad.abs().sum() > 0
    assert off_pred.grad is not None and torch.isfinite(off_pred.grad).all() and off_pred.grad.abs().sum() > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_offset_batch_sample_isolation_regression():
    """Codex follow-up P1 #1: Ensure offset error broadcasting does NOT mix samples across batch."""
    from src.losses import CompoundTopologyLoss

    B, H, W = 2, 4, 4
    # Sample 0 has foreground ONLY at (0, 0)
    # Sample 1 has foreground ONLY at (3, 3)
    target_fg = torch.zeros(B, 1, H, W)
    target_fg[0, 0, 0, 0] = 1.0
    target_fg[1, 0, 3, 3] = 1.0

    target_off = torch.zeros(B, 2, H, W)

    # Predictions
    off_pred = torch.zeros(B, 2, H, W, requires_grad=True)
    with torch.no_grad():
        off_pred[0] = 1.0
        off_pred[1] = 2.0

    # Dummy logits
    fg_logits = torch.zeros(B, 1, H, W, requires_grad=True)
    bnd_logits = torch.zeros(B, 1, H, W, requires_grad=True)
    ctr_logits = torch.zeros(B, 1, H, W, requires_grad=True)

    criterion = CompoundTopologyLoss(alpha=0, beta=0, gamma=0, w_bnd=0, w_ctr=0, w_off=1.0)
    loss, stats = criterion(
        fg_logits=fg_logits,
        bnd_logits=bnd_logits,
        ctr_logits=ctr_logits,
        off_pred=off_pred,
        target_fg=target_fg,
        target_skel=target_fg,
        target_bnd=target_fg,
        target_ctr=target_fg,
        target_off=target_off
    )

    loss.backward()

    # Sample 0 background at (3, 3) MUST have ZERO gradient!
    assert off_pred.grad[0, :, 3, 3].abs().sum().item() == 0.0, "Sample 0 received background gradients at (3,3) from sample 1!"
    # Sample 1 background at (0, 0) MUST have ZERO gradient!
    assert off_pred.grad[1, :, 0, 0].abs().sum().item() == 0.0, "Sample 1 received background gradients at (0,0) from sample 0!"

    # Sample 0 foreground at (0, 0) must have non-zero gradient
    assert off_pred.grad[0, :, 0, 0].abs().sum().item() > 0.0
    # Sample 1 foreground at (3, 3) must have non-zero gradient
    assert off_pred.grad[1, :, 3, 3].abs().sum().item() > 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_dice_loss_fp16_reduction_safety():
    """Codex follow-up P1 #2: Dice on native patch size (768x768) in FP16 must not overflow spatial sums."""
    from src.losses import TorchSoftDiceLoss

    B, H, W = 1, 768, 768
    # Logits resulting in probability 0.5 everywhere
    pred_fp16 = torch.zeros(B, 1, H, W, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(B, 1, H, W, dtype=torch.float16)
    target[:, :, 100:120, 100:120] = 1.0  # 20x20 foreground

    criterion = TorchSoftDiceLoss(from_logits=True)
    loss_fp16 = criterion(pred_fp16, target)

    # In FP16, spatial sum of 0.5 * 768 * 768 = 294,912 overflows FP16 max (65,504) if not promoted to FP32!
    assert torch.isfinite(loss_fp16), "Dice loss overflowed in FP16!"
    assert loss_fp16.item() < 0.9999, f"Dice loss {loss_fp16.item()} saturated at 1.0!"

    loss_fp16.backward()
    assert pred_fp16.grad is not None
    assert torch.isfinite(pred_fp16.grad).all()
    assert pred_fp16.grad.abs().sum().item() > 0.0, "Zero gradient in FP16 Dice!"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_cldice_empty_gt_sample_invariance():
    """Codex follow-up P2 #3: Non-empty sample clDice loss must NOT be diluted when an empty-GT sample is appended."""
    from src.losses import TorchSoftclDiceLoss

    H, W = 64, 64
    criterion = TorchSoftclDiceLoss(iters=5, from_logits=True)

    # Non-empty sample
    pred_single = torch.randn(1, 1, H, W)
    target_single = torch.zeros(1, 1, H, W)
    target_single[:, :, 20:40, 20:40] = 1.0
    skel_single = target_single.clone()

    loss_single = criterion(pred_single, skel_single, target_single).item()

    # Batch of 2: sample 0 is the same non-empty sample, sample 1 is an empty-GT negative patch
    pred_batched = torch.cat([pred_single, torch.randn(1, 1, H, W)], dim=0)
    target_batched = torch.cat([target_single, torch.zeros(1, 1, H, W)], dim=0)
    skel_batched = target_batched.clone()

    loss_batched = criterion(pred_batched, skel_batched, target_batched).item()

    # The empty sample must be masked out and NOT dilute the loss of sample 0!
    assert np.isclose(loss_single, loss_batched, atol=1e-5), f"Single loss {loss_single} != Batched loss {loss_batched}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_compound_loss_all_invalid_batch_graph_connection():
    """Codex follow-up P2 #4: Entirely invalid batch must produce graph-connected zero with finite 0.0 grads on all heads."""
    from src.losses import CompoundTopologyLoss

    B, H, W = 2, 32, 32
    fg_logits = torch.randn(B, 1, H, W, requires_grad=True)
    bnd_logits = torch.randn(B, 1, H, W, requires_grad=True)
    ctr_logits = torch.randn(B, 1, H, W, requires_grad=True)
    off_pred = torch.randn(B, 2, H, W, requires_grad=True)

    target = torch.zeros(B, 1, H, W)
    valid_mask = torch.zeros(B, 1, H, W)  # Entirely invalid

    criterion = CompoundTopologyLoss(cldice_iters=3)
    loss, _ = criterion(
        fg_logits=fg_logits,
        bnd_logits=bnd_logits,
        ctr_logits=ctr_logits,
        off_pred=off_pred,
        target_fg=target,
        target_skel=target,
        target_bnd=target,
        target_ctr=target,
        target_off=torch.zeros(B, 2, H, W),
        valid_mask=valid_mask
    )

    loss.backward()

    # Must be 0.0 tensor, NOT None!
    for name, tensor in [("fg", fg_logits), ("bnd", bnd_logits), ("ctr", ctr_logits), ("off", off_pred)]:
        assert tensor.grad is not None, f"{name} grad is None on invalid batch!"
        assert (tensor.grad == 0.0).all(), f"{name} grad has non-zero values on invalid batch!"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_fp16_zero_branch_safety_at_scale():
    """Feedback #2 [P1] #1: FP16 zero branches on native patch size (768x768) must return finite 0.0 and finite 0.0 grads without NaN."""
    from src.losses import CompoundTopologyLoss, TorchSoftDiceLoss, TorchSoftclDiceLoss

    B, C, H, W = 1, 1, 768, 768
    # All heads in half precision
    fg_logits = torch.ones(B, C, H, W, dtype=torch.float16, requires_grad=True)
    bnd_logits = torch.ones(B, C, H, W, dtype=torch.float16, requires_grad=True)
    ctr_logits = torch.ones(B, C, H, W, dtype=torch.float16, requires_grad=True)
    off_pred = torch.ones(B, 2, H, W, dtype=torch.float16, requires_grad=True)

    zero_target = torch.zeros(B, C, H, W, dtype=torch.float16)
    zero_off = torch.zeros(B, 2, H, W, dtype=torch.float16)
    zero_valid = torch.zeros(B, C, H, W, dtype=torch.float16)

    # 1. Standalone TorchSoftDiceLoss on zero valid mask
    dice = TorchSoftDiceLoss()
    loss_dice = dice(fg_logits, zero_target, valid_mask=zero_valid)
    assert not torch.isnan(loss_dice), "TorchSoftDiceLoss produced NaN on FP16 zero valid mask!"
    assert loss_dice.item() == 0.0
    loss_dice.backward()
    assert fg_logits.grad is not None and torch.isfinite(fg_logits.grad).all()
    assert (fg_logits.grad == 0.0).all()
    fg_logits.grad = None

    # 2. Standalone TorchSoftclDiceLoss on empty GT
    cldice = TorchSoftclDiceLoss(iters=3)
    loss_cldice = cldice(fg_logits, zero_target, zero_target, valid_mask=None)
    assert not torch.isnan(loss_cldice), "TorchSoftclDiceLoss produced NaN on FP16 empty GT!"
    assert loss_cldice.item() == 0.0
    loss_cldice.backward()
    assert fg_logits.grad is not None and torch.isfinite(fg_logits.grad).all()
    assert (fg_logits.grad == 0.0).all()
    fg_logits.grad = None

    # 3. CompoundTopologyLoss on zero valid mask
    comp = CompoundTopologyLoss(cldice_iters=3)
    loss_comp, stats = comp(
        fg_logits, bnd_logits, ctr_logits, off_pred,
        zero_target, zero_target, zero_target, zero_target, zero_off,
        valid_mask=zero_valid
    )
    assert not torch.isnan(loss_comp), "CompoundTopologyLoss produced NaN on FP16 zero valid mask!"
    assert loss_comp.item() == 0.0
    loss_comp.backward()
    for name, t in [("fg", fg_logits), ("bnd", bnd_logits), ("ctr", ctr_logits), ("off", off_pred)]:
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"{name} grad has non-finite values in FP16 zero branch!"
        assert (t.grad == 0.0).all(), f"{name} grad is not zero in FP16 zero branch!"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_valid_mask_shape_canonicalization_and_sample_isolation():
    """Feedback #2 [P1] #3: valid_mask shape canonicalization and cross-sample isolation for all loss heads."""
    from src.losses import CompoundTopologyLoss

    B, H, W = 2, 4, 4
    zero_logits = torch.zeros(B, 1, H, W, requires_grad=True)
    targets = torch.zeros(B, 1, H, W)
    off_pred = torch.zeros(B, 2, H, W, requires_grad=True)
    off_tgt = torch.zeros(B, 2, H, W)

    # Sample 0 valid, sample 1 invalid
    m_b1hw = torch.zeros(B, 1, H, W)
    m_b1hw[0] = 1.0
    m_bhw = torch.zeros(B, H, W)
    m_bhw[0] = 1.0

    comp = CompoundTopologyLoss(cldice_iters=3)

    # 1. Verify equivalence of [B, 1, H, W] and [B, H, W] shapes
    l1, stats1 = comp(zero_logits, zero_logits, zero_logits, off_pred, targets, targets, targets, targets, off_tgt, valid_mask=m_b1hw)
    l2, stats2 = comp(zero_logits, zero_logits, zero_logits, off_pred, targets, targets, targets, targets, off_tgt, valid_mask=m_bhw)

    assert torch.isclose(l1, l2), f"Mask [B,1,H,W] loss {l1.item()} != [B,H,W] loss {l2.item()}! Sample broadcasting bug!"
    assert np.isclose(stats1["loss_bce"], stats2["loss_bce"])
    assert np.isclose(stats1["loss_bce"], 0.693147, atol=1e-4)

    # 2. Rejection of invalid mask shapes
    bad_mask = torch.zeros(B, 2, H, W)  # Invalid channel count (must be 1)
    with pytest.raises(ValueError, match="must have channel size 1"):
        comp(zero_logits, zero_logits, zero_logits, off_pred, targets, targets, targets, targets, off_tgt, valid_mask=bad_mask)

    bad_spatial = torch.zeros(B, 1, H + 1, W)
    with pytest.raises(ValueError, match="does not match expected"):
        comp(zero_logits, zero_logits, zero_logits, off_pred, targets, targets, targets, targets, off_tgt, valid_mask=bad_spatial)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_strict_head_and_target_shape_validation_and_rejections():
    """Feedback #3 [P2] #2: Validate all prediction heads and targets at entry (batch, channel, spatial), prohibiting silent broadcasting."""
    from src.losses import CompoundTopologyLoss, TorchSoftDiceLoss, TorchSoftclDiceLoss

    B, H, W = 2, 4, 4
    fg_logits = torch.zeros(B, 1, H, W, requires_grad=True)
    bnd_logits = torch.zeros(B, 1, H, W, requires_grad=True)
    ctr_logits = torch.zeros(B, 1, H, W, requires_grad=True)
    off_pred = torch.zeros(B, 2, H, W, requires_grad=True)

    target_fg = torch.zeros(B, 1, H, W)
    target_skel = torch.zeros(B, 1, H, W)
    target_bnd = torch.zeros(B, 1, H, W)
    target_ctr = torch.zeros(B, 1, H, W)
    target_off = torch.zeros(B, 2, H, W)

    comp = CompoundTopologyLoss(cldice_iters=3)

    # 1. Normal successful call
    loss, stats = comp(fg_logits, bnd_logits, ctr_logits, off_pred, target_fg, target_skel, target_bnd, target_ctr, target_off)
    assert torch.isfinite(loss)

    # 2. Reproduction of Codex feedback #3: mismatched batch in target_ctr (shape 1,1,4,4 instead of 2,1,4,4)
    bad_batch_ctr = torch.zeros(1, 1, H, W)
    with pytest.raises(ValueError, match="target_ctr shape .* does not match expected"):
        comp(fg_logits, bnd_logits, ctr_logits, off_pred, target_fg, target_skel, target_bnd, bad_batch_ctr, target_off)

    # 3. Mismatched channel in target_off (shape 1,1,4,4 or 2,1,4,4 instead of 2,2,4,4)
    bad_chan_off = torch.zeros(B, 1, H, W)
    with pytest.raises(ValueError, match="target_off must have shape \\[B, 2, H, W\\]"):
        comp(fg_logits, bnd_logits, ctr_logits, off_pred, target_fg, target_skel, target_bnd, target_ctr, bad_chan_off)

    # 4. Mismatched spatial shape in target_skel
    bad_spatial_skel = torch.zeros(B, 1, H + 2, W)
    with pytest.raises(ValueError, match="target_skel shape .* does not match expected"):
        comp(fg_logits, bnd_logits, ctr_logits, off_pred, target_fg, bad_spatial_skel, target_bnd, target_ctr, target_off)

    # 5. Wrong shape in prediction heads (e.g. off_pred shaped [B, 1, H, W] instead of [B, 2, H, W])
    bad_off_pred = torch.zeros(B, 1, H, W, requires_grad=True)
    with pytest.raises(ValueError, match="off_pred must have shape \\[B, 2, H, W\\]"):
        comp(fg_logits, bnd_logits, ctr_logits, bad_off_pred, target_fg, target_skel, target_bnd, target_ctr, target_off)

    # 6. Standalone TorchSoftDiceLoss with wrong target shape (batch mismatch)
    dice = TorchSoftDiceLoss()
    with pytest.raises(ValueError, match="target shape .* does not match expected"):
        dice(fg_logits, torch.zeros(1, 1, H, W))
    # Standalone TorchSoftDiceLoss with wrong channel count in 4D
    with pytest.raises(ValueError, match="pred must have channel size 1"):
        dice(torch.zeros(B, 3, H, W), target_fg)

    # 7. Standalone TorchSoftclDiceLoss with wrong shapes
    cldice = TorchSoftclDiceLoss(iters=3)
    with pytest.raises(ValueError, match="target_mask shape .* does not match expected"):
        cldice(fg_logits, target_skel, torch.zeros(1, 1, H, W))



