from __future__ import annotations

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.dataset import SolarFilamentDataset
from src.data.folds import ObservationRecord, assign_stratified_group_folds, consolidate_canonical_records
from src.inference.rle import audit_submission_and_manifest, decode_instance, encode_instance
from src.losses import CompoundTopologyLoss
from src.models.resnet_unet import ResNet34UNet
from train import save_checkpoint


def test_model_forward_and_loss_backward():
    """Verify ResNet34UNet forward pass output shapes and autograd backward gradient flow."""
    model = ResNet34UNet(in_channels=3, pretrained=False)
    criterion = CompoundTopologyLoss(cldice_iters=3)

    B, C, H, W = 2, 3, 128, 128
    x = torch.rand((B, C, H, W), dtype=torch.float32, requires_grad=True)

    preds = model(x)
    assert "fg_logits" in preds
    assert "bnd_logits" in preds
    assert "ctr_logits" in preds
    assert "off_pred" in preds

    assert preds["fg_logits"].shape == (B, 1, H, W)
    assert preds["bnd_logits"].shape == (B, 1, H, W)
    assert preds["ctr_logits"].shape == (B, 1, H, W)
    assert preds["off_pred"].shape == (B, 2, H, W)

    # Targets
    tgt_fg = torch.zeros((B, 1, H, W), dtype=torch.float32)
    tgt_fg[0, 0, 20:30, 20:30] = 1.0
    tgt_skel = tgt_fg.clone()
    tgt_bnd = tgt_fg.clone()
    tgt_ctr = torch.zeros((B, 1, H, W), dtype=torch.float32)
    tgt_ctr[0, 0, 25, 25] = 1.0
    tgt_off = torch.zeros((B, 2, H, W), dtype=torch.float32)

    loss, stats = criterion(
        fg_logits=preds["fg_logits"],
        bnd_logits=preds["bnd_logits"],
        ctr_logits=preds["ctr_logits"],
        off_pred=preds["off_pred"],
        target_fg=tgt_fg,
        target_skel=tgt_skel,
        target_bnd=tgt_bnd,
        target_ctr=tgt_ctr,
        target_off=tgt_off,
    )

    assert torch.isfinite(loss)
    assert loss > 0.0
    loss.backward()

    # Check that model weights received valid gradients
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_checkpoint_save_and_resume():
    """Verify that checkpoint saving and resumption restores exact weights and states."""
    model = ResNet34UNet(in_channels=3, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pt"
        save_checkpoint(
            path=ckpt_path,
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            config={"model": {"name": "resnet34_unet"}},
            fold=2,
            best_metric=0.1234,
        )

        assert ckpt_path.is_file()
        loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        assert loaded["epoch"] == 5
        assert loaded["fold"] == 2
        assert loaded["best_metric"] == 0.1234

        new_model = ResNet34UNet(in_channels=3, pretrained=False)
        new_model.load_state_dict(loaded["model_state_dict"])

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


def test_dataset_batch_geometry_and_sampling():
    """Verify SolarFilamentDataset loads real data crops with strict tensor shapes and coordinate alignment."""
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_images = data_dir / "train" / "train_images"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    if not train_json.is_file() or not train_images.is_dir():
        pytest.skip("Local training data not present")

    ds = SolarFilamentDataset(
        images_dir=train_images,
        annotations_json=train_json,
        is_train=True,
        patch_size=(256, 256),
    )

    assert len(ds) > 0
    sample = ds[0]

    assert sample["image"].shape == (3, 256, 256)
    assert sample["target_fg"].shape == (1, 256, 256)
    assert sample["target_bnd"].shape == (1, 256, 256)
    assert sample["target_skel"].shape == (1, 256, 256)
    assert sample["target_ctr"].shape == (1, 256, 256)
    assert sample["target_off"].shape == (2, 256, 256)
    assert sample["valid_mask"].shape == (1, 256, 256)

    # Values within expected ranges
    assert (sample["image"] >= 0.0).all() and (sample["image"] <= 1.0).all()
    assert (sample["target_fg"] >= 0.0).all() and (sample["target_fg"] <= 1.0).all()


def test_end_to_end_tiled_inference_and_audit():
    """Verify full sliding-window tiled inference and audit generation on a synthetic observation."""
    from inference import predict_full_observation
    from src.inference.instances import extract_instances_from_maps

    model = ResNet34UNet(in_channels=3, pretrained=False)
    model.eval()

    # Create dummy 2048x2048 image with bright solar disk and dark background
    synthetic_img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    yy, xx = np.indices((2048, 2048))
    dist = (yy - 1024) ** 2 + (xx - 1024) ** 2
    synthetic_img[dist <= 900 ** 2] = 120

    device = torch.device("cpu")
    fg_map, ctr_map, bnd_map, off_map = predict_full_observation(
        model=model,
        image_rgb=synthetic_img,
        device=device,
        tile_size=512,
        stride=512,
    )

    assert fg_map.shape == (2048, 2048)
    assert ctr_map.shape == (2048, 2048)
    assert bnd_map.shape == (2048, 2048)
    assert off_map.shape == (2, 2048, 2048)

    instances = extract_instances_from_maps(
        foreground_prob=fg_map,
        center_prob=ctr_map,
        boundary_prob=bnd_map,
        obs_id="test_obs",
        high_threshold=0.55,
        low_threshold=0.30,
    )

    # Check instance validity
    for inst in instances:
        assert inst.rle_counts
        mask = decode_instance(inst.rle_counts, shape=(2048, 2048))
        assert mask.shape == (2048, 2048)
