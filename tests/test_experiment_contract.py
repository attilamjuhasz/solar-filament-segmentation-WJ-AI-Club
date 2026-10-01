"""
Tests for experiment contract safety, gradient accumulation normalization,
and resume reproducibility as prioritized by Codex.
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import compute_instance_diagnostics, evaluate_oof
from inference import run_inference
from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.dataset import SolarFilamentDataset
from src.inference.engine import compute_file_sha256, predict_full_observation
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import audit_submission_and_manifest, encode_instance, write_submission_csv
from src.models import ResNet34UNet
from train import save_checkpoint, train_one_epoch


def test_wrong_fold_and_unverified_provenance_rejection():
    """Verify that evaluator rejects mismatched fold and unverified provenance unless explicitly authorized."""
    model = ResNet34UNet(in_channels=3, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Checkpoint with fold 0
        ckpt_path_fold0 = Path(tmp_dir) / "ckpt_fold0.pt"
        save_checkpoint(
            path=ckpt_path_fold0,
            epoch=1,
            model=model,
            optimizer=optimizer,
            config={"model": {"name": "resnet34_unet", "backbone": "resnet34", "in_channels": 3, "pretrained": False}},
            fold=0,
            best_metric=0.5,
            train_observations=["obs_train_1"],
            val_observations=["obs_val_1"],
        )

        # Evaluator requested fold=1 on checkpoint fold=0 must raise ValueError
        with pytest.raises(ValueError, match="Fold mismatch! Checkpoint was trained on fold 0, but evaluation requested fold 1"):
            evaluate_oof(
                checkpoint_path=str(ckpt_path_fold0),
                fold=1,
                limit=1,
            )

        # 2. Checkpoint lacking fold provenance
        ckpt_path_no_fold = Path(tmp_dir) / "ckpt_no_fold.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "config": {"model": {"name": "resnet34_unet", "backbone": "resnet34", "in_channels": 3, "pretrained": False}},
            },
            str(ckpt_path_no_fold),
        )

        # Must raise ValueError unless --allow-unverified-provenance is passed
        with pytest.raises(ValueError, match="lacks complete authenticated provenance metadata"):
            evaluate_oof(
                checkpoint_path=str(ckpt_path_no_fold),
                fold=0,
                allow_unverified_provenance=False,
                limit=1,
            )


def test_missing_and_invalid_fold_assignment_rejection():
    """Verify that SolarFilamentDataset rejects missing or invalid fold assignments."""
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_images = data_dir / "train" / "train_images"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    if not train_json.is_file() or not train_images.is_dir():
        pytest.skip("Local training data not present")

    # Pass an incomplete fold assignment dict with only 1 observation
    incomplete_assignments = {"20150125172714 Mh": 0}

    with pytest.raises(KeyError, match="is missing from fold assignments"):
        SolarFilamentDataset(
            images_dir=train_images,
            annotations_json=train_json,
            fold_assignments=incomplete_assignments,
            target_fold=0,
            is_train=True,
            patch_size=(256, 256),
        )

    # Pass invalid fold assignment (negative integer)
    from src.data.annotations import load_coco_annotations
    ann_idx = load_coco_annotations(str(train_json))
    all_obs = list(ann_idx.by_observation.keys())
    invalid_assignments = {obs: 0 for obs in all_obs}
    invalid_assignments[all_obs[0]] = -1

    with pytest.raises(ValueError, match="Invalid fold assignment -1"):
        SolarFilamentDataset(
            images_dir=train_images,
            annotations_json=train_json,
            fold_assignments=invalid_assignments,
            target_fold=0,
            is_train=True,
            patch_size=(256, 256),
        )


def test_partial_gradient_accumulation_normalization():
    """
    Verify exact parameter update under partial gradient accumulation:
    3 batches with unit gradients, SGD lr=1.0, grad_accum_steps=2.
    Correct update: -2.0. Unnormalized division by fixed 2: -1.5.
    """
    w = nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.SGD([w], lr=1.0)

    total_loader_batches = 3
    grad_accum_steps = 2

    # Simulate 3 steps
    for step in range(total_loader_batches):
        loss = w * 1.0  # dloss/dw = 1.0

        current_window = min(
            grad_accum_steps,
            total_loader_batches - (step // grad_accum_steps) * grad_accum_steps,
        )
        scaled_loss = loss / current_window
        scaled_loss.backward()

        is_accum_boundary = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == total_loader_batches)
        if is_accum_boundary:
            optimizer.step()
            optimizer.zero_grad()

    # Step 0: scaled by 2 -> grad = 0.5
    # Step 1: scaled by 2 -> grad = 0.5 (sum = 1.0) -> step: w = 0 - 1.0*1.0 = -1.0
    # Step 2: remainder window = 1 -> scaled by 1 -> grad = 1.0 -> step: w = -1.0 - 1.0*1.0 = -2.0
    assert w.item() == pytest.approx(-2.0, abs=1e-6)


def test_two_step_interrupted_resumed_optimization_exactness():
    """Verify that saving and resuming training produces exact identical parameters, optimizer state, and loss."""
    def create_setup():
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
        return m, opt, sched

    # Synthetic batches
    torch.manual_seed(100)
    batch1_x = torch.randn(4, 8)
    batch1_y = torch.randint(0, 2, (4,))
    batch2_x = torch.randn(4, 8)
    batch2_y = torch.randint(0, 2, (4,))
    loss_fn = nn.CrossEntropyLoss()

    # --- Run Continuous (2 steps) ---
    model_cont, opt_cont, sched_cont = create_setup()

    # Step 1
    opt_cont.zero_grad()
    loss1_cont = loss_fn(model_cont(batch1_x), batch1_y)
    loss1_cont.backward()
    opt_cont.step()
    sched_cont.step()

    # Step 2
    opt_cont.zero_grad()
    loss2_cont = loss_fn(model_cont(batch2_x), batch2_y)
    loss2_cont.backward()
    opt_cont.step()
    sched_cont.step()

    # --- Run Interrupted -> Resumed ---
    model_res, opt_res, sched_res = create_setup()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_file = Path(tmp_dir) / "checkpoint_step1.pt"

        # Step 1
        opt_res.zero_grad()
        loss1_res = loss_fn(model_res(batch1_x), batch1_y)
        loss1_res.backward()
        opt_res.step()
        sched_res.step()

        # Save checkpoint after Step 1
        save_checkpoint(
            path=ckpt_file,
            epoch=1,
            model=model_res,
            optimizer=opt_res,
            scheduler=sched_res,
            scaler=None,
            config={"dummy": True},
            fold=0,
            best_metric=loss1_res.item(),
            global_step=1,
        )

        # Fresh model/opt/sched restored from checkpoint
        model_restored, opt_restored, sched_restored = create_setup()
        ckpt_data = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)

        model_restored.load_state_dict(ckpt_data["model_state_dict"])
        opt_restored.load_state_dict(ckpt_data["optimizer_state_dict"])
        sched_restored.load_state_dict(ckpt_data["scheduler_state_dict"])
        torch.set_rng_state(ckpt_data["torch_rng_state"])
        np.random.set_state(ckpt_data["numpy_rng_state"])
        random.setstate(ckpt_data["random_rng_state"])

        # Execute Step 2 on restored setup
        opt_restored.zero_grad()
        loss2_res = loss_fn(model_restored(batch2_x), batch2_y)
        loss2_res.backward()
        opt_restored.step()
        sched_restored.step()

    # Verify exact loss matching
    assert loss2_cont.item() == pytest.approx(loss2_res.item(), abs=1e-7)

    # Verify exact parameter matching
    for p_cont, p_res in zip(model_cont.parameters(), model_restored.parameters()):
        assert torch.allclose(p_cont, p_res, atol=1e-7)

    # Verify exact optimizer state matching
    cont_opt_state = opt_cont.state_dict()["state"]
    res_opt_state = opt_restored.state_dict()["state"]
    for k in cont_opt_state:
        for inner_k in ["exp_avg", "exp_avg_sq"]:
            assert torch.allclose(cont_opt_state[k][inner_k], res_opt_state[k][inner_k], atol=1e-7)


def test_predict_instances_csv_manifest_positive_and_empty_audit():
    """Verify predict -> instance extraction -> CSV -> manifest pipeline for positive and empty predictions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_pos = tmp_path / "submission_pos.csv"
        manifest_pos = tmp_path / "manifest_pos.json"
        csv_empty = tmp_path / "submission_empty.csv"
        manifest_empty = tmp_path / "manifest_empty.json"

        # Case 1: Positive detection
        fg_map = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)
        ctr_map = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)
        bnd_map = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)

        # Put a 20x20 foreground square
        fg_map[500:520, 500:520] = 0.95
        ctr_map[508:512, 508:512] = 0.90

        obs_canonical = "20150125172714Mh"
        instances = extract_instances_from_maps(
            foreground_prob=fg_map,
            center_prob=ctr_map,
            boundary_prob=bnd_map,
            obs_id=obs_canonical,
            method="connected_components",
        )
        assert len(instances) >= 1

        rows_pos = [(inst.filament_id, inst.rle_counts) for inst in instances]
        write_submission_csv(str(csv_pos), rows_pos)
        pos_sha256 = compute_file_sha256(csv_pos)

        with open(manifest_pos, "w", encoding="utf-8") as f:
            json.dump({
                "csv_sha256": pos_sha256,
                "total_instances": len(instances),
                "observations": [
                    {
                        "observation_id": obs_canonical,
                        "status": "processed",
                        "instance_count": len(instances),
                    }
                ],
            }, f)

        audit_pos = audit_submission_and_manifest(
            csv_path=str(csv_pos),
            manifest_path=str(manifest_pos),
            expected_observation_ids={obs_canonical},
            verify_rle_decoding=True,
        )
        assert audit_pos["is_valid"] is True
        assert audit_pos["total_instances"] == len(instances)

        # Case 2: Empty detection (all abstained)
        write_submission_csv(str(csv_empty), [])
        empty_sha256 = compute_file_sha256(csv_empty)

        with open(manifest_empty, "w", encoding="utf-8") as f:
            json.dump({
                "csv_sha256": empty_sha256,
                "total_instances": 0,
                "observations": [
                    {
                        "observation_id": obs_canonical,
                        "status": "abstained",
                        "instance_count": 0,
                    }
                ],
            }, f)

        audit_empty = audit_submission_and_manifest(
            csv_path=str(csv_empty),
            manifest_path=str(manifest_empty),
            expected_observation_ids={obs_canonical},
            verify_rle_decoding=True,
        )
        assert audit_empty["is_valid"] is True
        assert audit_empty["total_instances"] == 0
        assert audit_empty["csv_report"]["unique_observations"] == 0


class _UnitGradDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return {
            "fg_logits": self.w.view(1, 1, 1, 1),
            "bnd_logits": torch.zeros(1, 1, 1, 1),
            "ctr_logits": torch.zeros(1, 1, 1, 1),
            "off_pred": torch.zeros(1, 2, 1, 1),
        }


class _UnitGradCriterion(nn.Module):
    def forward(self, fg_logits, **kwargs):
        loss = fg_logits.sum() * 1.0
        return loss, {
            "loss_total": loss.item(),
            "loss_seg": loss.item(),
            "loss_bce": 0.0,
            "loss_dice": 0.0,
            "loss_cldice": 0.0,
            "loss_boundary": 0.0,
            "loss_center": 0.0,
            "loss_offset": 0.0,
        }


class _DummyBatchDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 3):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.zeros(3, 4, 4),
            "target_fg": torch.zeros(1, 4, 4),
            "target_skel": torch.zeros(1, 4, 4),
            "target_bnd": torch.zeros(1, 4, 4),
            "target_ctr": torch.zeros(1, 4, 4),
            "target_off": torch.zeros(2, 4, 4),
            "valid_mask": torch.ones(1, 4, 4),
        }


def test_production_train_one_epoch_gradient_accumulation():
    """Verify that production train_one_epoch correctly normalizes partial gradient accumulation windows."""
    model = _UnitGradDummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loader = DataLoader(_DummyBatchDataset(size=3), batch_size=1, shuffle=False)

    train_one_epoch(
        model=model,
        loader=loader,
        criterion=_UnitGradCriterion(),
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        grad_accum_steps=2,
        max_steps=3,
    )
    # 3 batches, accum 2: batch0 scaled by 2, batch1 scaled by 2 (accum boundary -> step -1.0)
    # batch2 scaled by 1 (remainder window 1, accum boundary -> step -1.0). Total: -2.0.
    assert model.w.item() == pytest.approx(-2.0, abs=1e-6)


def test_train_one_epoch_nonpositive_controls_rejection():
    """Verify train_one_epoch rejects nonpositive grad_accum_steps and max_steps."""
    model = _UnitGradDummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loader = DataLoader(_DummyBatchDataset(size=1), batch_size=1, shuffle=False)

    with pytest.raises(ValueError, match="grad_accum_steps must be positive integer"):
        train_one_epoch(model, loader, _UnitGradCriterion(), optimizer, None, torch.device("cpu"), grad_accum_steps=0)

    with pytest.raises(ValueError, match="grad_accum_steps must be positive integer"):
        train_one_epoch(model, loader, _UnitGradCriterion(), optimizer, None, torch.device("cpu"), grad_accum_steps=-1)

    with pytest.raises(ValueError, match="max_steps must be positive integer"):
        train_one_epoch(model, loader, _UnitGradCriterion(), optimizer, None, torch.device("cpu"), grad_accum_steps=1, max_steps=0)

    with pytest.raises(ValueError, match="max_steps must be positive integer"):
        train_one_epoch(model, loader, _UnitGradCriterion(), optimizer, None, torch.device("cpu"), grad_accum_steps=1, max_steps=-3)


def test_compute_instance_diagnostics_whole_disk_adversarial():
    """Verify that a whole-disk prediction over a tiny GT instance yields 1 missed GT and 1 spurious pred (0 match)."""
    gt_mask = np.zeros((1000, 1000), dtype=np.uint8)
    gt_mask[490:510, 490:510] = 1

    pred_mask = np.zeros((1000, 1000), dtype=np.uint8)
    pred_mask[100:900, 100:900] = 1

    stats = compute_instance_diagnostics(gt_masks=[gt_mask], pred_masks=[pred_mask], iou_thresh=0.50)
    assert stats["n_gt"] == 1
    assert stats["n_pred"] == 1
    assert stats["strict_matched_count"] == 0
    assert stats["missed_gt_count"] == 1
    assert stats["spurious_pred_count"] == 1
    assert stats["strict_unmatched_fn"] == 1
    assert stats["strict_unmatched_fp"] == 1


def test_dataset_rng_state_preservation_and_restoration():
    """Verify dataset RNG state preservation and restoration across resume."""
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_images = data_dir / "train" / "train_images"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    if not train_json.is_file() or not train_images.is_dir():
        pytest.skip("Local training data not present")

    from src.data.annotations import load_coco_annotations
    ann_idx = load_coco_annotations(str(train_json))
    all_obs = sorted(list(ann_idx.by_observation.keys()))
    fold_assignments = {obs: 0 for obs in all_obs}

    ds1 = SolarFilamentDataset(
        images_dir=train_images,
        annotations_json=train_json,
        fold_assignments=fold_assignments,
        target_fold=0,
        is_train=False,
        seed=12345,
    )
    _ = ds1.rng.rand(10)
    saved_state = ds1.rng.get_state()
    expected_next = ds1.rng.rand(5)

    ds2 = SolarFilamentDataset(
        images_dir=train_images,
        annotations_json=train_json,
        fold_assignments=fold_assignments,
        target_fold=0,
        is_train=False,
        seed=99999,
    )
    ds2.rng.set_state(saved_state)
    actual_next = ds2.rng.rand(5)

    np.testing.assert_allclose(actual_next, expected_next)


def test_solar_filament_dataset_strict_fold_bounds_and_bool_rejections():
    """Verify that SolarFilamentDataset rejects boolean target_fold, out-of-bound fold numbers, and invalid assignments."""
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_images = data_dir / "train" / "train_images"
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    if not train_json.is_file() or not train_images.is_dir():
        pytest.skip("Local training data not present")

    with pytest.raises(ValueError, match="Invalid target_fold True"):
        SolarFilamentDataset(images_dir=train_images, annotations_json=train_json, target_fold=True)

    with pytest.raises(ValueError, match="Invalid target_fold False"):
        SolarFilamentDataset(images_dir=train_images, annotations_json=train_json, target_fold=False)

    with pytest.raises(ValueError, match="Invalid target_fold 5"):
        SolarFilamentDataset(images_dir=train_images, annotations_json=train_json, target_fold=5)

    with pytest.raises(ValueError, match="Invalid target_fold -1"):
        SolarFilamentDataset(images_dir=train_images, annotations_json=train_json, target_fold=-1)

    from src.data.annotations import load_coco_annotations
    ann_idx = load_coco_annotations(str(train_json))
    all_obs = sorted(list(ann_idx.by_observation.keys()))

    bool_assignments = {obs: 0 for obs in all_obs}
    bool_assignments[all_obs[0]] = True
    with pytest.raises(ValueError, match="Invalid fold assignment True"):
        SolarFilamentDataset(images_dir=train_images, annotations_json=train_json, fold_assignments=bool_assignments, target_fold=0)

    out_assignments = {obs: 0 for obs in all_obs}
    out_assignments[all_obs[0]] = 5
    with pytest.raises(ValueError, match="Invalid fold assignment 5"):
        SolarFilamentDataset(images_dir=train_images, annotations_json=train_json, fold_assignments=out_assignments, target_fold=0)
