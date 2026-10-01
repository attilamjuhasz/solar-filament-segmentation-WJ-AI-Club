from pathlib import Path
import math
import numpy as np
import pytest
import torch

from src.contracts import NATIVE_IMAGE_SHAPE
from src.inference.config import DOCUMENTED_DEFAULTS, UNSET, resolve_inference_config
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import decode_instance
from evaluate import compute_instance_diagnostics
from src.evaluation.competition_adapter import evaluate_entry_pq


def test_resolver_with_checkpoint_postprocess_and_no_overrides():
    """Checkpoint with CC/high=.9/low=.7/area500 and no overrides resolves to those exact values."""
    ckpt_config = {
        "postprocess": {
            "method": "connected_components",
            "high_threshold": 0.90,
            "low_threshold": 0.70,
            "min_area": 500,
            "max_instances": 10,
        },
        "loss": {"center_weight": 0.0, "boundary_weight": 0.0},
    }
    resolved = resolve_inference_config(ckpt_config)
    assert resolved["method"] == "connected_components"
    assert resolved["high_threshold"] == 0.90
    assert resolved["low_threshold"] == 0.70
    assert resolved["min_area"] == 500
    assert resolved["max_instances"] == 10
    assert resolved["tile_size"] == DOCUMENTED_DEFAULTS["tile_size"]
    assert resolved["stride"] == DOCUMENTED_DEFAULTS["stride"]


def test_resolver_with_partial_override_and_unlimited_cap():
    """Setting only high=.92 overrides only high; max_instances=None explicitly preserves unlimited."""
    ckpt_config = {
        "postprocess": {
            "method": "connected_components",
            "high_threshold": 0.85,
            "low_threshold": 0.60,
            "min_area": 400,
            "max_instances": 12,
        }
    }
    overrides = {
        "high_threshold": 0.92,
        "max_instances": None,
    }
    resolved = resolve_inference_config(ckpt_config, overrides=overrides)
    assert resolved["high_threshold"] == 0.92
    assert resolved["low_threshold"] == 0.60
    assert resolved["min_area"] == 400
    assert resolved["max_instances"] is None  # explicitly unlimited


def test_resolver_rejects_unknown_keys_and_invalid_bounds():
    """Reject unknown parameters, invalid ordering, nonfinite values, and bools-as-ints."""
    with pytest.raises(ValueError, match="Unknown inference configuration parameter"):
        resolve_inference_config({}, overrides={"unsupported_key": 123})

    with pytest.raises(ValueError, match="Require 0.0 <= low_threshold <= high_threshold <= 1.0"):
        resolve_inference_config({}, overrides={"low_threshold": 0.90, "high_threshold": 0.50})

    with pytest.raises(ValueError, match="must be finite"):
        resolve_inference_config({}, overrides={"high_threshold": float("nan")})

    with pytest.raises(TypeError, match="must be an integer"):
        resolve_inference_config({}, overrides={"min_area": True})

    with pytest.raises(ValueError, match="Require stride <= tile_size"):
        resolve_inference_config({}, overrides={"tile_size": 256, "stride": 512})


def test_resolver_supervision_rejection_for_unsupervised_watershed():
    """Reject watershed when training supervision center_weight=0 or boundary_weight=0."""
    unsupervised_ckpt_cfg = {
        "loss": {
            "center_weight": 0.0,
            "boundary_weight": 0.0,
        }
    }
    with pytest.raises(ValueError, match="center_weight=0.0"):
        resolve_inference_config(unsupervised_ckpt_cfg, overrides={"method": "watershed"})

    semi_supervised = {
        "loss": {
            "center_weight": 1.0,
            "boundary_weight": 0.0,
        }
    }
    with pytest.raises(ValueError, match="boundary_weight=0.0"):
        resolve_inference_config(
            semi_supervised,
            overrides={"method": "watershed", "boundary_weight": 0.5}
        )


def test_connected_components_invariance_to_arbitrary_auxiliary_maps():
    """Pure connected components instance extraction must be strictly invariant to arbitrary auxiliary maps."""
    shape = (256, 256)
    np.random.seed(42)

    # Foreground map with a clear filament structure
    fg = np.zeros(shape, dtype=np.float32)
    fg[50:60, 20:200] = 0.95

    # Extraction with zero auxiliary maps
    insts_clean = extract_instances_from_maps(
        foreground_prob=fg,
        center_prob=None,
        boundary_prob=None,
        offset_field=None,
        obs_id="test_invariance",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=50,
        method="connected_components",
        shape=shape,
    )

    # Extraction with noisy auxiliary maps
    noisy_ctr = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
    noisy_bnd = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
    noisy_off = np.random.uniform(-10.0, 10.0, size=(2, *shape)).astype(np.float32)

    insts_noisy = extract_instances_from_maps(
        foreground_prob=fg,
        center_prob=noisy_ctr,
        boundary_prob=noisy_bnd,
        offset_field=noisy_off,
        obs_id="test_invariance",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=50,
        method="connected_components",
        shape=shape,
    )

    assert len(insts_clean) == len(insts_noisy) == 1
    assert insts_clean[0].rle_counts == insts_noisy[0].rle_counts
    assert insts_clean[0].bbox == insts_noisy[0].bbox


def test_single_spine_regression_and_marker_fragmentation():
    """Verify that a single continuous spine is preserved as 1 instance under CC,
    and demonstrate that unsupervised multiple marker peaks can shatter it under watershed.
    """
    H, W = 64, 128
    fg = np.zeros((H, W), dtype=np.float32)
    gt_mask = np.zeros((H, W), dtype=bool)

    # Continuous horizontal spine at rows 30:33, cols 10:118
    gt_mask[30:34, 10:118] = True
    fg[gt_mask] = 0.95

    # 1. Connected components produces exactly 1 instance matching GT
    insts_cc = extract_instances_from_maps(
        foreground_prob=fg,
        obs_id="spine_cc",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=16,
        method="connected_components",
        shape=(H, W),
    )
    assert len(insts_cc) == 1
    pred_mask_cc = decode_instance(insts_cc[0].rle_counts, shape=(H, W))
    np.testing.assert_array_equal(pred_mask_cc, gt_mask)

    # Evaluate CC with competition adapter
    total_iou, tp, fp, fn, o2m, m2o = evaluate_entry_pq([gt_mask], [pred_mask_cc])
    assert tp == 1
    assert fp == 0
    assert fn == 0
    assert total_iou == 1.0

    # 2. Watershed with multiple artificial center peaks shatters spine
    ctr = np.zeros((H, W), dtype=np.float32)
    for col in range(15, 115, 10):
        ctr[31:33, col:col+2] = 0.99

    insts_ws = extract_instances_from_maps(
        foreground_prob=fg,
        center_prob=ctr,
        boundary_prob=None,
        obs_id="spine_ws",
        high_threshold=0.85,
        low_threshold=0.60,
        center_threshold=0.35,
        marker_min_distance=5,
        min_area=1,
        max_peaks=20,
        method="watershed",
        shape=(H, W),
    )
    # Shattered into multiple fragments
    assert len(insts_ws) >= 5

    # The union of watershed predictions covers the GT, but each piece has IoU < 0.5 with the whole spine
    pred_masks_ws = [decode_instance(p.rle_counts, shape=(H, W)) for p in insts_ws]
    total_iou_ws, tp_ws, fp_ws, fn_ws, o2m_ws, m2o_ws = evaluate_entry_pq([gt_mask], pred_masks_ws)
    # All fragments fail the > 0.5 IoU threshold against the single whole spine
    assert tp_ws == 0
    assert fn_ws == 1
    assert fp_ws == len(insts_ws)


def test_exact_half_overlap_unmatched_strict_threshold():
    """Verify that an instance with exact 0.5 IoU (half overlap) remains unmatched (strict > 0.5)."""
    gt_mask = np.zeros((100, 100), dtype=bool)
    gt_mask[20:40, 20:40] = True  # area = 400

    # Prediction shares 200 pixels with GT and has 200 non-overlapping pixels:
    # intersection = 200, union = 400 + 400 - 200 = 600 -> IoU = 200/600 = 0.333
    # For exact 0.5 IoU:
    # If gt has area 100, and pred has area 100, and intersection is 50 -> union is 150 -> IoU is 50/150 = 0.333
    # For IoU = inter / (area_g + area_p - inter) = 0.5:
    # 2 * inter = area_g + area_p - inter => 3 * inter = area_g + area_p
    # Let area_g = 100, inter = 50, area_p = 50:
    # union = 100 + 50 - 50 = 100 => IoU = 50 / 100 = 0.5000000000 exactly!
    gt = np.zeros((50, 50), dtype=bool)
    gt[10:20, 10:20] = True  # area = 100

    pred = np.zeros((50, 50), dtype=bool)
    pred[10:20, 10:15] = True  # area = 50, all inside gt!
    # inter = 50, union = 100 + 50 - 50 = 100. IoU = 50/100 = 0.5!
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    exact_iou = inter / union
    assert exact_iou == 0.5

    # 1. In evaluate_entry_pq:
    total_iou, tp, fp, fn, _, _ = evaluate_entry_pq([gt], [pred])
    assert tp == 0, "Exact 0.5 IoU must not count as TP under strict > 0.5"
    assert fn == 1
    assert fp == 1

    # 2. In compute_instance_diagnostics:
    diag = compute_instance_diagnostics([gt], [pred], iou_thresh=0.5)
    assert diag["strict_matched_count"] == 0, "Exact 0.5 IoU must remain unmatched in diagnostics"
    assert diag["strict_unmatched_fn"] == 1
    assert diag["strict_unmatched_fp"] == 1


def test_real_evaluator_omitted_argument_resolution():
    """Verify that calling evaluate_oof with NO postprocessing overrides resolves the checkpoint's saved config."""
    import tempfile
    from evaluate import evaluate_oof
    from src.models import ResNet34UNet
    from train import save_checkpoint

    model = ResNet34UNet(in_channels=3, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    custom_postprocess = {
        "method": "connected_components",
        "high_threshold": 0.91,
        "low_threshold": 0.71,
        "min_area": 520,
        "max_instances": 11,
    }

    from src.data.folds import load_frozen_folds_manifest
    _, actual_manifest_sha = load_frozen_folds_manifest("artifacts/folds_manifest.json")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "ckpt_custom_postprocess.pt"
        save_checkpoint(
            path=ckpt_path,
            epoch=1,
            model=model,
            optimizer=optimizer,
            config={
                "model": {"name": "resnet34_unet", "backbone": "resnet34", "in_channels": 3, "pretrained": False},
                "postprocess": custom_postprocess,
                "loss": {"center_weight": 0.0, "boundary_weight": 0.0},
            },
            fold=0,
            best_metric=0.5,
            train_observations=["20110203082634Lh"],
            val_observations=["20141010195834Ch"],
            folds_manifest_sha256=actual_manifest_sha,
        )

        # Call evaluate_oof with NO postprocessing arguments passed
        # Because allow_unverified_provenance=True bypasses frozen manifest mismatch,
        # we can verify that the resolved postprocess params match custom_postprocess!
        report = evaluate_oof(
            checkpoint_path=str(ckpt_path),
            fold=0,
            limit=1,
            allow_unverified_provenance=True,
            cache_dir=str(Path(tmp_dir) / "cache"),
        )

        resolved_params = report["postprocess_params"]
        assert resolved_params["method"] == "connected_components"
        assert resolved_params["high_threshold"] == 0.91
        assert resolved_params["low_threshold"] == 0.71
        assert resolved_params["min_area"] == 520
        assert resolved_params["max_instances"] == 11


def test_canonical_partition_deduplication():
    """Verify that canonical and alias keys for the same physical observation map to one canonical entry."""
    from src.data.folds import get_deterministic_fold_partitions, canonical_observation_id

    # Simulated assignments containing alias pairs
    assignments = {
        "20170117175014Mh": 0,
        "050101-20170117175014Mh": 0,
        "20110914063134Lh": 0,
        "010101-20110914063134Lh": 0,
        "20120913165314Mh": 0,
        "20130212153254Bh": 0,
        "20121021145454Bh": 0,
    }

    parts = get_deterministic_fold_partitions(
        assignments, target_fold=0, use_migrated=False, num_tuning=2, num_confirmation=2
    )

    tuning_obs = parts["tuning"]
    conf_obs = parts["confirmation"]

    # All entries must be unique canonical observations
    assert len(tuning_obs) == len(set(tuning_obs))
    assert len(conf_obs) == len(set(conf_obs))

    # Pairwise disjoint
    assert set(tuning_obs).isdisjoint(set(conf_obs))

    # Both aliases map to canonical
    assert "20170117175014Mh" in (tuning_obs + conf_obs)
    assert "050101-20170117175014Mh" not in (tuning_obs + conf_obs)


def test_cache_fresh_equivalence(tmp_path):
    """Verify that cached maps produce identical extracted instance masks and RLEs as fresh maps."""
    from src.inference.engine import save_cached_prediction, load_cached_prediction

    shape = (256, 256)
    fg = np.zeros(shape, dtype=np.float32)
    fg[30:50, 40:180] = 0.92

    # Fresh extraction
    insts_fresh = extract_instances_from_maps(
        foreground_prob=fg,
        obs_id="cache_test",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=50,
        method="connected_components",
        shape=shape,
    )
    assert len(insts_fresh) == 1

    # Save to cache
    ckpt_hash = "abcdef0123456789"
    obs_id = "cache_test"
    save_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        fg=fg,
        image_sha256="test_img_sha",
        tile_size=512,
        stride=256,
        norm_mode="imagenet",
    )

    # Load from cache
    loaded = load_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        expected_image_sha256="test_img_sha",
        expected_tile_size=512,
        expected_stride=256,
        expected_norm_mode="imagenet",
    )
    assert loaded is not None
    loaded_fg, loaded_ctr, loaded_bnd, loaded_off = loaded

    # Cached extraction
    insts_cached = extract_instances_from_maps(
        foreground_prob=loaded_fg,
        obs_id="cache_test",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=50,
        method="connected_components",
        shape=shape,
    )
    assert len(insts_cached) == 1
    assert insts_fresh[0].rle_counts == insts_cached[0].rle_counts
    assert insts_fresh[0].bbox == insts_cached[0].bbox


def test_missing_and_incompatible_cache_rejection(tmp_path):
    """Verify missing cache returns None and incompatible metadata rejects stale caches."""
    from src.inference.engine import save_cached_prediction, load_cached_prediction

    ckpt_hash = "1234567890abcdef"
    obs_id = "test_incompat"

    # Missing cache
    assert load_cached_prediction(tmp_path, ckpt_hash, "non_existent_obs") is None

    # Populate cache
    fg = np.ones((64, 64), dtype=np.float32) * 0.8
    save_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        fg=fg,
        image_sha256="correct_hash",
        tile_size=512,
        stride=256,
        norm_mode="imagenet",
    )

    # Matching loads correctly
    assert load_cached_prediction(
        tmp_path, ckpt_hash, obs_id,
        expected_image_sha256="correct_hash", expected_tile_size=512, expected_stride=256, expected_norm_mode="imagenet"
    ) is not None

    # Mismatched image sha rejects
    assert load_cached_prediction(
        tmp_path, ckpt_hash, obs_id,
        expected_image_sha256="wrong_hash", expected_tile_size=512, expected_stride=256, expected_norm_mode="imagenet"
    ) is None

    # Mismatched tile size rejects
    assert load_cached_prediction(
        tmp_path, ckpt_hash, obs_id,
        expected_image_sha256="correct_hash", expected_tile_size=256, expected_stride=256, expected_norm_mode="imagenet"
    ) is None

    # Mismatched stride rejects
    assert load_cached_prediction(
        tmp_path, ckpt_hash, obs_id,
        expected_image_sha256="correct_hash", expected_tile_size=512, expected_stride=128, expected_norm_mode="imagenet"
    ) is None

    # Mismatched norm_mode rejects
    assert load_cached_prediction(
        tmp_path, ckpt_hash, obs_id,
        expected_image_sha256="correct_hash", expected_tile_size=512, expected_stride=256, expected_norm_mode="scale_0_1"
    ) is None


def test_cache_threshold_edge_fp32_vs_fp16(tmp_path):
    """Verify that a threshold-edge probability (0.84999 at high=0.85) produces 0 instances

    under fresh FP32 and FP32 cache, but would spuriously round up to 0.8501 under FP16 cache.
    """
    from src.inference.engine import save_cached_prediction, load_cached_prediction

    shape = (64, 64)
    fg = np.zeros(shape, dtype=np.float32)
    # Foreground block at 0.84999 - strictly below 0.85
    fg[15:45, 15:45] = np.float32(0.84999)

    # 1. Fresh extraction: strictly 0 instances
    insts_fresh = extract_instances_from_maps(
        foreground_prob=fg,
        obs_id="edge_test",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=16,
        method="connected_components",
        shape=shape,
    )
    assert len(insts_fresh) == 0, "Fresh map with values 0.84999 must not exceed 0.85 threshold"

    # 2. FP16 cache rounds 0.84999 to 0.85009765625 >= 0.85 (demonstrating defect)
    save_cached_prediction(
        cache_dir=tmp_path / "fp16",
        ckpt_hash="edge_ckpt",
        obs_id="edge_test",
        fg=fg,
        precision="float16",
    )
    loaded_fp16 = load_cached_prediction(
        cache_dir=tmp_path / "fp16",
        ckpt_hash="edge_ckpt",
        obs_id="edge_test",
        expected_precision="float16",
    )
    assert loaded_fp16 is not None
    insts_fp16 = extract_instances_from_maps(
        foreground_prob=loaded_fp16[0],
        obs_id="edge_test",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=16,
        method="connected_components",
        shape=shape,
    )
    assert len(insts_fp16) == 1, "FP16 rounding should cause spurious instance trigger at 0.85 threshold"

    # 3. FP32 cache preserves exact float value and produces 0 instances, matching fresh
    save_cached_prediction(
        cache_dir=tmp_path / "fp32",
        ckpt_hash="edge_ckpt",
        obs_id="edge_test",
        fg=fg,
        precision="float32",
    )
    loaded_fp32 = load_cached_prediction(
        cache_dir=tmp_path / "fp32",
        ckpt_hash="edge_ckpt",
        obs_id="edge_test",
        expected_precision="float32",
    )
    assert loaded_fp32 is not None
    insts_fp32 = extract_instances_from_maps(
        foreground_prob=loaded_fp32[0],
        obs_id="edge_test",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=16,
        method="connected_components",
        shape=shape,
    )
    assert len(insts_fp32) == 0, "FP32 cache must match fresh extraction and produce 0 instances"


def test_resolver_saved_unlimited_cap():
    """Checkpoint with saved postprocess max_instances: None resolves to None (unlimited), not 12."""
    ckpt_config = {
        "postprocess": {
            "method": "connected_components",
            "max_instances": None,
        }
    }
    resolved = resolve_inference_config(ckpt_config)
    assert resolved["max_instances"] is None, "Saved None cap in checkpoint must resolve to unlimited (None)"


def test_resolver_rejects_unknown_keys_in_checkpoint_postprocess_and_inference():
    """Checkpoint with unknown keys in postprocess or inference dicts must fail explicitly."""
    with pytest.raises(ValueError, match="Unknown inference configuration parameter in checkpoint postprocess"):
        resolve_inference_config({"postprocess": {"unsupported_key": 42}})

    with pytest.raises(ValueError, match="Unknown inference configuration parameter in checkpoint inference"):
        resolve_inference_config({"inference": {"bogus_option": "bad"}})


def test_strict_cache_metadata_and_range_validation(tmp_path):
    """Verify strict cache mode rejects legacy schema, wrong model_state_sha, and invalid ranges."""
    from src.inference.engine import save_cached_prediction, load_cached_prediction

    ckpt_hash = "strict_ckpt_hash"
    obs_id = "obs_strict"
    fg = np.ones((64, 64), dtype=np.float32) * 0.7

    save_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        fg=fg,
        image_sha256="img_hash_123",
        model_state_sha256="state_hash_123",
        precision="float32",
    )

    # 1. Matching strict load succeeds
    loaded = load_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        expected_image_sha256="img_hash_123",
        expected_model_state_sha256="state_hash_123",
        expected_spatial_shape=(64, 64),
        strict=True,
    )
    assert loaded is not None

    # 2. Mismatched model_state_sha fails
    assert load_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        expected_image_sha256="img_hash_123",
        expected_model_state_sha256="wrong_state_hash",
        expected_spatial_shape=(64, 64),
        strict=True,
    ) is None

    # 3. Mismatched spatial shape fails
    assert load_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id=obs_id,
        expected_image_sha256="img_hash_123",
        expected_model_state_sha256="state_hash_123",
        expected_spatial_shape=(2048, 2048),
        strict=True,
    ) is None

    # 4. Out-of-range map fails
    bad_fg = np.ones((64, 64), dtype=np.float32) * 2.5
    save_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id="obs_bad_range",
        fg=bad_fg,
        image_sha256="img_hash_bad",
        model_state_sha256="state_hash_bad",
        precision="float32",
    )
    assert load_cached_prediction(
        cache_dir=tmp_path,
        ckpt_hash=ckpt_hash,
        obs_id="obs_bad_range",
        expected_image_sha256="img_hash_bad",
        expected_model_state_sha256="state_hash_bad",
        expected_spatial_shape=(64, 64),
        strict=True,
    ) is None



