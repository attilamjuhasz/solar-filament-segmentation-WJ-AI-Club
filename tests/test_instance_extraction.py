import numpy as np
import pytest

from src.contracts import NATIVE_IMAGE_SHAPE
from src.inference.instances import (
    extract_instances_from_maps,
    hysteresis_threshold,
)
from src.inference.rle import decode_instance


def test_hysteresis_thresholding():
    prob = np.zeros((100, 100), dtype=np.float32)
    # Seed region (high confidence)
    prob[40:50, 40:50] = 0.8
    # Connected tail (medium confidence, should be retained)
    prob[50:60, 40:50] = 0.4
    # Isolated noise (medium confidence, not connected to seed -> should be rejected)
    prob[10:20, 10:20] = 0.4

    mask = hysteresis_threshold(prob, high_threshold=0.6, low_threshold=0.35)
    assert mask[45, 45] == True
    assert mask[55, 45] == True
    assert mask[15, 15] == False


def test_instance_watershed_separation():
    # Create two nearby filaments that touch lightly
    fg_prob = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)
    fg_prob[500:600, 500:550] = 0.9  # Filament 1
    fg_prob[500:600, 560:610] = 0.9  # Filament 2
    fg_prob[540:560, 550:560] = 0.4  # Faint bridge between them

    # Center heatmap with 2 distinct peaks
    center_prob = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)
    center_prob[550, 525] = 1.0  # Center 1
    center_prob[550, 585] = 1.0  # Center 2

    # Boundary prob ridge along the bridge
    boundary_prob = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)
    boundary_prob[500:600, 555] = 0.8

    instances = extract_instances_from_maps(
        foreground_prob=fg_prob,
        center_prob=center_prob,
        boundary_prob=boundary_prob,
        obs_id="20150125172714Mh",
        min_area=50
    )

    # Must be separated into 2 distinct instances
    assert len(instances) == 2
    assert instances[0].filament_id == "20150125172714Mh_1"
    assert instances[1].filament_id == "20150125172714Mh_2"

    # Decode and check that they don't overlap
    m1 = decode_instance(instances[0].rle_counts)
    m2 = decode_instance(instances[1].rle_counts)
    assert np.logical_and(m1 > 0, m2 > 0).sum() == 0


def test_small_artifact_filtering():
    fg_prob = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)
    # Tiny 5-pixel noise blob
    fg_prob[100, 100:105] = 0.9
    # Large 200-pixel valid filament
    fg_prob[300:320, 300:310] = 0.9

    instances = extract_instances_from_maps(
        foreground_prob=fg_prob,
        min_area=16,
        obs_id="20150125172714Mh"
    )

    assert len(instances) == 1
    assert instances[0].area == 200


def test_diagonal_spine_connectivity_preservation():
    """Codex P1 finding #6: 8-connectivity must preserve thin diagonal spines without fragmenting them."""
    fg_prob = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.float32)
    # 64-pixel diagonal spine
    for i in range(64):
        fg_prob[500 + i, 500 + i] = 0.9

    instances = extract_instances_from_maps(
        foreground_prob=fg_prob,
        min_area=16,
        obs_id="20150125172714Mh"
    )

    # Must extract exactly 1 unified instance with area 64
    assert len(instances) == 1
    assert instances[0].area == 64
    assert instances[0].filament_id == "20150125172714Mh_1"

    # Verify decoded mask matches the diagonal spine
    decoded = decode_instance(instances[0].rle_counts)
    assert decoded.sum() == 64
    for i in range(64):
        assert decoded[500 + i, 500 + i] == 1

