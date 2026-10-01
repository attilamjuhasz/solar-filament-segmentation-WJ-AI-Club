import numpy as np
import pytest

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.targets import (
    compute_instance_anchor,
    extract_instance_boundaries,
    extract_instance_skeletons,
    generate_center_heatmap_and_offsets,
    make_targets,
)


def test_independent_instance_boundaries():
    # Two touching 50x50 squares: [100:150, 100:150] and [100:150, 150:200]
    m1 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    m1[100:150, 100:150] = 1

    m2 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    m2[100:150, 150:200] = 1

    # If we extracted boundaries independently, the boundary at x=149 and x=150 must exist!
    bnd = extract_instance_boundaries([m1, m2])
    assert bnd[120, 149] == 1  # Boundary of m1
    assert bnd[120, 150] == 1  # Boundary of m2


def test_instance_anchor_inside_curved_structure():
    # Construct a C-shaped arc
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:30] = 1  # Left spine
    mask[20:30, 20:80] = 1  # Top branch
    mask[70:80, 20:80] = 1  # Bottom branch
    # Note: geometric centroid will be around (50, 50), which is in empty space!
    cy, cx = np.where(mask)
    geom_cy, geom_cx = int(cy.mean()), int(cx.mean())
    assert mask[geom_cy, geom_cx] == 0, "Centroid must fall in empty space for this test"

    anchor_y, anchor_x = compute_instance_anchor(mask)
    # The anchor must fall INSIDE the physical mask
    assert mask[anchor_y, anchor_x] == 1, "Anchor must fall strictly inside the physical structure"


def test_target_bundle_generation():
    m1 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    m1[500:550, 500:550] = 1

    bundle = make_targets([m1])
    assert bundle.foreground_mask.shape == NATIVE_IMAGE_SHAPE
    assert bundle.boundary_mask.shape == NATIVE_IMAGE_SHAPE
    assert bundle.center_heatmap.shape == NATIVE_IMAGE_SHAPE
    assert bundle.offset_field.shape == (2, 2048, 2048)

    # Center heatmap should have a peak near (525, 525)
    assert bundle.center_heatmap[520:530, 520:530].max() > 0.9


def test_crop_anchor_and_offset_invariance():
    """Codex P1 finding #5: A foreground pixel must receive identical offset targets regardless of crop boundaries."""
    H_full, W_full = 128, 128
    mask_full = np.zeros((H_full, W_full), dtype=np.uint8)
    mask_full[60:65, 20:110] = 1

    # 1. Full view bundle with fixed normalization scale
    norm_scale = (128.0, 128.0)
    bundle_full = make_targets(
        [mask_full],
        shape=(H_full, W_full),
        normalization_scale=norm_scale
    )
    # Find full anchor
    anchor_y, anchor_x = compute_instance_anchor(mask_full)
    assert (anchor_y, anchor_x) == (62, 64)

    # Offset at pixel (62, 80)
    full_dx = bundle_full.offset_field[1, 62, 80]
    expected_dx = (64 - 80) / 128.0  # -0.125
    assert np.isclose(full_dx, expected_dx)

    # 2. Crop [:, 70:]
    crop_x1 = 70
    mask_crop = mask_full[:, crop_x1:]
    assert mask_crop.shape == (128, 58)

    # When preserving full-observation anchors and origin:
    bundle_crop = make_targets(
        [mask_crop],
        shape=(128, 58),
        anchors=[(anchor_y, anchor_x)],
        origin=(0, crop_x1),
        normalization_scale=norm_scale
    )

    # Pixel (62, 80) in full frame corresponds to (62, 10) in crop frame
    crop_dx = bundle_crop.offset_field[1, 62, 10]

    # Invariance check: offset must be identically equal!
    assert np.isclose(full_dx, crop_dx), f"Full dx {full_dx} != Crop dx {crop_dx}"


def test_crop_anchor_misuse_rejections():
    """Verify Codex P2 finding #6: Declaring a non-zero origin requires explicit complete anchors."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    # Missing anchors on crop
    with pytest.raises(ValueError, match="Explicit global anchors are required"):
        generate_center_heatmap_and_offsets([mask], shape=(64, 64), origin=(0, 20), anchors=None)

    # Incomplete anchors on crop (2 instances, only 1 anchor)
    mask2 = np.zeros((64, 64), dtype=np.uint8)
    mask2[30:40, 30:40] = 1
    with pytest.raises(ValueError, match="Incomplete global anchors"):
        generate_center_heatmap_and_offsets([mask, mask2], shape=(64, 64), origin=(0, 20), anchors=[(15, 35)])

    # Invalid normalization scale
    with pytest.raises(ValueError, match="normalization_scale must have strictly positive values"):
        generate_center_heatmap_and_offsets([mask], shape=(64, 64), origin=(0, 0), normalization_scale=(-1.0, 2048.0))

