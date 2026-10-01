import numpy as np
import pytest

from src.inference.tiling import (
    get_axis_tile_starts,
    get_tile_grid,
    extract_tiles,
    blend_tiles,
)


def test_tile_grid_coverage():
    starts = get_axis_tile_starts(length=2048, tile_size=768, stride=512)
    assert starts == [0, 512, 1024, 1280]

    grid = get_tile_grid(canvas_spatial_shape=(2048, 2048), tile_size=768, stride=512)
    assert len(grid) == 16  # 4 x 4 grid

    # Verify every pixel in 2048x2048 is covered by at least one tile
    coverage = np.zeros((2048, 2048), dtype=np.uint8)
    for y1, x1, y2, x2 in grid:
        coverage[y1:y2, x1:x2] += 1
    assert coverage.min() >= 1


def test_tile_reconstruction_fidelity_2d():
    # Construct a continuous 2D gradient field on native 2048x2048
    y = np.linspace(0, 1, 2048, dtype=np.float32)
    x = np.linspace(0, 1, 2048, dtype=np.float32)
    canvas = np.outer(y, x)

    # Extract tiles
    tiles = extract_tiles(canvas, tile_size=768, stride=512)
    assert len(tiles) == 16

    # Reconstruct with Hann blending
    reconstructed = blend_tiles(tiles, canvas_spatial_shape=(2048, 2048), window_floor=0.01)

    # Reconstructed canvas should match the original within float32 tolerance
    max_diff = np.max(np.abs(canvas - reconstructed))
    assert max_diff < 1e-4, f"Max difference {max_diff} exceeds tolerance"


def test_2channel_offset_tiling_chw():
    """Codex P1 finding #4: Multi-channel [2, H, W] offset field tiling and blending."""
    canvas = np.zeros((2, 32, 32), dtype=np.float32)
    # Channel 0: Y offsets, Channel 1: X offsets
    canvas[0, :, :] = np.linspace(-1, 1, 32, dtype=np.float32)[:, None]
    canvas[1, :, :] = np.linspace(-1, 1, 32, dtype=np.float32)[None, :]

    tiles = extract_tiles(canvas, tile_size=16, stride=8, layout="CHW")
    # All tile bounding boxes must be within [0, 32]
    for (y1, x1, y2, x2), tile in tiles:
        assert 0 <= y1 < y2 <= 32
        assert 0 <= x1 < x2 <= 32
        assert tile.shape == (2, 16, 16)

    reconstructed = blend_tiles(tiles, canvas_spatial_shape=(32, 32), layout="CHW")
    assert reconstructed.shape == (2, 32, 32)
    max_diff = np.max(np.abs(canvas - reconstructed))
    assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance"


def test_hwc_tiling_fidelity():
    """Verify [H, W, C] layout tiling and reconstruction."""
    canvas = np.random.rand(32, 32, 3).astype(np.float32)
    tiles = extract_tiles(canvas, tile_size=16, stride=8, layout="HWC")
    for (y1, x1, y2, x2), tile in tiles:
        assert tile.shape == (16, 16, 3)

    reconstructed = blend_tiles(tiles, canvas_spatial_shape=(32, 32), layout="HWC")
    assert reconstructed.shape == (32, 32, 3)
    max_diff = np.max(np.abs(canvas - reconstructed))
    assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance"


def test_small_or_invalid_inputs_rejection():
    """Verify that tile_size > length or negative parameters raise ValueError."""
    with pytest.raises(ValueError, match="cannot exceed length"):
        get_axis_tile_starts(length=10, tile_size=16, stride=8)

    with pytest.raises(ValueError, match="must all be positive"):
        get_axis_tile_starts(length=0, tile_size=16, stride=8)


def test_blend_tiles_uncovered_rejection():
    """Verify Codex P2 finding #5: missing coverage raises ValueError across 2D, CHW, and HWC."""
    # Only supply top-left tile on an 8x8 canvas (tile is 4x4)
    tl_tile = np.ones((4, 4), dtype=np.float32)
    tiles_2d = [((0, 0, 4, 4), tl_tile)]
    with pytest.raises(ValueError, match="do not completely cover canvas spatial dimensions"):
        blend_tiles(tiles_2d, canvas_spatial_shape=(8, 8))

    # CHW missing coverage
    tl_chw = np.ones((2, 4, 4), dtype=np.float32)
    tiles_chw = [((0, 0, 4, 4), tl_chw)]
    with pytest.raises(ValueError, match="do not completely cover canvas spatial dimensions"):
        blend_tiles(tiles_chw, canvas_spatial_shape=(8, 8), layout="CHW")

    # HWC missing coverage
    tl_hwc = np.ones((4, 4, 2), dtype=np.float32)
    tiles_hwc = [((0, 0, 4, 4), tl_hwc)]
    with pytest.raises(ValueError, match="do not completely cover canvas spatial dimensions"):
        blend_tiles(tiles_hwc, canvas_spatial_shape=(8, 8), layout="HWC")


def test_blend_tiles_out_of_bounds_and_inconsistent_tiles():
    """Verify tile bounds and shape mismatch checks."""
    tile = np.ones((4, 4), dtype=np.float32)
    # Out of bounds: square 4x4 box placed at (6, 6, 10, 10) on an (8, 8) canvas
    with pytest.raises(ValueError, match="out of canvas bounds"):
        blend_tiles([((6, 6, 10, 10), tile)], canvas_spatial_shape=(8, 8))

    # Inconsistent non-square tile box
    bad_tile = np.ones((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected square tiles"):
        blend_tiles([((0, 0, 4, 5), bad_tile)], canvas_spatial_shape=(8, 8))

