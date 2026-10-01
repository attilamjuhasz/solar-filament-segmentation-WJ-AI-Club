from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np


def get_axis_tile_starts(length: int = 2048, tile_size: int = 768, stride: int = 512) -> List[int]:
    """Calculate 1D start coordinates for overlapping tile coverage.
    
    Guarantees:
    - length, tile_size, stride must be positive integers.
    - All starts are non-negative and satisfy start + tile_size <= length.
    - The final tile aligns flush with the end boundary (starts[-1] == length - tile_size).
    - Every position 0 <= p < length is covered by at least one tile.
    """
    if length <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError(f"length ({length}), tile_size ({tile_size}), and stride ({stride}) must all be positive")
    
    if tile_size > length:
        raise ValueError(f"tile_size ({tile_size}) cannot exceed length ({length}) without padding")

    starts: List[int] = []
    pos = 0
    while pos + tile_size <= length:
        starts.append(pos)
        pos += stride

    final_start = length - tile_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    # Verification: every pixel from 0 to length - 1 must be covered
    covered = np.zeros(length, dtype=bool)
    for s in starts:
        covered[s : s + tile_size] = True
    if not covered.all():
        raise RuntimeError(f"Tile configuration leaves uncovered pixels for length {length}, tile {tile_size}, stride {stride}")

    return starts


def get_tile_grid(
    canvas_spatial_shape: Tuple[int, int] = (2048, 2048),
    tile_size: int = 768,
    stride: int = 512
) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes (ymin, xmin, ymax, xmax) for all tiles across the canvas."""
    H, W = canvas_spatial_shape
    y_starts = get_axis_tile_starts(H, tile_size, stride)
    x_starts = get_axis_tile_starts(W, tile_size, stride)

    grid: List[Tuple[int, int, int, int]] = []
    for y in y_starts:
        for x in x_starts:
            grid.append((y, x, y + tile_size, x + tile_size))
    return grid


def create_2d_hann_window(tile_size: int, floor: float = 0.05) -> np.ndarray:
    """Create a 2D Hann (raised cosine) weighting window to suppress edge artifacts."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    w1d = np.hanning(tile_size)
    w2d = np.outer(w1d, w1d)
    w2d = np.maximum(w2d, floor)
    return w2d.astype(np.float32)


def extract_tiles(
    canvas: np.ndarray,
    tile_size: int = 768,
    stride: int = 512,
    layout: Optional[str] = None
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """Extract overlapping tiles from canvas supporting HW, CHW, or HWC explicit layouts.
    
    Returns list of ((ymin, xmin, ymax, xmax), tile_array).
    """
    ndim = canvas.ndim
    if ndim == 2:
        H, W = canvas.shape
        grid = get_tile_grid((H, W), tile_size, stride)
        return [((y1, x1, y2, x2), canvas[y1:y2, x1:x2]) for y1, x1, y2, x2 in grid]

    elif ndim == 3:
        # Determine layout
        if layout == "CHW" or (layout is None and canvas.shape[0] in (1, 2, 3, 4) and canvas.shape[2] > 4):
            C, H, W = canvas.shape
            grid = get_tile_grid((H, W), tile_size, stride)
            return [((y1, x1, y2, x2), canvas[:, y1:y2, x1:x2]) for y1, x1, y2, x2 in grid]
        elif layout == "HWC" or (layout is None and canvas.shape[2] in (1, 2, 3, 4) and canvas.shape[0] > 4):
            H, W, C = canvas.shape
            grid = get_tile_grid((H, W), tile_size, stride)
            return [((y1, x1, y2, x2), canvas[y1:y2, x1:x2, :]) for y1, x1, y2, x2 in grid]
        else:
            raise ValueError(f"Ambiguous 3D canvas shape {canvas.shape}. Please specify layout='CHW' or layout='HWC'.")

    else:
        raise ValueError(f"Unsupported canvas dimensionality: ndim={ndim}, shape={canvas.shape}")


def blend_tiles(
    tiles: Sequence[Tuple[Tuple[int, int, int, int], np.ndarray]],
    canvas_spatial_shape: Tuple[int, int] = (2048, 2048),
    window_floor: float = 0.05,
    layout: Optional[str] = None
) -> np.ndarray:
    """Blend overlapping probability tiles using smooth 2D Hann window weighting.
    
    Supports 2D (H, W), 3D CHW (C, H, W), and 3D HWC (H, W, C).
    """
    if not tiles:
        raise ValueError("Cannot blend empty sequence of tiles")

    H, W = canvas_spatial_shape
    sample_box, sample_tile = tiles[0]
    tile_h, tile_w = sample_box[2] - sample_box[0], sample_box[3] - sample_box[1]
    if tile_h != tile_w:
        raise ValueError(f"Expected square tiles, got ({tile_h}, {tile_w})")
    
    hann_w = create_2d_hann_window(tile_h, floor=window_floor)

    # Validate all tiles for bounds and shape consistency
    is_chw = (sample_tile.ndim == 3) and ((layout == "CHW") or (layout is None and sample_tile.shape[0] in (1, 2, 3, 4) and sample_tile.shape[1] == tile_h))
    is_hwc = (sample_tile.ndim == 3) and ((layout == "HWC") or (layout is None and sample_tile.shape[2] in (1, 2, 3, 4) and sample_tile.shape[0] == tile_h))

    for i, ((y1, x1, y2, x2), tile) in enumerate(tiles):
        if not (0 <= y1 < y2 <= H and 0 <= x1 < x2 <= W):
            raise ValueError(f"Tile {i} bounding box ({y1}, {x1}, {y2}, {x2}) is out of canvas bounds (0, 0, {H}, {W})")
        box_h, box_w = y2 - y1, x2 - x1
        if box_h != tile_h or box_w != tile_w:
            raise ValueError(f"Tile {i} box dimensions ({box_h}, {box_w}) do not match expected tile size ({tile_h}, {tile_w})")
        if sample_tile.ndim == 2 and tile.shape != (tile_h, tile_w):
            raise ValueError(f"Tile {i} array shape {tile.shape} does not match expected ({tile_h}, {tile_w})")
        elif sample_tile.ndim == 3:
            if is_chw and tile.shape != (sample_tile.shape[0], tile_h, tile_w):
                raise ValueError(f"Tile {i} array shape {tile.shape} does not match expected ({sample_tile.shape[0]}, {tile_h}, {tile_w})")
            elif is_hwc and tile.shape != (tile_h, tile_w, sample_tile.shape[2]):
                raise ValueError(f"Tile {i} array shape {tile.shape} does not match expected ({tile_h}, {tile_w}, {sample_tile.shape[2]})")

    if sample_tile.ndim == 2:
        accum_map = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        for (y1, x1, y2, x2), tile in tiles:
            accum_map[y1:y2, x1:x2] += tile.astype(np.float32) * hann_w
            weight_sum[y1:y2, x1:x2] += hann_w

        uncovered = (weight_sum < 1e-5)
        if uncovered.any():
            raise ValueError(f"Tiles do not completely cover canvas spatial dimensions ({H}, {W}). Found {int(uncovered.sum())} uncovered pixels.")

        return accum_map / np.maximum(weight_sum, 1e-6)

    elif sample_tile.ndim == 3:
        # Determine layout
        is_chw = (layout == "CHW") or (layout is None and sample_tile.shape[0] in (1, 2, 3, 4) and sample_tile.shape[1] == tile_h)
        is_hwc = (layout == "HWC") or (layout is None and sample_tile.shape[2] in (1, 2, 3, 4) and sample_tile.shape[0] == tile_h)

        if is_chw:
            C = sample_tile.shape[0]
            accum_map = np.zeros((C, H, W), dtype=np.float32)
            weight_sum = np.zeros((1, H, W), dtype=np.float32)
            hann_w_exp = hann_w[None, :, :]

            for (y1, x1, y2, x2), tile in tiles:
                accum_map[:, y1:y2, x1:x2] += tile.astype(np.float32) * hann_w_exp
                weight_sum[:, y1:y2, x1:x2] += hann_w_exp

            uncovered = (weight_sum[0] < 1e-5)
            if uncovered.any():
                raise ValueError(f"Tiles do not completely cover canvas spatial dimensions ({H}, {W}). Found {int(uncovered.sum())} uncovered pixels.")

            return accum_map / np.maximum(weight_sum, 1e-6)

        elif is_hwc:
            C = sample_tile.shape[2]
            accum_map = np.zeros((H, W, C), dtype=np.float32)
            weight_sum = np.zeros((H, W, 1), dtype=np.float32)
            hann_w_exp = hann_w[:, :, None]

            for (y1, x1, y2, x2), tile in tiles:
                accum_map[y1:y2, x1:x2, :] += tile.astype(np.float32) * hann_w_exp
                weight_sum[y1:y2, x1:x2, :] += hann_w_exp

            uncovered = (weight_sum[:, :, 0] < 1e-5)
            if uncovered.any():
                raise ValueError(f"Tiles do not completely cover canvas spatial dimensions ({H}, {W}). Found {int(uncovered.sum())} uncovered pixels.")

            return accum_map / np.maximum(weight_sum, 1e-6)

        else:
            raise ValueError(f"Unsupported 3D tile shape {sample_tile.shape}. Specify layout='CHW' or layout='HWC'.")

    else:
        raise ValueError(f"Unsupported tile ndim {sample_tile.ndim}")
