from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np
from scipy import ndimage
from skimage import morphology

from src.contracts import NATIVE_IMAGE_SHAPE, TargetBundle

NATIVE_NORM_SCALE: Tuple[float, float] = (2048.0, 2048.0)


def extract_instance_boundaries(
    instance_masks: Sequence[np.ndarray],
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> np.ndarray:
    """Extract boundaries independently for each instance to preserve separation cues where instances touch.
    
    B_g = Union_i (Boundary of G_i)
    """
    composite_boundary = np.zeros(shape, dtype=np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity boundary

    for mask in instance_masks:
        binary = (mask > 0).astype(bool)
        if not binary.any():
            continue
        eroded = ndimage.binary_erosion(binary, structure=struct)
        boundary = np.logical_xor(binary, eroded)
        composite_boundary |= boundary.astype(np.uint8)

    return composite_boundary


def extract_instance_skeletons(
    instance_masks: Sequence[np.ndarray],
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> np.ndarray:
    """Extract 1-pixel medial skeletons for all instances using topological thinning."""
    composite_skeleton = np.zeros(shape, dtype=np.uint8)

    for mask in instance_masks:
        binary = (mask > 0).astype(bool)
        if not binary.any():
            continue
        skel = morphology.skeletonize(binary)
        composite_skeleton |= skel.astype(np.uint8)

    return composite_skeleton


def compute_instance_anchor(
    mask: np.ndarray,
    skeleton: Optional[np.ndarray] = None
) -> Tuple[int, int]:
    """Find a robust internal anchor for a filament instance.
    
    Projects the geometric centroid onto the skeleton to guarantee that curved
    c-shaped or s-shaped filaments have an anchor that lies strictly inside the physical spine.
    """
    binary = (mask > 0).astype(bool)
    if not binary.any():
        return (0, 0)

    if skeleton is None or not skeleton.any():
        skeleton = morphology.skeletonize(binary)

    # Compute geometric centroid
    y_coords, x_coords = np.where(binary)
    cy = float(y_coords.mean())
    cx = float(x_coords.mean())

    # Get all skeleton coordinates
    skel_y, skel_x = np.where(skeleton > 0)
    if len(skel_y) == 0:
        # Fallback to closest point on mask
        skel_y, skel_x = y_coords, x_coords

    # Find skeleton pixel with minimum Euclidean distance to centroid
    dists = (skel_y - cy) ** 2 + (skel_x - cx) ** 2
    best_idx = int(np.argmin(dists))

    return int(skel_y[best_idx]), int(skel_x[best_idx])


def generate_center_heatmap_and_offsets(
    instance_masks: Sequence[np.ndarray],
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE,
    anchors: Optional[Sequence[Tuple[int, int]]] = None,
    origin: Tuple[int, int] = (0, 0),
    normalization_scale: Tuple[float, float] = NATIVE_NORM_SCALE,
    sigma: float = 6.0
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Generate Gaussian center heatmap and 2D normalized offset vectors for each instance.
    
    Guarantees:
    - Global anchor coordinates are preserved across crops and tiles.
    - Offset vectors are normalized by a fixed native reference scale (2048x2048),
      ensuring that identical physical pixels have identical offset targets in full
      views, crops, and overlapping tiles.
    - origin specifies the (y0, x0) offset of the current crop within the global frame.
    """
    H, W = shape
    center_heatmap = np.zeros((H, W), dtype=np.float32)
    offset_field = np.zeros((2, H, W), dtype=np.float32)

    y_grid, x_grid = np.indices((H, W), dtype=np.float32)
    y0, x0 = origin
    scale_y, scale_x = normalization_scale
    if scale_y <= 0 or scale_x <= 0:
        raise ValueError(f"normalization_scale must have strictly positive values, got ({scale_y}, {scale_x})")

    # Codex P2 finding #6: When origin != (0, 0), callers must supply complete global anchors
    # to prevent silent fallback to crop-local anchors which would distort global offsets.
    if (y0 != 0 or x0 != 0):
        if anchors is None:
            raise ValueError(f"Explicit global anchors are required when origin is non-zero (origin={origin}).")
        if len(anchors) < len(instance_masks):
            raise ValueError(
                f"Incomplete global anchors: expected at least {len(instance_masks)} anchors for instances, "
                f"got {len(anchors)}."
            )

    computed_anchors: List[Tuple[int, int]] = []

    for idx, mask in enumerate(instance_masks):
        binary = (mask > 0).astype(bool)
        if not binary.any():
            if anchors is not None and idx < len(anchors):
                computed_anchors.append(anchors[idx])
            else:
                computed_anchors.append((0, 0))
            continue

        # If anchors were precomputed globally, use them; otherwise compute locally in global coordinates
        if anchors is not None and idx < len(anchors):
            global_anchor_y, global_anchor_x = anchors[idx]
        else:
            local_y, local_x = compute_instance_anchor(binary)
            global_anchor_y = y0 + local_y
            global_anchor_x = x0 + local_x

        computed_anchors.append((global_anchor_y, global_anchor_x))

        # Local anchor in current crop coordinate system
        local_anchor_y = global_anchor_y - y0
        local_anchor_x = global_anchor_x - x0

        # Place Gaussian peak around local anchor
        radius = int(3 * sigma)
        y_min = max(0, local_anchor_y - radius)
        y_max = min(H, local_anchor_y + radius + 1)
        x_min = max(0, local_anchor_x - radius)
        x_max = min(W, local_anchor_x + radius + 1)

        if y_max > y_min and x_max > x_min:
            sub_y, sub_x = np.indices((y_max - y_min, x_max - x_min), dtype=np.float32)
            sub_y += y_min
            sub_x += x_min

            dist_sq = (sub_y - local_anchor_y) ** 2 + (sub_x - local_anchor_x) ** 2
            gaussian = np.exp(-dist_sq / (2.0 * sigma ** 2))

            center_heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                center_heatmap[y_min:y_max, x_min:x_max],
                gaussian
            )

        # Offset field: (global_anchor - (origin + local_coords)) / native_scale
        # dy = (global_anchor_y - (y0 + y_grid)) / scale_y
        dy = (local_anchor_y - y_grid[binary]) / float(scale_y)
        dx = (local_anchor_x - x_grid[binary]) / float(scale_x)

        offset_field[0, binary] = dy
        offset_field[1, binary] = dx

    return center_heatmap, offset_field, computed_anchors


def make_targets(
    instance_masks: Sequence[np.ndarray],
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE,
    anchors: Optional[Sequence[Tuple[int, int]]] = None,
    origin: Tuple[int, int] = (0, 0),
    normalization_scale: Tuple[float, float] = NATIVE_NORM_SCALE,
    valid_mask: Optional[np.ndarray] = None
) -> TargetBundle:
    """Generate complete multi-task supervision bundle with consistent coordinate normalization."""
    if valid_mask is None:
        valid_mask = np.ones(shape, dtype=np.uint8)

    # 1. Semantic foreground union
    foreground_mask = np.zeros(shape, dtype=np.uint8)
    for m in instance_masks:
        foreground_mask |= (m > 0).astype(np.uint8)

    # 2. Boundary mask
    boundary_mask = extract_instance_boundaries(instance_masks, shape)

    # 3. Center heatmap and continuous offsets with global anchor preservation
    center_heatmap, offset_field, _ = generate_center_heatmap_and_offsets(
        instance_masks,
        shape=shape,
        anchors=anchors,
        origin=origin,
        normalization_scale=normalization_scale
    )

    return TargetBundle(
        foreground_mask=foreground_mask,
        boundary_mask=boundary_mask,
        center_heatmap=center_heatmap,
        offset_field=offset_field,
        valid_mask=valid_mask,
        instance_masks=list(instance_masks)
    )
