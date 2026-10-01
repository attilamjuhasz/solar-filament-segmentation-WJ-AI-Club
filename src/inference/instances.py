from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.segmentation import watershed

from src.contracts import InstancePrediction, NATIVE_IMAGE_SHAPE
from src.inference.rle import encode_instance

# Consistent 8-connectivity structuring element to prevent diagonal spine fragmentation
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def hysteresis_threshold(
    foreground_prob: np.ndarray,
    high_threshold: float = 0.55,
    low_threshold: float = 0.30,
    connectivity: np.ndarray = CONNECTIVITY_8
) -> np.ndarray:
    """Retain low-confidence pixels only if connected to high-confidence seed regions using 8-connectivity."""
    high_mask = foreground_prob >= high_threshold
    low_mask = foreground_prob >= low_threshold

    # Label connected components in the low mask with 8-connectivity
    labeled, num_features = ndimage.label(low_mask, structure=connectivity)
    if num_features == 0:
        return np.zeros_like(high_mask, dtype=bool)

    # Find labels that overlap with high_mask
    valid_labels = np.unique(labeled[high_mask])
    # Remove background 0 if present
    valid_labels = valid_labels[valid_labels != 0]

    return np.isin(labeled, valid_labels)


from skimage.feature import peak_local_max


def extract_instances_from_maps(
    foreground_prob: np.ndarray,
    center_prob: Optional[np.ndarray] = None,
    boundary_prob: Optional[np.ndarray] = None,
    offset_field: Optional[np.ndarray] = None,
    obs_id: str = "observation",
    high_threshold: float = 0.55,
    low_threshold: float = 0.30,
    min_area: int = 16,
    method: str = "watershed",  # "watershed" or "connected_components"
    center_threshold: float = 0.35,
    boundary_weight: float = 0.50,
    marker_min_distance: int = 7,
    marker_cap_per_component: Optional[int] = None,
    max_peaks: int = 200,
    max_instances: int = 200,
    merge_compatible_offsets: bool = False,
    max_anchor_dist: float = 35.0,
    max_boundary_barrier: float = 0.40,
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> List[InstancePrediction]:
    """Extract delineated filament instances using configurable 8-connected hysteresis and watershed/CC."""
    # 1. Semantic binary support via 8-connected hysteresis
    support_mask = hysteresis_threshold(
        foreground_prob,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        connectivity=CONNECTIVITY_8
    )
    if not support_mask.any():
        return []

    if method == "connected_components":
        # Pure 8-connected components control
        segmented_labels, num_labels = ndimage.label(support_mask, structure=CONNECTIVITY_8)
    elif method == "watershed":
        # 2. Derive markers using consistent 8-connectivity
        if center_prob is not None:
            # Suppress centers outside foreground support
            masked_center = (center_prob.astype(np.float32)) * support_mask
            if marker_min_distance > 1:
                coords = peak_local_max(
                    masked_center,
                    min_distance=marker_min_distance,
                    threshold_abs=center_threshold,
                    num_peaks=max_peaks,
                )
                local_max = np.zeros_like(support_mask, dtype=bool)
                if len(coords) > 0:
                    local_max[coords[:, 0], coords[:, 1]] = True
            else:
                local_max = morphology.local_maxima(masked_center, connectivity=2)
                local_max &= (masked_center >= center_threshold)
            markers, num_markers = ndimage.label(local_max, structure=CONNECTIVITY_8)
        else:
            # Fallback to connected components with 8-connectivity
            labeled_cc, num_cc = ndimage.label(support_mask, structure=CONNECTIVITY_8)
            markers, num_markers = labeled_cc, num_cc

        # Find connected components of support mask
        labeled_components, num_comp = ndimage.label(support_mask, structure=CONNECTIVITY_8)
        next_marker = int(markers.max()) + 1

        # Guarantee that every isolated connected component has at least one marker
        component_slices = ndimage.find_objects(labeled_components)
        for comp_idx, sl in enumerate(component_slices, start=1):
            if sl is None:
                continue
            sub_comp = (labeled_components[sl] == comp_idx)
            sub_markers = markers[sl]
            comp_marker_ids = np.unique(sub_markers[sub_comp])
            comp_marker_ids = comp_marker_ids[comp_marker_ids != 0]

            if len(comp_marker_ids) == 0:
                # Place marker at peak distance inside local bounding box
                dt_sub = ndimage.distance_transform_edt(sub_comp)
                peak_yx_sub = np.unravel_index(np.argmax(dt_sub), dt_sub.shape)
                global_y = sl[0].start + peak_yx_sub[0]
                global_x = sl[1].start + peak_yx_sub[1]
                markers[global_y, global_x] = next_marker
                next_marker += 1
            elif marker_cap_per_component is not None and len(comp_marker_ids) > marker_cap_per_component:
                # Keep top K markers with highest center probability inside local component
                if center_prob is not None:
                    sub_center = center_prob[sl]
                    marker_scores = []
                    for m_id in comp_marker_ids:
                        score = float(sub_center[sub_markers == m_id].max())
                        marker_scores.append((score, m_id))
                    marker_scores.sort(key=lambda x: x[0], reverse=True)
                    drop_ids = set([m_id for _, m_id in marker_scores[marker_cap_per_component:]])
                    for m_id in drop_ids:
                        sub_markers[sub_markers == m_id] = 0

        # 3. Construct watershed energy surface
        energy = (1.0 - foreground_prob).astype(np.float32)
        if boundary_prob is not None and boundary_weight > 0.0:
            energy += boundary_weight * boundary_prob.astype(np.float32)

        # 4. Marker-controlled watershed with 8-connectivity
        segmented_labels = watershed(
            image=energy,
            markers=markers,
            connectivity=CONNECTIVITY_8,
            mask=support_mask
        )
    else:
        raise ValueError(f"Unknown instance extraction method: '{method}'")

    # 5. Extract instances and encode using fast bounding-box slices
    instances: List[InstancePrediction] = []
    component_slices = ndimage.find_objects(segmented_labels)

    candidate_instances = []
    for label_id, sl in enumerate(component_slices, start=1):
        if sl is None:
            continue
        sub_mask = (segmented_labels[sl] == label_id)
        area = int(sub_mask.sum())
        if area < min_area:
            continue

        sub_fg = foreground_prob[sl]
        score = float(sub_fg[sub_mask].mean())

        ymin = sl[0].start
        ymax = sl[0].stop - 1
        xmin = sl[1].start
        xmax = sl[1].stop - 1

        candidate_instances.append({
            "sl": sl,
            "sub_mask": sub_mask,
            "score": score,
            "area": area,
            "bbox": (ymin, xmin, ymax, xmax)
        })

    # Sort candidates by score descending and cap at max_instances
    candidate_instances.sort(key=lambda x: x["score"], reverse=True)
    if max_instances is not None:
        candidate_instances = candidate_instances[:max_instances]

    instance_counter = 1
    for c in candidate_instances:
        inst_mask = np.zeros(shape, dtype=np.uint8)
        inst_mask[c["sl"]][c["sub_mask"]] = 1

        filament_id = f"{obs_id}_{instance_counter}"
        rle = encode_instance(inst_mask, expected_shape=shape)

        instances.append(InstancePrediction(
            filament_id=filament_id,
            rle_counts=rle,
            score=c["score"],
            area=c["area"],
            bbox=c["bbox"]
        ))
        instance_counter += 1

    return instances
