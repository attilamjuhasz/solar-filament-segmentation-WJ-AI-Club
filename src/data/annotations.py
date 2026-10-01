from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from pycocotools import mask as coco_mask

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.manifest import canonical_observation_id


@dataclass
class AnnotationInstance:
    """An individual filament annotation instance from ground truth with lazy mask decoding."""
    instance_id: str
    raw_segmentation: Any
    area: int
    category_id: int = 1
    spine: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    _cached_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def get_mask(self, shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE) -> np.ndarray:
        """Decode mask on the fly only when requested to avoid 30+ GB memory bloat."""
        if self._cached_mask is not None and self._cached_mask.shape == shape:
            return self._cached_mask
        mask = decode_coco_segmentation(self.raw_segmentation, shape=shape)
        return mask

    @property
    def mask(self) -> np.ndarray:
        """Convenience property decoding to native (2048, 2048) resolution."""
        return self.get_mask(NATIVE_IMAGE_SHAPE)


@dataclass
class ObservationAnnotations:
    """All annotated filament instances for one observation by a specific annotator."""
    annotator_image_id: str
    observation_id: str
    file_name: str
    instances: List[AnnotationInstance] = field(default_factory=list)

    @property
    def union_mask(self) -> np.ndarray:
        """Semantic foreground union of all instances for this observation."""
        if not self.instances:
            return np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
        union = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
        for inst in self.instances:
            union |= (inst.mask > 0).astype(np.uint8)
        return union


def decode_coco_segmentation(
    seg: Any,
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> np.ndarray:
    """Decode polygon list, uncompressed RLE, or compressed RLE into a binary mask of exact given shape.
    
    Guarantees:
    - Validates that decoded dimensions strictly match shape.
    - If RLE specifies a size that disagrees with shape (e.g. 8x8 requested at 16x16), raises ValueError.
    """
    H, W = shape
    if isinstance(seg, list):
        rles = coco_mask.frPyObjects(seg, H, W)
        mask = coco_mask.decode(rles)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
    
    elif isinstance(seg, dict):
        counts = seg.get("counts")
        seg_size = seg.get("size")
        if seg_size is not None and tuple(seg_size) != shape:
            raise ValueError(f"Segmentation mask native size {seg_size} != requested shape {shape}")

        if isinstance(counts, list):
            rles = coco_mask.frPyObjects(seg, H, W)
            mask = np.ascontiguousarray(coco_mask.decode(rles), dtype=np.uint8)
        elif isinstance(counts, (str, bytes)):
            rle_dict = {
                "size": list(shape),
                "counts": counts.encode("ascii") if isinstance(counts, str) else counts
            }
            mask = np.ascontiguousarray(coco_mask.decode(rle_dict), dtype=np.uint8)
        else:
            raise ValueError(f"Unrecognized segmentation dictionary format: {type(counts)}")
    
    elif isinstance(seg, np.ndarray):
        if seg.shape != shape:
            raise ValueError(f"Direct array shape {seg.shape} != expected {shape}")
        mask = (seg > 0).astype(np.uint8)
    
    else:
        raise ValueError(f"Unsupported segmentation data type: {type(seg)}")

    if mask.shape != shape:
        raise ValueError(f"Decoded mask shape {mask.shape} does not match requested shape {shape}")

    return mask


class AnnotationIndex:
    """Index mapping canonical observation IDs and annotator IDs to instance annotations."""

    def __init__(self) -> None:
        self.by_observation: Dict[str, List[ObservationAnnotations]] = {}
        self.by_annotator_image: Dict[str, ObservationAnnotations] = {}

    def add_entry(self, entry: ObservationAnnotations) -> None:
        self.by_annotator_image[entry.annotator_image_id] = entry
        self.by_observation.setdefault(entry.observation_id, []).append(entry)

    def get_observation_variants(self, obs_id: str) -> List[ObservationAnnotations]:
        clean_id = canonical_observation_id(obs_id)
        if clean_id not in self.by_observation:
            raise KeyError(f"Observation ID '{clean_id}' not found in annotation index")
        return self.by_observation[clean_id]

    def __len__(self) -> int:
        return len(self.by_annotator_image)


def load_coco_annotations(
    json_path: str,
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> AnnotationIndex:
    """Parse official MAGFiLO COCO-format training annotations into an indexed structure.
    
    Strict validation:
    1. Rejects non-COCO dictionaries missing 'images' or 'annotations' keys.
    2. Detects and reports orphan annotations referencing non-existent images.
    3. Uses lazy on-demand mask decoding (~50 MB footprint vs 33 GB eager materialization).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise ValueError(f"Invalid COCO annotation schema in {json_path}: missing 'images' or 'annotations' keys")

    # 1. Index images by their annotator image_id (e.g. '040301-20140609195854Bh')
    images_by_id = {img["id"]: img for img in data["images"] if "id" in img}
    if not images_by_id and data["images"]:
        raise ValueError("No valid image entries with 'id' field found in COCO dataset")

    # 2. Group raw annotations by image_id, flagging orphans
    anns_by_img_id: Dict[str, List[Dict[str, Any]]] = {}
    orphan_count = 0
    for ann in data["annotations"]:
        img_id = ann.get("image_id")
        if img_id not in images_by_id:
            orphan_count += 1
            continue
        anns_by_img_id.setdefault(img_id, []).append(ann)

    if orphan_count > 0:
        raise ValueError(f"Found {orphan_count} orphan annotations referencing images not present in 'images' list")

    index = AnnotationIndex()

    for img_id, img_info in images_by_id.items():
        fname = img_info.get("file_name", "")
        if not fname:
            raise ValueError(f"Image entry {img_id} is missing 'file_name'")
        obs_id = canonical_observation_id(fname)
        
        raw_anns = anns_by_img_id.get(img_id, [])
        instances = []

        for ann in raw_anns:
            inst_id = str(ann["id"])
            area = int(ann.get("area", 0))

            instances.append(AnnotationInstance(
                instance_id=inst_id,
                raw_segmentation=ann["segmentation"],
                area=area,
                category_id=int(ann.get("category_id", 1)),
                spine=ann.get("spine"),
                bbox=ann.get("bbox")
            ))

        entry = ObservationAnnotations(
            annotator_image_id=img_id,
            observation_id=obs_id,
            file_name=fname,
            instances=instances
        )
        index.add_entry(entry)

    return index
