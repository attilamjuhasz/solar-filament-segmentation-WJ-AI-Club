from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

NATIVE_IMAGE_SHAPE: Tuple[int, int] = (2048, 2048)
IOU_HIT_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class ObservationMeta:
    """Metadata representing a unique physical solar observation timestamp."""
    observation_id: str
    station: str
    timestamp: str  # ISO or YYYYMMDDHHMMSS format


@dataclass(frozen=True)
class InstancePrediction:
    """A single predicted filament instance in native coordinates."""
    filament_id: str
    rle_counts: str
    score: float = 1.0
    area: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (ymin, xmin, ymax, xmax)


@dataclass
class PQStats:
    """Panoptic Quality statistics and diagnostic breakdown."""
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    total_iou: float
    entry_count: int
    one_to_many_count: int = 0
    many_to_one_count: int = 0


@dataclass
class TargetBundle:
    """Collection of multi-task supervision targets for a patch or full image."""
    foreground_mask: np.ndarray        # [H, W] uint8 (0 or 1)
    boundary_mask: np.ndarray          # [H, W] uint8 (0 or 1)
    center_heatmap: np.ndarray         # [H, W] float32 [0.0, 1.0]
    offset_field: np.ndarray           # [2, H, W] float32 normalized (dy, dx)
    valid_mask: np.ndarray             # [H, W] uint8 (0 or 1), solar disk validity
    instance_masks: List[np.ndarray] = field(default_factory=list)  # list of [H, W] binary masks


@dataclass
class PredictionFields:
    """Output continuous predictions from a dense model."""
    foreground_prob: np.ndarray        # [H, W] float32 in [0, 1]
    boundary_prob: np.ndarray          # [H, W] float32 in [0, 1]
    center_prob: np.ndarray            # [H, W] float32 in [0, 1]
    offset_vectors: np.ndarray         # [2, H, W] float32
