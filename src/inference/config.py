from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

# Sentinel object to distinguish omitted/unset parameters from explicit None (e.g. unlimited cap)
UNSET = object()

DOCUMENTED_DEFAULTS: Dict[str, Any] = {
    "method": "connected_components",
    "high_threshold": 0.85,
    "low_threshold": 0.60,
    "center_threshold": 0.35,
    "boundary_weight": 0.50,
    "marker_min_distance": 7,
    "marker_cap_per_component": None,
    "max_peaks": 200,
    "min_area": 400,
    "max_instances": 12,
    "tile_size": 512,
    "stride": 256,
    "tile_batch_size": 16,
    "norm_mode": "imagenet",
    "precision": "float32",
}


def resolve_inference_config(
    checkpoint_config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve and validate an immutable inference configuration.

    Precedence order:
    1. DOCUMENTED_DEFAULTS
    2. Checkpoint saved postprocess / data / inference settings
    3. Caller overrides (explicit values only; UNSET sentinels ignored)

    Guarantees:
    - Never mutates checkpoint metadata or caller dictionaries.
    - Explicit zero or None (for caps) is preserved and distinct from unset.
    - Rejects invalid types, bools-as-ints, non-finite values, and invalid bounds.
    - Rejects unknown keys.
    - Rejects unsupervised learned heads when watershed is requested.
    """
    ckpt_cfg = checkpoint_config or {}
    resolved = copy.deepcopy(DOCUMENTED_DEFAULTS)

    # 1. Merge checkpoint saved configuration
    if "postprocess" in ckpt_cfg and isinstance(ckpt_cfg["postprocess"], dict):
        for k, v in ckpt_cfg["postprocess"].items():
            if k not in DOCUMENTED_DEFAULTS:
                raise ValueError(f"Unknown inference configuration parameter in checkpoint postprocess: '{k}'")
            if v is not UNSET:
                resolved[k] = v

    if "inference" in ckpt_cfg and isinstance(ckpt_cfg["inference"], dict):
        for k, v in ckpt_cfg["inference"].items():
            if k not in DOCUMENTED_DEFAULTS:
                raise ValueError(f"Unknown inference configuration parameter in checkpoint inference: '{k}'")
            if v is not UNSET:
                resolved[k] = v

    if "data" in ckpt_cfg and isinstance(ckpt_cfg["data"], dict):
        if "norm_mode" in ckpt_cfg["data"] and ckpt_cfg["data"]["norm_mode"] is not None:
            resolved["norm_mode"] = ckpt_cfg["data"]["norm_mode"]

    # 2. Merge explicit caller overrides
    if overrides is not None:
        for k, v in overrides.items():
            if k not in DOCUMENTED_DEFAULTS:
                raise ValueError(f"Unknown inference configuration parameter: '{k}'")
            if v is not UNSET:
                resolved[k] = v

    # 3. Strict validation
    # Method
    if resolved["method"] not in ("connected_components", "watershed"):
        raise ValueError(
            f"Invalid instance extraction method: '{resolved['method']}'. "
            f"Supported methods are 'connected_components' and 'watershed'."
        )

    # Float thresholds
    for key in ("high_threshold", "low_threshold", "center_threshold", "boundary_weight"):
        val = resolved[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Parameter '{key}' must be numeric, got {type(val).__name__}: {val}")
        if not math.isfinite(val):
            raise ValueError(f"Parameter '{key}' must be finite, got: {val}")

    if not (0.0 <= resolved["low_threshold"] <= resolved["high_threshold"] <= 1.0):
        raise ValueError(
            f"Require 0.0 <= low_threshold <= high_threshold <= 1.0, got "
            f"low_threshold={resolved['low_threshold']}, high_threshold={resolved['high_threshold']}"
        )

    if not (0.0 <= resolved["center_threshold"] <= 1.0):
        raise ValueError(
            f"Require 0.0 <= center_threshold <= 1.0, got {resolved['center_threshold']}"
        )

    if resolved["boundary_weight"] < 0.0:
        raise ValueError(
            f"Require boundary_weight >= 0.0, got {resolved['boundary_weight']}"
        )

    # Integer area and sizes
    for int_key in ("min_area", "tile_size", "stride", "max_peaks", "marker_min_distance", "tile_batch_size"):
        val = resolved[int_key]
        if isinstance(val, bool) or not isinstance(val, int):
            raise TypeError(f"Parameter '{int_key}' must be an integer (rejecting bool), got {type(val).__name__}: {val}")

    if resolved["min_area"] < 0:
        raise ValueError(f"Require min_area >= 0, got {resolved['min_area']}")

    if resolved["tile_size"] <= 0:
        raise ValueError(f"Require tile_size > 0, got {resolved['tile_size']}")

    if resolved["stride"] <= 0:
        raise ValueError(f"Require stride > 0, got {resolved['stride']}")

    if resolved["tile_batch_size"] <= 0:
        raise ValueError(f"Require positive tile_batch_size (> 0), got {resolved['tile_batch_size']}")

    if resolved["stride"] > resolved["tile_size"]:
        raise ValueError(
            f"Require stride <= tile_size to guarantee complete coverage, got "
            f"stride={resolved['stride']} > tile_size={resolved['tile_size']}"
        )

    if resolved["max_peaks"] <= 0:
        raise ValueError(f"Require max_peaks > 0, got {resolved['max_peaks']}")

    if resolved["marker_min_distance"] < 1:
        raise ValueError(f"Require marker_min_distance >= 1, got {resolved['marker_min_distance']}")

    # Caps (can be None for unlimited)
    for cap_key in ("max_instances", "marker_cap_per_component"):
        val = resolved[cap_key]
        if val is not None:
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"Parameter '{cap_key}' must be an integer or None (unlimited), got {type(val).__name__}: {val}")
            if val <= 0:
                raise ValueError(f"Parameter '{cap_key}' must be positive when specified, got {val}")

    # Norm mode and precision
    if resolved["norm_mode"] not in ("imagenet", "scale_0_1"):
        raise ValueError(f"Invalid norm_mode: '{resolved['norm_mode']}'. Supported: 'imagenet', 'scale_0_1'")

    if resolved["precision"] not in ("float32", "float16"):
        raise ValueError(f"Invalid precision: '{resolved['precision']}'. Supported: 'float32', 'float16'")

    # 4. Supervision consistency checks: reject learned auxiliary heads if un-supervised
    if resolved["method"] == "watershed":
        loss_cfg = ckpt_cfg.get("loss", {})
        ctr_w = float(loss_cfg.get("center_weight", 0.0))
        bnd_w = float(loss_cfg.get("boundary_weight", 0.0))
        if ctr_w <= 0.0:
            raise ValueError(
                "Method 'watershed' requests learned center markers, but checkpoint "
                "training supervision has center_weight=0.0."
            )
        if resolved["boundary_weight"] > 0.0 and bnd_w <= 0.0:
            raise ValueError(
                f"Method 'watershed' requests boundary_weight={resolved['boundary_weight']} > 0, "
                "but checkpoint training supervision has boundary_weight=0.0."
            )

    return resolved
