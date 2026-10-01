from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List

import numpy as np

from src.contracts import NATIVE_IMAGE_SHAPE


def compute_ensemble_foreground_prob(
    fg_parent: np.ndarray,
    fg_b1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute strictly verified weighted foreground probability average: p = (1 - alpha) * parent + alpha * B1.

    Guarantees:
    - alpha is finite and strictly in (0.0, 1.0)
    - both arrays are strictly float32 numpy ndarrays (rejects float64 or other types)
    - shapes match and equal native (2048, 2048)
    - all values are finite and in [0.0, 1.0]
    - output is float32, in [0.0, 1.0]
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a float, got {type(alpha).__name__}: {alpha!r}")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not (0.0 < alpha_f < 1.0):
        raise ValueError(f"alpha must be finite and strictly between 0.0 and 1.0, got: {alpha}")

    if not isinstance(fg_parent, np.ndarray):
        raise TypeError(f"fg_parent must be a numpy ndarray, got {type(fg_parent).__name__}")
    if not isinstance(fg_b1, np.ndarray):
        raise TypeError(f"fg_b1 must be a numpy ndarray, got {type(fg_b1).__name__}")

    if fg_parent.dtype != np.float32:
        raise TypeError(f"fg_parent must have float32 dtype, got: {fg_parent.dtype}")
    if fg_b1.dtype != np.float32:
        raise TypeError(f"fg_b1 must have float32 dtype, got: {fg_b1.dtype}")

    if fg_parent.shape != NATIVE_IMAGE_SHAPE or fg_b1.shape != NATIVE_IMAGE_SHAPE:
        raise ValueError(
            f"Shape mismatch: parent {fg_parent.shape} vs b1 {fg_b1.shape}, expected {NATIVE_IMAGE_SHAPE}"
        )

    if not np.isfinite(fg_parent).all() or not np.isfinite(fg_b1).all():
        raise ValueError("Non-finite values detected in component foreground probability maps")

    if (fg_parent < 0.0).any() or (fg_parent > 1.0).any():
        raise ValueError(f"Parent map out of probability range [0, 1]: min={fg_parent.min()}, max={fg_parent.max()}")
    if (fg_b1 < 0.0).any() or (fg_b1 > 1.0).any():
        raise ValueError(f"B1 map out of probability range [0, 1]: min={fg_b1.min()}, max={fg_b1.max()}")

    # Convex combination in FP32
    w_parent = np.float32(1.0 - alpha_f)
    w_b1 = np.float32(alpha_f)
    p_ens = w_parent * fg_parent + w_b1 * fg_b1

    # Numerical guard clip
    p_ens = np.clip(p_ens, 0.0, 1.0)
    return p_ens.astype(np.float32)


def canonical_ensemble_definition_json(
    components: List[Dict[str, Any]],
    postprocess_params: Dict[str, Any],
) -> str:
    """Build canonical JSON string (sorted keys, compact separators) for ensemble definition."""
    if not isinstance(components, list) or len(components) == 0:
        raise ValueError("components must be a non-empty list")
    if not isinstance(postprocess_params, dict):
        raise ValueError("postprocess_params must be a dict")

    seen_names = set()
    clean_components = []
    for c in components:
        if not isinstance(c, dict):
            raise TypeError(f"Component must be a dict, got {type(c).__name__}")
        name = c.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Component name must be a non-empty string, got {name!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate component name detected: {name!r}")
        seen_names.add(name)

        c_sha = c.get("checkpoint_sha256")
        if not isinstance(c_sha, str) or len(c_sha) != 64:
            raise ValueError(f"Component '{name}' missing valid 64-hex checkpoint_sha256")

        m_sha = c.get("model_state_sha256")
        if not isinstance(m_sha, str) or len(m_sha) != 64:
            raise ValueError(f"Component '{name}' missing valid 64-hex model_state_sha256")

        weight = c.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"Component '{name}' weight must be numeric, got {weight!r}")
        weight_f = float(weight)
        if not math.isfinite(weight_f) or not (0.0 < weight_f < 1.0):
            raise ValueError(f"Component '{name}' weight must be finite in (0, 1), got {weight}")

        clean_components.append({
            "name": name,
            "checkpoint_sha256": c_sha.lower(),
            "model_state_sha256": m_sha.lower(),
            "weight": weight_f,
        })

    # Sort components deterministically by name
    clean_components.sort(key=lambda x: x["name"])

    # Validate postprocess_params for non-finite values
    clean_pp = {}
    for k, v in postprocess_params.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"postprocess_params key '{k}' has non-finite value: {v}")
        clean_pp[str(k)] = v

    def_dict = {
        "components": clean_components,
        "postprocess_params": clean_pp,
    }
    return json.dumps(def_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_ensemble_definition_sha256(
    components: List[Dict[str, Any]],
    postprocess_params: Dict[str, Any],
) -> str:
    """Compute SHA256 of canonical ensemble definition."""
    canonical_json = canonical_ensemble_definition_json(components, postprocess_params)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
