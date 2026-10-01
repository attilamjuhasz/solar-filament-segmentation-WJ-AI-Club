from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from pycocotools import mask as coco_mask

from src.contracts import NATIVE_IMAGE_SHAPE


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA256 checksum of a file to cryptographically bind manifest to CSV."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_compressed_rle_runs(counts_str: str | bytes) -> List[int]:
    """Parse variable-length LEB integer run lengths directly from COCO compressed RLE string.
    
    Reconstructs the alternating [zeros, ones, zeros, ones, ...] run counts.
    Strictly rejects unexpected characters, quotes, whitespace, or negative/truncated runs.
    """
    if isinstance(counts_str, bytes):
        try:
            s = counts_str.decode("ascii")
        except UnicodeDecodeError as e:
            raise ValueError(f"COCO compressed RLE contains non-ASCII bytes: {e}")
    elif isinstance(counts_str, str):
        s = counts_str
    else:
        raise TypeError(f"Expected str or bytes for compressed RLE, got {type(counts_str)}")

    if not s:
        raise ValueError("Compressed RLE string cannot be empty")

    # Reject quotes, commas, newlines, tabs, and spaces
    if any(ch in s for ch in "\"'\r\n, \t"):
        raise ValueError("Forbidden character (quotes, whitespace, newline, or comma) in compressed RLE")

    runs: List[int] = []
    p = 0
    m = 0
    str_len = len(s)

    while p < str_len:
        x = 0
        k = 0
        more = 1
        while more:
            if p >= str_len:
                raise ValueError("Unexpected end of compressed RLE string (truncated variable-length integer)")
            ch_code = ord(s[p]) - 48
            if ch_code < 0 or ch_code > 63:
                raise ValueError(
                    f"Invalid character '{s[p]}' (ASCII {ord(s[p])}) in COCO compressed RLE: "
                    f"valid alphabet is ASCII 48 ('0') to 111 ('o')"
                )
            if k >= 6:
                raise ValueError(f"Variable-length integer exceeds 32 bits in compressed RLE at index {p}")
            x |= (ch_code & 0x1F) << (5 * k)
            more = ch_code & 0x20
            p += 1
            k += 1
            if not more and (ch_code & 0x10):
                x |= (-1 << (5 * k))

        if m > 2:
            x += runs[m - 2]
        
        if x < 0:
            raise ValueError(f"Decoded negative run length {x} at position {m}")
        runs.append(x)
        m += 1

    return runs


def validate_compressed_rle(
    counts_str: str | bytes,
    expected_shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> Dict[str, Any]:
    """Parse and mathematically validate compressed RLE run lengths BEFORE decoding.
    
    Guarantees:
    1. Sum of all run lengths MUST exactly match expected total pixels (H * W).
       For 2048x2048, total pixels must be 4,194,304. An encoded 512x512 mask (262,144) is rejected!
    2. Runs must be non-negative and non-empty.
    3. Foreground area (odd runs) must be > 0 and < total_pixels.
    """
    runs = parse_compressed_rle_runs(counts_str)
    total_pixels = sum(runs)
    expected_pixels = expected_shape[0] * expected_shape[1]

    if total_pixels != expected_pixels:
        raise ValueError(
            f"RLE total pixel count {total_pixels} does not match expected shape {expected_shape} "
            f"({expected_pixels} pixels). Detected dimension mismatch (e.g. 512x512 mask encoded as 2048x2048)."
        )

    # Foreground pixels are at odd indices (runs[1], runs[3], ...)
    fg_area = sum(runs[1::2])
    if fg_area == 0:
        raise ValueError("Decoded RLE has zero foreground area (empty mask)")
    if fg_area >= total_pixels:
        raise ValueError(f"Decoded RLE is completely full foreground ({fg_area} == {total_pixels})")

    return {
        "total_pixels": total_pixels,
        "foreground_area": fg_area,
        "run_count": len(runs),
        "is_valid": True,
    }


def encode_instance(
    mask: np.ndarray,
    expected_shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE
) -> str:
    """Encode a binary 2D numpy mask to unquoted COCO compressed RLE string.
    
    Strict guarantees:
    1. Shape must exactly equal expected_shape (default: (2048, 2048)).
    2. Must be strictly binary (values in {0, 1}).
    3. Must have non-zero area (empty masks are not valid instances).
    4. Encoded counts must not contain quotes, newlines, or commas.
    5. Performs an immediate run-length and decode check to ensure lossless round-trip.
    """
    value = np.asarray(mask)

    if value.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, but got {value.shape}")

    if not np.logical_or(value == 0, value == 1).all():
        raise ValueError("Mask must contain only binary values (0 or 1)")

    if not value.any():
        raise ValueError("An empty mask cannot be encoded as an individual filament instance")

    binary = np.asfortranarray(value, dtype=np.uint8)
    encoded = coco_mask.encode(binary)
    counts = encoded["counts"].decode("ascii")

    # Validate parsed run lengths against expected shape
    validate_compressed_rle(counts, expected_shape)

    decoded = coco_mask.decode({
        "size": list(expected_shape),
        "counts": counts.encode("ascii"),
    })

    if not np.array_equal(decoded, binary):
        raise ValueError("RLE round-trip mismatch: decoded mask does not equal source binary mask")

    return counts


def decode_instance(
    rle_counts: str,
    shape: Tuple[int, int] = NATIVE_IMAGE_SHAPE,
    strict_validation: bool = True
) -> np.ndarray:
    """Decode a compressed COCO RLE string back into a uint8 binary array.
    
    Pre-validates that run lengths sum to shape[0] * shape[1] before calling pycocotools.
    """
    if not isinstance(rle_counts, str) or not rle_counts:
        raise ValueError("rle_counts must be a non-empty ASCII string")
    
    if strict_validation:
        validate_compressed_rle(rle_counts, shape)

    decoded = coco_mask.decode({
        "size": list(shape),
        "counts": rle_counts.encode("ascii")
    })
    return np.ascontiguousarray(decoded, dtype=np.uint8)


def write_submission_csv(
    output_path: str,
    rows: Sequence[Tuple[str, str]]
) -> int:
    """Write submission.csv with strict QUOTE_NONE and LF line terminators.
    
    Rows is a sequence of (filament_id, segmentation_rle).
    Returns number of instance rows written.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(
            handle,
            quoting=csv.QUOTE_NONE,
            escapechar=None,
            lineterminator="\n",
        )
        writer.writerow(["filament_id", "segmentation_rle"])
        for fid, rle in rows:
            # Pre-validate that no row contains quotes
            if '"' in fid or "'" in fid or '"' in rle or "'" in rle:
                raise ValueError(f"Row for {fid} contains quotes, which is forbidden in strict submission")
            writer.writerow([fid, rle])
    return len(rows)


def validate_submission_csv(
    filepath: str,
    expected_observation_ids: Optional[Set[str]] = None,
    verify_rle_decoding: bool = True
) -> Dict[str, Any]:
    """Strictly audit a submission CSV against Kaggle competition requirements.
    
    Checks:
    - Raw unquoted format (no quotes anywhere on any line)
    - Header is exactly 'filament_id,segmentation_rle'
    - No NaN/null values or empty rows
    - Unique filament_ids
    - Filament ID format matches <canonical_obs_id>_<idx>
    - If expected_observation_ids is provided (even if empty set), enforces that all IDs belong to it
    - Run-lengths validated BEFORE decoding: sum must be exactly 2048^2 (4,194,304)
    - Decoded mask area > 0
    """
    seen_fids: Set[str] = set()
    seen_obs: Set[str] = set()
    total_rows = 0
    total_area = 0

    with open(filepath, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if not lines:
        raise ValueError(f"Submission file {filepath} is empty")

    header = lines[0].rstrip("\r\n")
    if header != "filament_id,segmentation_rle":
        raise ValueError(f"Invalid header '{header}'. Expected exactly 'filament_id,segmentation_rle'")

    for line_idx, raw_line in enumerate(lines[1:], start=2):
        line = raw_line.rstrip("\r\n")
        if not line:
            raise ValueError(f"Line {line_idx}: empty line in CSV")

        # Strict check: CSV must not contain quote characters
        if '"' in line or "'" in line:
            raise ValueError(f"Line {line_idx} contains quote characters, violating unquoted ASCII CSV requirement")

        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Line {line_idx} has {len(parts)} columns, expected exactly 2: {line}")

        fid, rle = parts[0], parts[1]
        if not fid:
            raise ValueError(f"Line {line_idx}: empty filament_id")
        if not rle:
            raise ValueError(f"Line {line_idx}: empty segmentation_rle")

        if fid in seen_fids:
            raise ValueError(f"Line {line_idx}: duplicate filament_id '{fid}'")
        seen_fids.add(fid)

        # Check format: e.g. 20110120105534Ch_1
        m = re.match(r"^(\d{14}[A-Za-z]{2})_(\d+)$", fid)
        if not m:
            raise ValueError(f"Line {line_idx}: filament_id '{fid}' does not match pattern <obs_id>_<idx>")

        obs_id = m.group(1)
        seen_obs.add(obs_id)

        # Strict check: expected_observation_ids constraint
        if expected_observation_ids is not None:
            if obs_id not in expected_observation_ids:
                raise ValueError(f"Line {line_idx}: observation '{obs_id}' not in expected test set")

        # Validate run length bounds (must sum to 2048^2)
        rle_info = validate_compressed_rle(rle, NATIVE_IMAGE_SHAPE)
        total_area += rle_info["foreground_area"]

        if verify_rle_decoding:
            mask = decode_instance(rle, NATIVE_IMAGE_SHAPE, strict_validation=False)
            area = int(mask.sum())
            if area == 0:
                raise ValueError(f"Line {line_idx}: filament '{fid}' decoded to 0 pixels area")
            if area != rle_info["foreground_area"]:
                raise ValueError(
                    f"Line {line_idx}: RLE run-length foreground area ({rle_info['foreground_area']}) "
                    f"does not match decoded mask area ({area})"
                )

        total_rows += 1

    # Codex finding #5: A header-only CSV is structurally valid representing legitimate
    # zero-detection / all-abstained inference. Completion across expected observations
    # is audited via audit_submission_and_manifest.
    return {
        "total_instances": total_rows,
        "unique_observations": len(seen_obs),
        "total_pixel_area": total_area,
        "is_valid": True,
    }


def audit_submission_and_manifest(
    csv_path: str | Path,
    manifest_path: str | Path,
    expected_observation_ids: Optional[Set[str]] = None,
    verify_rle_decoding: bool = True
) -> Dict[str, Any]:
    """Strictly audit a submission CSV together with its observation-completion manifest.
    
    Checks:
    1. CSV structural and RLE validity via validate_submission_csv.
    2. Manifest schema and cryptographic SHA256 binding to the CSV (preventing stale manifest reuse).
    3. Terminal observation statuses: strictly 'processed' (if instances > 0) or 'abstained' (if instances == 0).
       Any 'failed' or unknown status raises ValueError.
    4. Exact instance count reconciliation:
       - Each observation's instance count in manifest must exactly match the number of rows in CSV.
       - The sum of instance counts across manifest observations must equal total_instances and CSV rows.
       - Instance counts must be finite non-negative integers.
    5. Observation set completeness:
       - Manifest observation IDs must be unique.
       - Manifest observations must exactly equal expected_observation_ids (including when expected set is empty).
    """
    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # 1. Audit CSV syntax and RLE encoding
    csv_report = validate_submission_csv(
        str(csv_path),
        expected_observation_ids=expected_observation_ids,
        verify_rle_decoding=verify_rle_decoding
    )

    # Count rows per observation in CSV
    csv_obs_counts: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        fid = line.split(",")[0]
        obs = fid.rsplit("_", 1)[0]
        csv_obs_counts[obs] = csv_obs_counts.get(obs, 0) + 1

    # 2. Parse and validate manifest schema
    with open(manifest_path, "r", encoding="utf-8") as f:
        try:
            manifest_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse manifest JSON: {e}")

    if not isinstance(manifest_data, dict):
        raise ValueError("Manifest root must be a JSON object")

    # Cryptographic binding check: csv_sha256 must be present, valid 64-hex lowercase, and match
    csv_sha256 = compute_file_sha256(csv_path)
    manifest_csv_sha256 = manifest_data.get("csv_sha256")
    if manifest_csv_sha256 is None:
        raise ValueError("Manifest missing required 'csv_sha256' field")
    if not isinstance(manifest_csv_sha256, str) or not re.fullmatch(r"^[0-9a-f]{64}$", manifest_csv_sha256):
        raise ValueError(
            f"Manifest 'csv_sha256' must be a valid 64-character lowercase hex string, got {manifest_csv_sha256!r}"
        )
    if manifest_csv_sha256 != csv_sha256:
        raise ValueError(
            f"Stale manifest binding: manifest csv_sha256 ({manifest_csv_sha256}) "
            f"does not match actual CSV SHA256 ({csv_sha256})"
        )

    # Ensemble manifest audit
    ensemble_type = manifest_data.get("ensemble_type")
    if ensemble_type is not None:
        if ensemble_type != "two_component_foreground_ensemble":
            raise ValueError(f"Unsupported ensemble_type: {ensemble_type!r}; expected 'two_component_foreground_ensemble'")

        inf_policy = manifest_data.get("inference_policy")
        if inf_policy != "identity":
            raise ValueError(f"Ensemble manifest inference_policy must be 'identity', got {inf_policy!r}")

        components = manifest_data.get("components")
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError(f"Ensemble manifest must contain exactly 2 components, got: {components}")

        comp_names = [c.get("name") if isinstance(c, dict) else None for c in components]
        if sorted(comp_names) != ["b1", "parent"]:
            raise ValueError(f"Ensemble components must be exactly one 'parent' and one 'b1', got {comp_names}")

        seen_ckpt_shas = set()
        total_weight = 0.0
        comp_records = []
        for c in components:
            if not isinstance(c, dict):
                raise ValueError(f"Invalid component in ensemble manifest: {c}")
            c_name = c.get("name")
            c_path = c.get("checkpoint_path")
            c_sha = c.get("checkpoint_sha256")
            m_sha = c.get("model_state_sha256")
            w = c.get("weight")

            if not c_name or not isinstance(c_name, str):
                raise ValueError(f"Component missing valid name: {c}")
            if not isinstance(c_sha, str) or not re.fullmatch(r"^[0-9a-f]{64}$", c_sha):
                raise ValueError(f"Component '{c_name}' checkpoint_sha256 must be 64-hex lowercase, got {c_sha!r}")
            if c_sha in seen_ckpt_shas:
                raise ValueError(f"Duplicate checkpoint hash detected across components: {c_sha}")
            seen_ckpt_shas.add(c_sha)

            if not isinstance(m_sha, str) or not re.fullmatch(r"^[0-9a-f]{64}$", m_sha):
                raise ValueError(f"Component '{c_name}' model_state_sha256 must be 64-hex lowercase, got {m_sha!r}")
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise TypeError(f"Component '{c_name}' weight must be numeric, got {w!r}")
            w_float = float(w)
            if not math.isfinite(w_float) or not (0.0 < w_float < 1.0):
                raise ValueError(f"Component '{c_name}' weight must be strictly between 0 and 1, got {w}")
            total_weight += w_float

            # Mandatory real readable checkpoint file and SHA verification
            if not c_path or not isinstance(c_path, (str, Path)):
                raise ValueError(f"Component '{c_name}' missing required checkpoint_path")
            c_path_obj = Path(c_path)
            if not c_path_obj.is_file():
                raise FileNotFoundError(f"Component '{c_name}' checkpoint file not found at: {c_path}")

            act_sha = compute_file_sha256(c_path_obj)
            if act_sha != c_sha:
                raise ValueError(f"Component '{c_name}' checkpoint file SHA mismatch: expected {c_sha}, got {act_sha}")

            # Verify actual model-state hash derived directly from the checkpoint file
            try:
                import torch
                from src.inference.engine import compute_state_dict_sha256
                ckpt_obj = torch.load(str(c_path_obj), map_location="cpu", weights_only=False)
                state_dict = ckpt_obj.get("model_state_dict", ckpt_obj)
                act_m_sha = compute_state_dict_sha256(state_dict)
                if act_m_sha != m_sha:
                    raise ValueError(
                        f"Component '{c_name}' model_state_sha256 mismatch: manifest declared {m_sha}, "
                        f"but actual tensor state digest is {act_m_sha}"
                    )
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                raise ValueError(f"Failed to load and verify model state from checkpoint '{c_path}': {e}") from e

            comp_records.append({
                "name": c_name,
                "checkpoint_sha256": c_sha,
                "model_state_sha256": m_sha,
                "weight": w_float,
            })

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Ensemble component weights must sum to 1.0, got {total_weight}")

        ens_def_sha = manifest_data.get("ensemble_definition_sha256")
        if not ens_def_sha or not isinstance(ens_def_sha, str) or not re.fullmatch(r"^[0-9a-f]{64}$", ens_def_sha):
            raise ValueError(f"Manifest missing valid 64-hex 'ensemble_definition_sha256', got {ens_def_sha!r}")

        from src.inference.ensemble import compute_ensemble_definition_sha256
        pp_params = manifest_data.get("postprocess_params", {})
        expected_def_sha = compute_ensemble_definition_sha256(comp_records, pp_params)
        if ens_def_sha != expected_def_sha:
            raise ValueError(
                f"Tampered ensemble definition hash: manifest has {ens_def_sha}, "
                f"recomputed canonical hash is {expected_def_sha}"
            )

        # Mandatory selection_config_path and SHA verification
        sel_cfg_path = manifest_data.get("selection_config_path")
        sel_cfg_sha = manifest_data.get("selection_config_sha256")
        if not sel_cfg_path or not isinstance(sel_cfg_path, (str, Path)):
            raise ValueError("Ensemble manifest missing required 'selection_config_path'")
        sel_cfg_path_obj = Path(sel_cfg_path)
        if not sel_cfg_path_obj.is_file():
            raise FileNotFoundError(f"selection_config file not found at: {sel_cfg_path}")

        if not sel_cfg_sha or not isinstance(sel_cfg_sha, str) or not re.fullmatch(r"^[0-9a-f]{64}$", sel_cfg_sha):
            raise ValueError(f"Ensemble manifest missing valid 64-hex 'selection_config_sha256', got {sel_cfg_sha!r}")

        act_cfg_sha = compute_file_sha256(sel_cfg_path_obj)
        if act_cfg_sha != sel_cfg_sha:
            raise ValueError(
                f"selection_config file content SHA mismatch: expected {sel_cfg_sha}, got {act_cfg_sha}"
            )

        # Load and reconcile selection config artifact against manifest
        with open(sel_cfg_path_obj, "r", encoding="utf-8") as f:
            sel_cfg_data = json.load(f)

        # Check components in selection config
        cfg_comps = {c["name"]: c for c in sel_cfg_data.get("components", []) if isinstance(c, dict)}
        for c in comp_records:
            name = c["name"]
            if name not in cfg_comps:
                raise ValueError(f"Component '{name}' in manifest missing from selection config")
            if cfg_comps[name].get("checkpoint_sha256") != c["checkpoint_sha256"]:
                raise ValueError(f"Component '{name}' checkpoint_sha256 mismatch between manifest and selection config")
            if cfg_comps[name].get("model_state_sha256") != c["model_state_sha256"]:
                raise ValueError(f"Component '{name}' model_state_sha256 mismatch between manifest and selection config")
            if abs(float(cfg_comps[name].get("weight", 0.0)) - c["weight"]) > 1e-6:
                raise ValueError(f"Component '{name}' weight mismatch between manifest and selection config")

        # Validate postprocessing and runtime metadata fields
        inf_cfg = sel_cfg_data.get("inference_config", {})
        for param_dict, dict_name in [(pp_params, "manifest postprocess_params"), (inf_cfg, "selection inference_config")]:
            if param_dict.get("inference_policy") != "identity":
                raise ValueError(f"{dict_name} inference_policy must be 'identity', got {param_dict.get('inference_policy')!r}")
            if param_dict.get("precision") != "float32":
                raise ValueError(f"{dict_name} precision must be 'float32', got {param_dict.get('precision')!r}")
            if param_dict.get("norm_mode") != "imagenet":
                raise ValueError(f"{dict_name} norm_mode must be 'imagenet', got {param_dict.get('norm_mode')!r}")
            if param_dict.get("tile_size") != 512:
                raise ValueError(f"{dict_name} tile_size must be 512, got {param_dict.get('tile_size')!r}")
            if param_dict.get("stride") != 256:
                raise ValueError(f"{dict_name} stride must be 256, got {param_dict.get('stride')!r}")
            t_batch = param_dict.get("tile_batch_size")
            if isinstance(t_batch, bool) or not isinstance(t_batch, int) or t_batch <= 0:
                raise ValueError(f"{dict_name} tile_batch_size must be a positive integer, got {t_batch!r}")
            if param_dict.get("method") != "connected_components":
                raise ValueError(f"{dict_name} method must be 'connected_components', got {param_dict.get('method')!r}")

        # Reconcile threshold & area fields between pp_params and inf_cfg
        for key in ["high_threshold", "low_threshold", "min_area", "max_instances"]:
            if pp_params.get(key) != inf_cfg.get(key):
                raise ValueError(
                    f"Mismatch on '{key}' between manifest postprocess_params ({pp_params.get(key)}) "
                    f"and selection inference_config ({inf_cfg.get(key)})"
                )

    observations = manifest_data.get("observations")
    if observations is None or not isinstance(observations, list):
        raise ValueError("Manifest must contain an 'observations' list")

    manifest_obs_ids: Set[str] = set()
    total_manifest_instances = 0
    processed_count = 0
    abstained_count = 0

    for item in observations:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid observation record in manifest: {item}")
        obs_id = item.get("observation_id")
        status = item.get("status")
        count = item.get("instance_count")

        if not obs_id or not isinstance(obs_id, str):
            raise ValueError(f"Invalid or missing observation_id in manifest record: {item}")
        if obs_id in manifest_obs_ids:
            raise ValueError(f"Duplicate observation_id in manifest: '{obs_id}'")
        manifest_obs_ids.add(obs_id)

        # Reject booleans (isinstance(True, int) is True in Python!) and enforce non-negative integer
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Observation '{obs_id}' has invalid instance_count: {count}; must be non-negative integer")

        if status == "processed":
            if count == 0:
                raise ValueError(f"Observation '{obs_id}' has status 'processed' but instance_count is 0")
            processed_count += 1
        elif status == "abstained":
            if count != 0:
                raise ValueError(f"Observation '{obs_id}' has status 'abstained' but instance_count is {count} (must be 0)")
            abstained_count += 1
        else:
            raise ValueError(f"Observation '{obs_id}' has invalid terminal status '{status}'. Must be 'processed' or 'abstained'")

        # Verify against actual CSV rows for this observation
        csv_count = csv_obs_counts.get(obs_id, 0)
        if count != csv_count:
            raise ValueError(
                f"Instance count mismatch for observation '{obs_id}': "
                f"manifest claims {count}, but CSV has {csv_count} rows"
            )

        total_manifest_instances += count

    # Aggregate reconciliation
    if total_manifest_instances != csv_report["total_instances"]:
        raise ValueError(
            f"Aggregate instance count mismatch: manifest sum {total_manifest_instances} "
            f"!= CSV total {csv_report['total_instances']}"
        )

    # Validate and reconcile any optional aggregate summary fields if provided
    total_obs_count = len(manifest_obs_ids)
    summary_checks = [
        ("total_instances", total_manifest_instances),
        ("total_test_observations", total_obs_count),
        ("detected_observations_count", processed_count),
        ("abstained_observations_count", abstained_count),
    ]
    for key, expected_val in summary_checks:
        if key in manifest_data:
            val = manifest_data[key]
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"Manifest summary field '{key}' must be a non-negative integer, got {val!r}")
            if val != expected_val:
                raise ValueError(
                    f"Manifest summary field '{key}' mismatch: manifest has {val}, "
                    f"but calculated from observation records is {expected_val}"
                )

    # Exact expected observation set audit
    if expected_observation_ids is not None:
        if manifest_obs_ids != expected_observation_ids:
            missing = expected_observation_ids - manifest_obs_ids
            extra = manifest_obs_ids - expected_observation_ids
            raise ValueError(
                f"Manifest observation set mismatch: missing {len(missing)} ({missing}), extra {len(extra)} ({extra})"
            )

    return {
        "csv_report": csv_report,
        "manifest_path": str(manifest_path),
        "csv_sha256": csv_sha256,
        "total_manifest_observations": len(manifest_obs_ids),
        "processed_observations": processed_count,
        "abstained_observations": abstained_count,
        "total_instances": total_manifest_instances,
        "is_valid": True,
    }
