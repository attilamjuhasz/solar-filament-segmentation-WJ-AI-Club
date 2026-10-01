from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.contracts import InstancePrediction, NATIVE_IMAGE_SHAPE
from src.data.dataset import extract_solar_disk_mask, load_solar_image, normalize_solar_image
from src.inference.instances import extract_instances_from_maps
from src.inference.tiling import blend_tiles, extract_tiles


def compute_bytes_sha256(data: bytes) -> str:
    """Compute lowercase SHA-256 hash of byte buffer."""
    return hashlib.sha256(data).hexdigest().lower()


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute lowercase SHA-256 hash of file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def compute_state_dict_sha256(state_dict: dict) -> str:
    """Compute deterministic lowercase SHA-256 hash of PyTorch model state_dict tensors."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest().lower()


def predict_full_observation(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    stride: int = 384,
    tile_batch_size: int = 8,
    use_amp: bool = True,
    norm_mode: str = "scale_0_1",
    include_aux: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Perform sliding-window tiled inference on full 2048x2048 solar observation.
    
    Returns:
        (foreground_prob, center_prob, boundary_prob, offset_field) in native (2048, 2048) resolution.
    """
    model.eval()
    H, W, _ = image_rgb.shape
    img_chw = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
    norm_tensor = normalize_solar_image(img_chw, mode=norm_mode)
    img_norm_chw = norm_tensor.numpy()

    # Extract overlapping tiles
    tiles = extract_tiles(img_norm_chw, tile_size=tile_size, stride=stride, layout="CHW")

    fg_tiles = []
    ctr_tiles = []
    bnd_tiles = []
    off_tiles = []

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Process in mini-batches
    with torch.no_grad():
        for i in range(0, len(tiles), tile_batch_size):
            batch_slice = tiles[i : i + tile_batch_size]
            batch_boxes = [box for box, _ in batch_slice]
            batch_arrays = [tile for _, tile in batch_slice]

            batch_t = torch.from_numpy(np.stack(batch_arrays, axis=0)).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
                preds = model(batch_t)
                fg_batch = torch.sigmoid(preds["fg_logits"].float()).cpu().numpy()
                if include_aux:
                    ctr_batch = torch.sigmoid(preds["ctr_logits"].float()).cpu().numpy()
                    bnd_batch = torch.sigmoid(preds["bnd_logits"].float()).cpu().numpy()
                    off_batch = preds["off_pred"].float().cpu().numpy()

            for b_idx, box in enumerate(batch_boxes):
                fg_tiles.append((box, fg_batch[b_idx, 0]))
                if include_aux:
                    ctr_tiles.append((box, ctr_batch[b_idx, 0]))
                    bnd_tiles.append((box, bnd_batch[b_idx, 0]))
                    off_tiles.append((box, off_batch[b_idx]))

    # Reconstruct native 2048x2048 maps using smooth Hann window weighting
    full_fg = blend_tiles(fg_tiles, canvas_spatial_shape=(H, W))
    disk_mask = extract_solar_disk_mask(image_rgb)
    full_fg *= disk_mask

    if not include_aux:
        return full_fg, None, None, None

    full_ctr = blend_tiles(ctr_tiles, canvas_spatial_shape=(H, W))
    full_bnd = blend_tiles(bnd_tiles, canvas_spatial_shape=(H, W))
    full_off = blend_tiles(off_tiles, canvas_spatial_shape=(H, W))

    full_ctr *= disk_mask
    full_bnd *= disk_mask
    full_off[0] *= disk_mask
    full_off[1] *= disk_mask

    return full_fg, full_ctr, full_bnd, full_off


def get_cache_path(
    cache_dir: Union[str, Path],
    ckpt_hash: str,
    obs_id: str,
    inference_policy: str = "identity",
) -> Path:
    """Standardized path for cached probability/offset maps."""
    if inference_policy == "identity":
        return Path(cache_dir) / f"{ckpt_hash[:16]}_{obs_id}.npz"
    return Path(cache_dir) / f"{ckpt_hash[:16]}_{obs_id}_{inference_policy}.npz"


def load_cached_prediction(
    cache_dir: Union[str, Path],
    ckpt_hash: str,
    obs_id: str,
    expected_image_sha256: Optional[str] = None,
    expected_model_state_sha256: Optional[str] = None,
    expected_spatial_shape: Optional[Tuple[int, int]] = None,
    expected_tile_size: int = 512,
    expected_stride: int = 256,
    expected_norm_mode: str = "imagenet",
    expected_precision: str = "float32",
    expected_preprocessing_version: Optional[str] = "v3",
    expected_inference_policy: str = "identity",
    requires_aux: bool = False,
    strict: bool = False,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    """Retrieve pre-computed prediction maps from disk if available and metadata strictly matches."""
    cache_file = get_cache_path(cache_dir, ckpt_hash, obs_id, inference_policy=expected_inference_policy)
    if not cache_file.is_file():
        return None
    try:
        with np.load(str(cache_file)) as data:
            schema_ver = int(data.get("schema_version", 1))
            if strict:
                # In strict mode, require schema version >= 3 and complete valid metadata
                if schema_ver < 3:
                    return None

                # Strict mode rejects missing/empty expected hashes and spatial shape
                if not expected_image_sha256 or not str(expected_image_sha256).strip():
                    return None
                if not expected_model_state_sha256 or not str(expected_model_state_sha256).strip():
                    return None
                if expected_spatial_shape is None:
                    return None

                required_keys = (
                    "ckpt_hash", "obs_id", "spatial_shape", "tile_size", "stride", "norm_mode", "precision"
                )
                for k in required_keys:
                    if k not in data:
                        return None
                    val_str = str(data[k])
                    if not val_str or val_str.strip() == "":
                        return None

                # Exact full string checks
                if str(data["ckpt_hash"]) != str(ckpt_hash):
                    return None
                if str(data["obs_id"]) != str(obs_id):
                    return None
                if "model_state_sha256" not in data or str(data["model_state_sha256"]) != str(expected_model_state_sha256):
                    return None
                if "image_sha256" not in data or str(data["image_sha256"]) != str(expected_image_sha256):
                    return None

                cached_shape = tuple(int(x) for x in data["spatial_shape"])
                if cached_shape != expected_spatial_shape:
                    return None
                if int(data["tile_size"]) != expected_tile_size:
                    return None
                if int(data["stride"]) != expected_stride:
                    return None
                if str(data["norm_mode"]) != expected_norm_mode:
                    return None
                if str(data["precision"]) != expected_precision:
                    return None
                if expected_preprocessing_version is not None and str(data.get("preprocessing_version", "")) != expected_preprocessing_version:
                    return None
                if str(data.get("inference_policy", "identity")) != expected_inference_policy:
                    return None

                # Reject cache missing auxiliary arrays when policy requires them
                if requires_aux:
                    if "ctr" not in data or data["ctr"].size == 0:
                        return None
                    if "bnd" not in data or data["bnd"].size == 0:
                        return None
                    if "off" not in data or data["off"].size == 0:
                        return None

                # Verify raw storage array dtype and shape
                raw_fg = data["fg"]
                if raw_fg.shape != expected_spatial_shape:
                    return None
                if expected_precision == "float32" and raw_fg.dtype != np.float32:
                    return None
                if expected_precision == "float16" and raw_fg.dtype != np.float16:
                    return None
            else:
                # Lenient / legacy mode
                if "schema_version" in data:
                    if expected_image_sha256 is not None:
                        cached_img_sha = str(data.get("image_sha256", ""))
                        if cached_img_sha and cached_img_sha != expected_image_sha256:
                            return None

                    if expected_model_state_sha256 is not None:
                        cached_ms_sha = str(data.get("model_state_sha256", ""))
                        if cached_ms_sha and cached_ms_sha != expected_model_state_sha256:
                            return None

                    if expected_spatial_shape is not None and "spatial_shape" in data:
                        cached_shape = tuple(int(x) for x in data["spatial_shape"])
                        if cached_shape != expected_spatial_shape:
                            return None

                    if int(data.get("tile_size", -1)) != expected_tile_size:
                        return None
                    if int(data.get("stride", -1)) != expected_stride:
                        return None
                    if str(data.get("norm_mode", "")) != expected_norm_mode:
                        return None
                    if str(data.get("precision", "")) != expected_precision:
                        return None

            fg = data["fg"].astype(np.float32)
            if not np.isfinite(fg).all():
                return None
            if fg.min() < -1e-5 or fg.max() > (1.0 + 1e-5):
                return None
            np.clip(fg, 0.0, 1.0, out=fg)

            ctr = data["ctr"].astype(np.float32) if ("ctr" in data and data["ctr"].size > 0) else None
            bnd = data["bnd"].astype(np.float32) if ("bnd" in data and data["bnd"].size > 0) else None
            off = data["off"].astype(np.float32) if ("off" in data and data["off"].size > 0) else None

            if ctr is not None:
                if not np.isfinite(ctr).all() or ctr.min() < -1e-5 or ctr.max() > (1.0 + 1e-5):
                    return None
                np.clip(ctr, 0.0, 1.0, out=ctr)
            if bnd is not None:
                if not np.isfinite(bnd).all() or bnd.min() < -1e-5 or bnd.max() > (1.0 + 1e-5):
                    return None
                np.clip(bnd, 0.0, 1.0, out=bnd)
            if off is not None and not np.isfinite(off).all():
                return None

            return fg, ctr, bnd, off
    except Exception:
        return None


def save_cached_prediction(
    cache_dir: Union[str, Path],
    ckpt_hash: str,
    obs_id: str,
    fg: np.ndarray,
    ctr: Optional[np.ndarray] = None,
    bnd: Optional[np.ndarray] = None,
    off: Optional[np.ndarray] = None,
    image_sha256: str = "",
    model_state_sha256: str = "",
    tile_size: int = 512,
    stride: int = 256,
    norm_mode: str = "imagenet",
    precision: str = "float32",
    preprocessing_version: str = "v3",
    inference_policy: str = "identity",
) -> Path:
    """Persist prediction maps with full metadata and controlled precision (FP32/FP16)."""
    if precision not in ("float32", "float16"):
        raise ValueError(f"Unsupported cache precision '{precision}'. Must be 'float32' or 'float16'.")

    dtype = np.float32 if precision == "float32" else np.float16

    cache_path = get_cache_path(cache_dir, ckpt_hash, obs_id, inference_policy=inference_policy)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "fg": fg.astype(dtype),
        "ctr": ctr.astype(dtype) if ctr is not None else np.array([], dtype=dtype),
        "bnd": bnd.astype(dtype) if bnd is not None else np.array([], dtype=dtype),
        "off": off.astype(dtype) if off is not None else np.array([], dtype=dtype),
        "ckpt_hash": np.array(str(ckpt_hash)),
        "model_state_sha256": np.array(str(model_state_sha256)),
        "obs_id": np.array(str(obs_id)),
        "image_sha256": np.array(str(image_sha256)),
        "spatial_shape": np.array(list(fg.shape)),
        "tile_size": np.array(int(tile_size)),
        "stride": np.array(int(stride)),
        "norm_mode": np.array(str(norm_mode)),
        "precision": np.array(str(precision)),
        "preprocessing_version": np.array(str(preprocessing_version)),
        "inference_policy": np.array(str(inference_policy)),
        "schema_version": np.array(3),
    }
    np.savez_compressed(str(cache_path), **save_dict)
    return cache_path

