from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


def apply_geometric_flips(
    sample: Dict[str, Any],
    aug_rng: np.random.RandomState,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
) -> Dict[str, Any]:
    """Apply label-aligned horizontal and vertical geometric flips to training crops.
    
    Guarantees:
    - Dedicated seeded augmentation RNG (separate from annotator and dataset sampling RNGs).
    - Horizontal flip (p=0.5): flips image, fg, bnd, skel, ctr, valid_mask horizontally.
      For target_off (2, H, W): flips spatial coords horizontally and negates dx (channel 1).
    - Vertical flip (p=0.5): flips image, fg, bnd, skel, ctr, valid_mask vertically.
      For target_off (2, H, W): flips spatial coords vertically and negates dy (channel 0).
    - Preserves normalization and tensor dtypes exactly with zero interpolation or resizing.
    """
    do_hflip = bool(aug_rng.uniform(0.0, 1.0) < hflip_prob)
    do_vflip = bool(aug_rng.uniform(0.0, 1.0) < vflip_prob)

    if not do_hflip and not do_vflip:
        return sample

    out_sample = dict(sample)

    spatial_keys = ["image", "target_fg", "target_bnd", "target_skel", "target_ctr", "valid_mask"]

    if do_hflip:
        for k in spatial_keys:
            if k in out_sample and out_sample[k] is not None:
                val = out_sample[k]
                if isinstance(val, torch.Tensor):
                    out_sample[k] = torch.flip(val, dims=[-1])
                elif isinstance(val, np.ndarray):
                    out_sample[k] = np.flip(val, axis=-1).copy()

        if "target_off" in out_sample and out_sample["target_off"] is not None:
            off = out_sample["target_off"]
            if isinstance(off, torch.Tensor):
                # Channel 0: dy, Channel 1: dx
                off_flipped = torch.flip(off, dims=[-1]).clone()
                off_flipped[1] = -off_flipped[1]
                out_sample["target_off"] = off_flipped
            elif isinstance(off, np.ndarray):
                off_flipped = np.flip(off, axis=-1).copy()
                off_flipped[1] = -off_flipped[1]
                out_sample["target_off"] = off_flipped

    if do_vflip:
        for k in spatial_keys:
            if k in out_sample and out_sample[k] is not None:
                val = out_sample[k]
                if isinstance(val, torch.Tensor):
                    out_sample[k] = torch.flip(val, dims=[-2])
                elif isinstance(val, np.ndarray):
                    out_sample[k] = np.flip(val, axis=-2).copy()

        if "target_off" in out_sample and out_sample["target_off"] is not None:
            off = out_sample["target_off"]
            if isinstance(off, torch.Tensor):
                # Channel 0: dy, Channel 1: dx
                off_flipped = torch.flip(off, dims=[-2]).clone()
                off_flipped[0] = -off_flipped[0]
                out_sample["target_off"] = off_flipped
            elif isinstance(off, np.ndarray):
                off_flipped = np.flip(off, axis=-2).copy()
                off_flipped[0] = -off_flipped[0]
                out_sample["target_off"] = off_flipped

    return out_sample
