from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.contracts import NATIVE_IMAGE_SHAPE, TargetBundle
from src.data.annotations import AnnotationIndex, ObservationAnnotations, load_coco_annotations
from src.data.manifest import canonical_observation_id
from src.data.targets import (
    NATIVE_NORM_SCALE,
    compute_instance_anchor,
    extract_instance_skeletons,
    make_targets,
)
from src.training.augmentation import apply_geometric_flips


def load_solar_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from disk as RGB uint8 array of shape [H, W, 3]."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Solar image not found: {path}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def extract_solar_disk_mask(image: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """Derive solar disk validity mask [H, W] from RGB image.
    
    Pixels outside the solar disk / telescope frame are near black.
    """
    if image.ndim == 3:
        intensity = np.mean(image, axis=2)
    else:
        intensity = image.astype(np.float32)
    return (intensity > threshold).astype(np.uint8)


def normalize_solar_image(image_chw: np.ndarray, mode: str = "scale_0_1") -> torch.Tensor:
    """Normalize solar image CHW array to PyTorch float32 tensor."""
    t = torch.from_numpy(np.ascontiguousarray(image_chw)).float() / 255.0
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        t = (t - mean) / std
    return t


class SolarFilamentDataset(Dataset):
    """PyTorch Dataset for Solar Filament Segmentation & Instance Delination.
    
    Guarantees:
    - Lazy mask decoding from indexed COCO annotations to avoid host memory bloat.
    - Observation-group fold partitioning: all annotator variants of an observation
      belong strictly to the same fold.
    - Variant sampling policy: in train mode, samples exactly ONE annotator variant
      per observation per epoch without unioning across different annotators.
    - Preserves native coordinate normalization and global anchors when cropping patches.
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        annotations_json: Optional[Union[str, Path]] = None,
        annotation_index: Optional[AnnotationIndex] = None,
        fold_assignments: Optional[Dict[str, int]] = None,
        target_fold: Optional[int] = None,
        is_train: bool = True,
        patch_size: Tuple[int, int] = (512, 512),
        crop_mode: str = "random",  # 'random', 'foreground_biased', or 'center'
        fg_crop_prob: float = 0.6,
        seed: Optional[int] = None,
        max_observations: Optional[int] = None,
        norm_mode: str = "scale_0_1",
        mining_bank: Optional[Dict[str, List[List[int]]]] = None,
        crop_policy: str = "standard",
        use_geometric_augmentation: bool = False,
        aug_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.is_train = is_train
        self.max_observations = max_observations
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.fg_crop_prob = fg_crop_prob
        self.norm_mode = norm_mode
        self.mining_bank = mining_bank
        self.crop_policy = crop_policy
        self.use_geometric_augmentation = use_geometric_augmentation
        self.rng = np.random.RandomState(seed if seed is not None else 2026)
        if use_geometric_augmentation:
            self.aug_rng = np.random.RandomState(aug_seed if aug_seed is not None else ((seed if seed is not None else 2026) + 1000))
        else:
            self.aug_rng = None

        # Index annotations if provided
        if annotation_index is not None:
            self.index = annotation_index
        elif annotations_json is not None:
            self.index = load_coco_annotations(str(annotations_json))
        else:
            self.index = None

        # Build observation records & resolve file paths
        self.observations: List[str] = []
        self._image_path_cache: Dict[str, Path] = {}

        # Validate target_fold parameter
        if target_fold is not None:
            if isinstance(target_fold, bool) or not isinstance(target_fold, int) or target_fold < 0 or target_fold >= 5:
                raise ValueError(f"Invalid target_fold {target_fold}; must be integer in [0, 4]")

        if self.index is not None:
            # We have annotations (training / validation)
            all_obs = sorted(list(self.index.by_observation.keys()))
            if fold_assignments is not None and target_fold is not None:
                for obs_id in all_obs:
                    canon_id = canonical_observation_id(obs_id)
                    f = fold_assignments.get(obs_id)
                    if f is None:
                        f = fold_assignments.get(canon_id)
                    if f is None:
                        raise KeyError(
                            f"Observation '{obs_id}' (canonical '{canon_id}') is missing from fold assignments"
                        )
                    if isinstance(f, bool) or not isinstance(f, int) or f < 0 or f >= 5:
                        raise ValueError(
                            f"Invalid fold assignment {f} for observation '{obs_id}'; must be integer in [0, 4]"
                        )
                    if is_train and f != target_fold:
                        self.observations.append(obs_id)
                    elif (not is_train) and f == target_fold:
                        self.observations.append(obs_id)
            else:
                self.observations = all_obs
            if max_observations is not None and max_observations > 0:
                self.observations = self.observations[:max_observations]
        else:
            # Inference mode without annotations: discover all images in images_dir
            image_files = sorted(
                list(self.images_dir.glob("*.jpeg"))
                + list(self.images_dir.glob("*.jpg"))
                + list(self.images_dir.glob("*.png"))
            )
            seen_obs = set()
            for p in image_files:
                obs_id = canonical_observation_id(p.stem)
                if obs_id not in seen_obs:
                    seen_obs.add(obs_id)
                    self.observations.append(obs_id)
                    self._image_path_cache[obs_id] = p

    def __len__(self) -> int:
        return len(self.observations)

    def _resolve_image_path(self, obs_id: str, file_name: Optional[str] = None) -> Path:
        """Find the local image file corresponding to an observation or annotator file_name."""
        if obs_id in self._image_path_cache:
            return self._image_path_cache[obs_id]

        candidates = []
        if file_name:
            candidates.append(self.images_dir / file_name)
            candidates.append(self.images_dir / Path(file_name).name)

        clean_id = canonical_observation_id(obs_id)
        candidates.append(self.images_dir / f"{clean_id}.jpeg")
        candidates.append(self.images_dir / f"{clean_id}.jpg")
        candidates.append(self.images_dir / f"{clean_id}.png")

        # Check by prefix / glob
        for cand in candidates:
            if cand.is_file():
                self._image_path_cache[obs_id] = cand
                return cand

        # Fallback search by stem
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for match in self.images_dir.glob(f"*{clean_id}*{ext}"):
                if match.is_file():
                    self._image_path_cache[obs_id] = match
                    return match

        raise FileNotFoundError(f"Could not find image file for observation {obs_id} in {self.images_dir}")

    def get_observation_annotations(self, obs_id: str) -> List[ObservationAnnotations]:
        """Return all available annotator variants for this observation."""
        if self.index is None:
            return []
        return self.index.get_observation_variants(obs_id)

    def sample_annotator_variant(self, obs_id: str) -> ObservationAnnotations:
        """Sample ONE annotator variant for this observation to avoid multi-annotator mask mixing."""
        variants = self.get_observation_annotations(obs_id)
        if not variants:
            raise KeyError(f"No annotator variants found for observation {obs_id}")
        if len(variants) == 1 or not self.is_train:
            return variants[0]
        # Randomly choose one variant for training epoch
        idx = int(self.rng.randint(0, len(variants)))
        return variants[idx]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        obs_id = self.observations[index]

        # Case 1: Inference / Unannotated mode
        if self.index is None:
            img_path = self._resolve_image_path(obs_id)
            img_rgb = load_solar_image(img_path)
            valid_mask = extract_solar_disk_mask(img_rgb)
            # Normalize image to float32 [3, H, W] in [0, 1]
            img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
            valid_t = torch.from_numpy(valid_mask).unsqueeze(0).float()
            return {
                "observation_id": obs_id,
                "image": img_t,
                "valid_mask": valid_t,
                "raw_image": img_rgb,
            }

        # Case 2: Annotated observation
        variant = self.sample_annotator_variant(obs_id)
        img_path = self._resolve_image_path(obs_id, file_name=variant.file_name)
        img_rgb = load_solar_image(img_path)
        valid_mask = extract_solar_disk_mask(img_rgb)

        # Decode instances on-demand
        raw_instances = variant.instances
        instance_masks = [inst.get_mask(NATIVE_IMAGE_SHAPE) for inst in raw_instances if inst.area > 0]
        # Filter empty instances if any
        instance_masks = [m for m in instance_masks if (m > 0).any()]

        # Precompute global anchors for all instances in native (2048, 2048) frame
        global_anchors: List[Tuple[int, int]] = []
        for m in instance_masks:
            anchor = compute_instance_anchor(m)
            global_anchors.append(anchor)

        H, W = NATIVE_IMAGE_SHAPE
        patch_h, patch_w = self.patch_size

        if self.is_train and (patch_h < H or patch_w < W):
            # Sample crop origin (y0, x0)
            if self.crop_policy == "mining_bank":
                r = float(self.rng.uniform(0.0, 1.0))
                if r < 0.55 and len(global_anchors) > 0:
                    # 55% foreground-biased crop
                    chosen_idx = int(self.rng.randint(0, len(global_anchors)))
                    ay, ax = global_anchors[chosen_idx]
                    y_min = max(0, ay - patch_h + 1)
                    y_max = min(H - patch_h, ay)
                    x_min = max(0, ax - patch_w + 1)
                    x_max = min(W - patch_w, ax)
                    y0 = int(self.rng.randint(y_min, y_max + 1)) if y_max >= y_min else 0
                    x0 = int(self.rng.randint(x_min, x_max + 1)) if x_max >= x_min else 0
                elif r < 0.80:
                    # 25% hard-negative bank crop (redirect to uniform if unavailable for this observation)
                    c_id = canonical_observation_id(obs_id)
                    if self.mining_bank and c_id in self.mining_bank and len(self.mining_bank[c_id]) > 0:
                        boxes = self.mining_bank[c_id]
                        b_idx = int(self.rng.randint(0, len(boxes)))
                        box = boxes[b_idx]
                        y0, x0 = int(box[0]), int(box[1])
                    else:
                        y0 = int(self.rng.randint(0, H - patch_h + 1))
                        x0 = int(self.rng.randint(0, W - patch_w + 1))
                else:
                    # 20% uniform random crop
                    y0 = int(self.rng.randint(0, H - patch_h + 1))
                    x0 = int(self.rng.randint(0, W - patch_w + 1))
            else:
                # Standard crop policy
                use_fg = (self.rng.uniform() < self.fg_crop_prob) and (len(global_anchors) > 0)
                if use_fg:
                    # Pick a random instance anchor to center or include in the crop
                    chosen_idx = int(self.rng.randint(0, len(global_anchors)))
                    ay, ax = global_anchors[chosen_idx]
                    y_min = max(0, ay - patch_h + 1)
                    y_max = min(H - patch_h, ay)
                    x_min = max(0, ax - patch_w + 1)
                    x_max = min(W - patch_w, ax)
                    y0 = int(self.rng.randint(y_min, y_max + 1)) if y_max >= y_min else 0
                    x0 = int(self.rng.randint(x_min, x_max + 1)) if x_max >= x_min else 0
                else:
                    y0 = int(self.rng.randint(0, H - patch_h + 1))
                    x0 = int(self.rng.randint(0, W - patch_w + 1))

            origin = (y0, x0)
            crop_img = img_rgb[y0 : y0 + patch_h, x0 : x0 + patch_w]
            crop_valid = valid_mask[y0 : y0 + patch_h, x0 : x0 + patch_w]

            # Slice instance masks to crop
            crop_instances = []
            crop_anchors = []
            for inst_m, g_anchor in zip(instance_masks, global_anchors):
                sub_m = inst_m[y0 : y0 + patch_h, x0 : x0 + patch_w]
                if (sub_m > 0).any():
                    crop_instances.append(sub_m)
                    crop_anchors.append(g_anchor)

            # Generate target bundle with preserved global anchors and native normalization
            targets = make_targets(
                instance_masks=crop_instances,
                shape=(patch_h, patch_w),
                anchors=crop_anchors,
                origin=origin,
                normalization_scale=NATIVE_NORM_SCALE,
                valid_mask=crop_valid,
            )

            # Medial skeleton extraction
            skel = extract_instance_skeletons(crop_instances, shape=(patch_h, patch_w))

            # Build PyTorch tensors
            crop_chw = np.ascontiguousarray(crop_img.transpose(2, 0, 1))
            img_tensor = normalize_solar_image(crop_chw, mode=self.norm_mode)
            fg_tensor = torch.from_numpy(targets.foreground_mask).unsqueeze(0).float()
            bnd_tensor = torch.from_numpy(targets.boundary_mask).unsqueeze(0).float()
            skel_tensor = torch.from_numpy(skel).unsqueeze(0).float()
            ctr_tensor = torch.from_numpy(targets.center_heatmap).unsqueeze(0).float()
            off_tensor = torch.from_numpy(targets.offset_field).float()  # [2, H, W]
            valid_tensor = torch.from_numpy(targets.valid_mask).unsqueeze(0).float()

            sample = {
                "observation_id": obs_id,
                "variant_id": variant.annotator_image_id,
                "origin": origin,
                "image": img_tensor,
                "target_fg": fg_tensor,
                "target_bnd": bnd_tensor,
                "target_skel": skel_tensor,
                "target_ctr": ctr_tensor,
                "target_off": off_tensor,
                "valid_mask": valid_tensor,
            }

            if self.use_geometric_augmentation and self.aug_rng is not None:
                sample = apply_geometric_flips(sample, self.aug_rng)

            return sample

        else:
            # Full native size (Validation or full evaluation)
            targets = make_targets(
                instance_masks=instance_masks,
                shape=(H, W),
                anchors=global_anchors,
                origin=(0, 0),
                normalization_scale=NATIVE_NORM_SCALE,
                valid_mask=valid_mask,
            )
            skel = extract_instance_skeletons(instance_masks, shape=(H, W))

            img_chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
            img_tensor = normalize_solar_image(img_chw, mode=self.norm_mode)
            fg_tensor = torch.from_numpy(targets.foreground_mask).unsqueeze(0).float()
            bnd_tensor = torch.from_numpy(targets.boundary_mask).unsqueeze(0).float()
            skel_tensor = torch.from_numpy(skel).unsqueeze(0).float()
            ctr_tensor = torch.from_numpy(targets.center_heatmap).unsqueeze(0).float()
            off_tensor = torch.from_numpy(targets.offset_field).float()
            valid_tensor = torch.from_numpy(targets.valid_mask).unsqueeze(0).float()

            return {
                "observation_id": obs_id,
                "variant_id": variant.annotator_image_id,
                "origin": (0, 0),
                "image": img_tensor,
                "target_fg": fg_tensor,
                "target_bnd": bnd_tensor,
                "target_skel": skel_tensor,
                "target_ctr": ctr_tensor,
                "target_off": off_tensor,
                "valid_mask": valid_tensor,
                "gt_instances": instance_masks,
            }
