import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
from scripts.diagnose_overfit import *

def analyze():
    manifest_path = Path("artifacts/folds_manifest.json")
    assignments, _ = load_frozen_folds_manifest(manifest_path)
    data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_images = data_dir / "train" / "train_images"

    train_ds = SolarFilamentDataset(
        images_dir=train_images,
        annotations_json=train_json,
        fold_assignments=assignments,
        target_fold=0,
        is_train=True,
        patch_size=(512, 512),
        fg_crop_prob=1.0,
        norm_mode="imagenet",
        seed=42,
    )

    fixed_samples = []
    for i in range(50):
        sample = train_ds[i]
        if int(sample["target_fg"].sum().item()) >= 1000:
            fixed_samples.append(sample)
            if len(fixed_samples) >= 4:
                break

    for idx, s in enumerate(fixed_samples):
        obs_id = s["observation_id"]
        origin = s["origin"]
        variants = train_ds.get_observation_annotations(obs_id)
        print(f"\nSample {idx}: Obs {obs_id}, Origin {origin}")
        print(f"Total annotator variants for this observation: {len(variants)}")
        for v_idx, v in enumerate(variants):
            print(f"  Variant {v_idx} ({v.annotator_image_id}): {len(v.instances)} instances")
            # Check how many instances overlap with this crop
            overlap_areas = []
            for inst in v.instances:
                m = inst.get_mask((2048, 2048))[origin[0]:origin[0]+512, origin[1]:origin[1]+512]
                if (m > 0).any():
                    overlap_areas.append(int((m > 0).sum()))
            print(f"    In-crop instance areas: {overlap_areas}, Total in-crop FG: {sum(overlap_areas)}")

if __name__ == "__main__":
    analyze()
