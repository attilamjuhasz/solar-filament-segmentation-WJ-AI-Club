import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.dataset import SolarFilamentDataset
from src.data.folds import load_frozen_folds_manifest

manifest_path = Path("artifacts/folds_manifest.json")
assignments, _ = load_frozen_folds_manifest(manifest_path)
data_dir = Path("data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
train_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
train_images = data_dir / "train" / "train_images"

train_ds = SolarFilamentDataset(images_dir=train_images, annotations_json=train_json, fold_assignments=assignments, target_fold=0, is_train=True, patch_size=(512, 512), fg_crop_prob=1.0, norm_mode="imagenet", seed=42)

for i in [0, 1, 2, 3]:
    s = train_ds[i]
    v = s["valid_mask"][0]
    img = s["image"]
    fg = s["target_fg"][0]
    print(f"Sample {i}: obs {s['observation_id']}, origin {s['origin']}, valid mean: {v.mean().item():.3f}, valid sum: {v.sum().item()}, fg sum: {fg.sum().item()}, img min: {img.min().item():.3f}, max: {img.max().item():.3f}")
