import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from scripts.diagnose_overfit import *

def inspect():
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_images = torch.stack([s["image"] for s in fixed_samples]).to(device)
    batch_fg = torch.stack([s["target_fg"] for s in fixed_samples]).to(device)
    batch_valid = torch.stack([s["valid_mask"] for s in fixed_samples]).to(device)

    model = ResNet34UNet(in_channels=3, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PureBCEDiceLoss()

    for step in range(1, 151):
        model.train()
        optimizer.zero_grad()
        preds = model(batch_images)["fg_logits"].float()
        loss, _ = criterion(preds, batch_fg, valid_mask=batch_valid)
        loss.backward()
        optimizer.step()

        if step in [50, 100, 150]:
            model.eval()
            with torch.no_grad():
                eval_preds = model(batch_images)["fg_logits"].float()
                probs = torch.sigmoid(eval_preds)
                print(f"\n--- STEP {step} ---")
                for i in range(4):
                    p_i = probs[i, 0]
                    t_i = batch_fg[i, 0]
                    for thresh in [0.2, 0.35, 0.5, 0.65, 0.8]:
                        bin_p = (p_i >= thresh).float()
                        inter = (bin_p * t_i).sum().item()
                        p_sum = bin_p.sum().item()
                        t_sum = t_i.sum().item()
                        dice = (2.0 * inter) / (p_sum + t_sum) if (p_sum + t_sum) > 0 else 0.0
                        prec = inter / p_sum if p_sum > 0 else 0.0
                        rec = inter / t_sum if t_sum > 0 else 0.0
                        print(f"Sample {i} @ thresh {thresh:.2f}: Dice={dice:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, PredArea={int(p_sum)}, GTArea={int(t_sum)}")

if __name__ == "__main__":
    inspect()
