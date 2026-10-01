import json
from pathlib import Path
import sys
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluate import evaluate_oof
from src.inference.engine import compute_file_sha256

def main():
    ckpt_path = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt")
    file_sha = compute_file_sha256(ckpt_path)
    print(f"Evaluating {ckpt_path.name} (SHA: {file_sha}) on CONFIRMATION...")

    report = evaluate_oof(
        checkpoint_path=str(ckpt_path),
        fold=0,
        split="confirmation",
        method="connected_components",
        high_threshold=0.85,
        low_threshold=0.60,
        min_area=400,
        max_instances=12,
        tile_size=512,
        stride=256,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
    )

    out_file = Path("artifacts/reports/eval_selected_epoch06_confirmation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    ov = report["overall"]
    print("=== CONFIRMATION RESULTS (9 UNIQUE PHYSICAL GROUPS) ===")
    print(f"PQ: {ov['pq']:.4f} (SQ: {ov['sq']:.4f}, RQ: {ov['rq']:.4f})")
    print(f"Dice: {ov['mean_dice']:.4f}")
    print(f"TP: {ov['tp']}, FP: {ov['fp']}, FN: {ov['fn']}")
    print(f"Miss Rate: {ov['miss_rate']*100:.1f}%, Frag Rate: {ov['fragmentation_rate']*100:.1f}%, Spurious Rate: {ov['spurious_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
