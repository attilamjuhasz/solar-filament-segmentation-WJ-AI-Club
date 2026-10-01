import glob
import json
import os
from pathlib import Path
import sys
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluate import evaluate_oof
from src.inference.engine import compute_file_sha256

def rescore_epochs():
    ckpt_dir = Path("artifacts/runs/run_20260930_111126_15cb60/checkpoints")
    ckpt_paths = sorted(glob.glob(str(ckpt_dir / "epoch_*.pt")))
    
    print(f"Found {len(ckpt_paths)} immutable epoch checkpoints.")
    
    results = []
    frozen_params = {
        "method": "connected_components",
        "high_threshold": 0.85,
        "low_threshold": 0.60,
        "min_area": 400,
        "max_instances": 12,
        "tile_size": 512,
        "stride": 256,
    }
    
    for cp in ckpt_paths:
        p = Path(cp)
        file_sha = compute_file_sha256(p)
        print(f"\n--- Evaluating {p.name} (SHA: {file_sha[:12]}...) on TUNING ---")
        
        report = evaluate_oof(
            checkpoint_path=str(p),
            fold=0,
            split="tuning",
            method=frozen_params["method"],
            high_threshold=frozen_params["high_threshold"],
            low_threshold=frozen_params["low_threshold"],
            min_area=frozen_params["min_area"],
            max_instances=frozen_params["max_instances"],
            tile_size=frozen_params["tile_size"],
            stride=frozen_params["stride"],
            device_str="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        ov = report["overall"]
        ckpt_meta = torch.load(str(p), map_location="cpu", weights_only=False)
        epoch_num = ckpt_meta.get("epoch")
        
        res_entry = {
            "epoch": epoch_num,
            "filename": p.name,
            "path": str(p),
            "file_sha256": file_sha,
            "pq": ov["pq"],
            "sq": ov["sq"],
            "rq": ov["rq"],
            "mean_dice": ov["mean_dice"],
            "tp": ov["tp"],
            "fp": ov["fp"],
            "fn": ov["fn"],
            "total_gt": ov["total_gt_instances"],
            "total_pred": ov["total_pred_instances"],
            "miss_rate": ov["miss_rate"],
            "fragmentation_rate": ov["fragmentation_rate"],
            "spurious_rate": ov["spurious_rate"],
        }
        results.append(res_entry)
        print(f"Epoch {epoch_num:02d}: PQ={ov['pq']:.4f}, SQ={ov['sq']:.4f}, RQ={ov['rq']:.4f}, Dice={ov['mean_dice']:.4f}, TP={ov['tp']}, FP={ov['fp']}, FN={ov['fn']}")
        
    # Sort by highest strict tuning PQ, tie-break earliest epoch
    results_sorted = sorted(results, key=lambda x: (-x["pq"], x["epoch"]))
    
    out_path = Path("artifacts/reports/immutable_epochs_tuning_rescore.json")
    summary = {
        "ranking_rule": "highest strict tuning PQ, tie-break earliest epoch",
        "frozen_postprocess_params": frozen_params,
        "tuning_observations_count": 10,
        "selected_best_epoch": results_sorted[0],
        "all_epochs": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"Selection complete! Best immutable checkpoint: Epoch {results_sorted[0]['epoch']} ({results_sorted[0]['filename']})")
    print(f"Strict Tuning PQ: {results_sorted[0]['pq']:.4f} (TP: {results_sorted[0]['tp']}, FP: {results_sorted[0]['fp']}, FN: {results_sorted[0]['fn']}, Dice: {results_sorted[0]['mean_dice']:.4f})")
    print(f"Summary report written to: {out_path}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    rescore_epochs()
