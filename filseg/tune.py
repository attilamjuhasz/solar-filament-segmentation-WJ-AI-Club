"""Tune post-processing parameters for PQ on the validation split.

Predicts each validation image once, then tries parameter values one at a
time (coordinate search), keeping whichever raises PQ.

    python -m filseg.tune --data /path/to/MAGFiLO_1.0_Kaggle_2026 --weights weights/model.pt
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .data import DataPaths, find_data_root, index_images, load_annotations
from .dataset import FilamentDataset
from .metrics import format_summary, summarize
from .model import load_checkpoint
from .postprocess import PostParams
from .validate import predict_val_probs, score_probs

SEARCH_SPACE = {
    "thr_high": [0.3, 0.4, 0.5, 0.6, 0.7],
    "thr_low": [0.1, 0.2, 0.3, 0.4, 0.5],
    "min_area": [0, 25, 50, 100, 200],
    "merge_dist": [0, 6, 12, 20, 30, 45],
    "close_radius": [0, 2, 3, 5, 8],
    "min_instance_area": [100, 200, 300, 500, 800, 1200],
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="weights/model.pt")
    p.add_argument("--out", default="weights/postprocess.json")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--tta", action="store_true", help="flip test-time augmentation (match predict.py)")
    p.add_argument("--cache-dir", default="cache")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.weights, device)
    paths = DataPaths(find_data_root(args.data))
    val_stems = set(meta["val_stems"])
    val_items = [i for i in load_annotations(paths.train_json) if i.stem in val_stems]
    val_ds = FilamentDataset(val_items, index_images(paths.train_images), meta["size"], None, False, args.cache_dir)

    print(f"predicting {len(val_stems)} validation images...")
    probs = predict_val_probs(model, val_ds, device, tta=args.tta)

    def evaluate(params: PostParams) -> float:
        return summarize(score_probs(probs, val_items, params))["pq"]

    best = PostParams.load(args.out) if Path(args.out).exists() else PostParams()
    best_pq = evaluate(best)
    print(f"start: PQ {best_pq:.4f} {best}")
    for rnd in range(args.rounds):
        for name, values in SEARCH_SPACE.items():
            for value in values:
                cand = replace(best, **{name: value})
                if cand.thr_low > cand.thr_high or cand == best:
                    continue
                pq = evaluate(cand)
                if pq > best_pq:
                    best, best_pq = cand, pq
                    print(f"  round {rnd + 1}: {name}={value} -> PQ {best_pq:.4f}")

    best.save(args.out)
    print(f"best: {best}\n{format_summary(summarize(score_probs(probs, val_items, best)))}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
