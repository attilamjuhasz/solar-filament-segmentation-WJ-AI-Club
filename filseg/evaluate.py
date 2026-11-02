"""Score a submission CSV against the training annotations.

    # Check the CSV writer/reader and metric: ground truth scored against itself should give PQ ~1
    python -m filseg.evaluate --data /path/to/MAGFiLO_1.0_Kaggle_2026 --gt-as-pred

    # Score predictions made on the validation images
    python -m filseg.evaluate --data ... --csv val_submission.csv --weights weights/model.pt
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from .data import DataPaths, find_data_root, group_by_stem, load_annotations
from .metrics import format_summary, score_image, summarize
from .submission import read_submission, rows_for_image, write_submission


def gt_submission(items, path: Path) -> None:
    """Write the first annotator's filaments of each image as if they were predictions.
    (Where two GT filaments overlap, the shared pixels go to the first one.)"""
    rows = []
    for stem, group in group_by_stem(items).items():
        first = group[0]
        labels = np.zeros((first.height, first.width), np.int32)
        for k, rle in enumerate(first.rles(), start=1):
            labels[mask_utils.decode(rle).astype(bool) & (labels == 0)] = k
        rows.extend(rows_for_image(stem, labels))
    write_submission(rows, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--csv", help="submission CSV to score")
    p.add_argument("--gt-as-pred", action="store_true", help="score the ground truth against itself")
    p.add_argument("--weights", help="if given, only score the checkpoint's validation images")
    args = p.parse_args(argv)

    items = load_annotations(DataPaths(find_data_root(args.data)).train_json)
    if args.weights:
        from .model import load_checkpoint

        _, meta = load_checkpoint(args.weights)
        items = [i for i in items if i.stem in set(meta["val_stems"])]

    if args.gt_as_pred:
        csv = Path(tempfile.mkdtemp()) / "gt.csv"
        gt_submission(items, csv)
    elif args.csv:
        csv = Path(args.csv)
    else:
        p.error("pass --csv or --gt-as-pred")

    preds = read_submission(csv)
    groups = group_by_stem(items)
    same, other = [], []
    for stem, group in groups.items():
        for n, item in enumerate(group):
            score = score_image(item.image_id, item.rles(), preds.get(stem, []))
            (other if args.gt_as_pred and n > 0 else same).append(score)

    if args.gt_as_pred:
        print("Same annotator (expect PQ ~1.0; slightly lower only where GT filaments overlap; Dice must be 1.0):")
        print("  " + format_summary(summarize(same)))
        if other:
            print("Other annotators of the same image (how much humans agree; a rough ceiling):")
            print("  " + format_summary(summarize(other)))
    else:
        print(format_summary(summarize(same)))


if __name__ == "__main__":
    main()
