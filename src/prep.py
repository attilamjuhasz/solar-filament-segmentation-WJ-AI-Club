"""Rasterize COCO polygon annotations into per-annotator instance maps and build a fold table.

Outputs (under data/cache/):
  inst/<image_id>.png   uint16 instance map at native 2048x2048 (0 = background, k = k-th annotation)
  meta.csv              one row per annotation version: image_id, file_name, year, date, n_inst, fold
"""
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
DATA = os.path.join(ROOT, "MAGFiLO_1.0_Kaggle_2026")
ANN = os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CACHE = os.path.join(ROOT, "cache")
N_FOLDS = 5


def poly_to_mask(seg, h, w):
    # pycocotools rasterization == what the official metric uses for GT
    return mu.decode(mu.merge(mu.frPyObjects(seg, h, w)))


def main():
    os.makedirs(os.path.join(CACHE, "inst"), exist_ok=True)
    d = json.load(open(ANN))
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)

    rows = []
    for im in d["images"]:
        h, w = im["height"], im["width"]
        inst = np.zeros((h, w), np.uint16)
        # paint larger instances first so small ones stay visible where they overlap
        items = sorted(anns[im["id"]], key=lambda a: -a["area"])
        for k, a in enumerate(items, 1):
            inst[poly_to_mask(a["segmentation"], h, w) > 0] = k
        cv2.imwrite(os.path.join(CACHE, "inst", im["id"] + ".png"), inst)
        rows.append(dict(image_id=im["id"], file_name=im["file_name"],
                         date=im["date_captured"][:10], year=int(im["date_captured"][:4]),
                         n_inst=len(items)))

    meta = pd.DataFrame(rows)
    # Group folds by observation date so same-day frames (and all annotator versions
    # of one file) never straddle train/val. Round-robin over dates sorted within year
    # keeps every fold spread across the solar cycle.
    dates = meta[["date", "year"]].drop_duplicates().sort_values(["year", "date"]).reset_index(drop=True)
    dates["fold"] = np.arange(len(dates)) % N_FOLDS
    meta = meta.merge(dates[["date", "fold"]], on="date")
    meta.to_csv(os.path.join(CACHE, "meta.csv"), index=False)
    print(meta.groupby("fold").agg(files=("file_name", "nunique"), versions=("image_id", "size"),
                                   inst=("n_inst", "sum")))


if __name__ == "__main__":
    main()
