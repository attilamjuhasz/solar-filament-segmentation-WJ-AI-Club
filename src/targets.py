"""Stage-1 targets at 1024 (area-downsampled from the pycocotools 2048 rasters).

  data/cache/fg1024/<reading_id>.npy   uint8 soft fg of one annotator (0..255)
  data/cache/un1024/<stem>.npy         uint8 soft fg of the union of all annotators
  data/cache/boxes.json                {stem: [[cx, cy, w, h] native px of every GT instance, all readings]}
"""
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, DATA, load_inst, load_meta  # noqa: E402


def down(m):
    return np.round(cv2.resize(m.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)


def main():
    for sub in ("fg1024", "un1024"):
        os.makedirs(os.path.join(CACHE, sub), exist_ok=True)
    meta = load_meta()
    for fname, g in tqdm(meta.groupby("file_name")):
        union = np.zeros((2048, 2048), bool)
        for rid in g.image_id:
            fg = load_inst(rid) > 0
            union |= fg
            np.save(os.path.join(CACHE, "fg1024", rid + ".npy"), down(fg))
        np.save(os.path.join(CACHE, "un1024", fname[:-5] + ".npy"), down(union))

    d = json.load(open(os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")))
    stem_of = {im["id"]: im["file_name"][:-5] for im in d["images"]}
    boxes = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.setdefault(stem_of[a["image_id"]], []).append([x + w / 2, y + h / 2, w, h])
    json.dump(boxes, open(os.path.join(CACHE, "boxes.json"), "w"))


if __name__ == "__main__":
    main()
