"""Predict filaments for a folder of images and write the submission CSV.

    python -m filseg.predict --images /path/to/test/test_images --out submission.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .data import index_images, read_image
from .infer import predict_instances
from .model import load_checkpoint
from .postprocess import PostParams
from .submission import rows_for_image, write_submission


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of H-alpha images")
    p.add_argument("--weights", default="weights/model.pt")
    p.add_argument("--postprocess", default="weights/postprocess.json")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--no-tta", action="store_true")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.weights, device)
    params = PostParams.load(args.postprocess) if Path(args.postprocess).exists() else PostParams()
    print(f"device {device} | post-processing {params}")

    images = index_images(args.images)
    rows, empty, t0 = [], 0, time.time()
    for stem, path in tqdm(images.items()):
        labels = predict_instances(model, read_image(path), meta["size"], device, params, tta=not args.no_tta)
        image_rows = rows_for_image(stem, labels)
        empty += not image_rows
        rows.extend(image_rows)

    write_submission(rows, args.out)
    elapsed = time.time() - t0
    print(
        f"wrote {len(rows)} filaments for {len(images)} images to {args.out} "
        f"({empty} images with none) in {elapsed:.0f}s ({elapsed / max(1, len(images)):.2f}s/image)"
    )


if __name__ == "__main__":
    main()
