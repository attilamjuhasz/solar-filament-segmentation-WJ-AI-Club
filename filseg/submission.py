"""Read and write the competition CSV: one row per filament,
``filament_id`` = ``<image stem>_<n>``, ``segmentation_rle`` = RLE counts (no quotes)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .rle import encode, to_rle

HEADER = "filament_id,segmentation_rle"


def rows_for_image(stem: str, labels: np.ndarray) -> list[tuple[str, str]]:
    rows = []
    for k in range(1, int(labels.max()) + 1):
        mask = labels == k
        if mask.any():
            rows.append((f"{stem}_{len(rows) + 1}", encode(mask)))
    return rows


def write_submission(rows: list[tuple[str, str]], path: str | Path) -> None:
    # RLE counts use characters '0'..'o', which never include a comma or a quote,
    # so plain lines are a valid CSV and match the "no quotations" requirement.
    with open(path, "w", newline="\n") as f:
        f.write(HEADER + "\n")
        for filament_id, counts in rows:
            f.write(f"{filament_id},{counts}\n")


def read_submission(path: str | Path, height: int = 2048, width: int = 2048) -> dict[str, list[dict]]:
    """CSV -> {image stem: [COCO RLE, ...]}"""
    by_image: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        header = f.readline().strip()
        if header != HEADER:
            raise ValueError(f"unexpected header {header!r}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            filament_id, counts = line.split(",", 1)
            stem = filament_id.rsplit("_", 1)[0]
            by_image[stem].append(to_rle(counts.strip().strip('"'), height, width))
    return by_image
