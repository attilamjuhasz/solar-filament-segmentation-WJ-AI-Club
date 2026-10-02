"""Submission RLE (pycocotools compressed counts of a Fortran-order 2048x2048 mask) and CSV I/O."""
import csv
import os

import numpy as np
from pycocotools import mask as mu

H = W = 2048


def encode(mask):
    return mu.encode(np.asfortranarray((np.asarray(mask) > 0).astype(np.uint8)))["counts"].decode("ascii")


def to_rle_obj(counts):
    return {"size": [H, W], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def decode(counts):
    return mu.decode(to_rle_obj(counts)).astype(bool)


def _full(m):
    """Accept a full (2048, 2048) mask or a local crop (x0, y0, mask)."""
    if isinstance(m, tuple):
        x, y, loc = m
        full = np.zeros((H, W), bool)
        full[y:y + loc.shape[0], x:x + loc.shape[1]] = loc
        return full
    return m


def write_submission(path, preds):
    """preds: {stem: [full bool mask or (x0, y0, local mask), ...]}; masks must not overlap within a stem."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(preds):
            assert "_" not in stem and "-" not in stem, stem
            for j, m in enumerate(preds[stem], 1):
                w.writerow([f"{stem}_{j}", encode(_full(m))])
                n += 1
    return n


def validate_submission(path, test_stems):
    """Raise AssertionError on anything Kaggle would reject; return summary dict."""
    test_stems = set(test_stems)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["filament_id", "segmentation_rle"], rows[0]
    ids = [r[0] for r in rows[1:]]
    assert len(ids) == len(set(ids)), "duplicate filament_id"
    per_stem = {}
    for fid, counts in rows[1:]:
        stem, _, k = fid.rpartition("_")
        assert stem in test_stems, f"unknown stem {stem}"
        assert k.isdigit(), fid
        rle = to_rle_obj(counts)
        assert mu.area(rle) > 0, f"empty mask {fid}"
        assert list(mu.decode(rle).shape) == [H, W]
        assert encode(mu.decode(rle)) == counts, f"non-canonical RLE {fid}"
        per_stem.setdefault(stem, []).append(rle)
    for stem, rles in per_stem.items():
        union = mu.merge(rles, intersect=False)
        assert int(mu.area(union)) == sum(int(mu.area(r)) for r in rles), f"overlapping masks in {stem}"
    return dict(rows=len(ids), stems=len(per_stem), test_stems=len(test_stems))
