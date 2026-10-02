"""Sanity checks: perfect prediction = 1, annotator ceiling in the reported range, RLE round trip + validator."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import DATA, load_meta  # noqa: E402
from metric import evaluate, load_gt_rles, reading_stats  # noqa: E402
from rle import decode, encode, validate_submission, write_submission  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

ANN = os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")


def main():
    meta = load_meta()
    val = set(meta[meta.fold == 0].file_name.str[:-5])
    gt = load_gt_rles(ANN, val)
    # 1) each reading scored against itself -> PQ 1
    for rid, (stem, g) in list(gt.items())[:20]:
        s, tp, fp, fn = reading_stats(g, g)
        assert tp == len(g) and fp == fn == 0 and abs(s - tp) < 1e-9, rid
    # 2) IoU just below 0.5 must not match
    a = np.zeros((2048, 2048), np.uint8); a[100:200, 100:200] = 1
    b = np.zeros((2048, 2048), np.uint8); b[100:200, 151:251] = 1  # IoU = 49*100/(151*100)
    s, tp, fp, fn = reading_stats([mu.encode(np.asfortranarray(a))], [mu.encode(np.asfortranarray(b))])
    assert (tp, fp, fn) == (0, 1, 1)
    # 3) annotator-as-prediction ceiling: first reading of each stem predicts against all readings
    first = {}
    for rid, (stem, g) in sorted(gt.items()):
        first.setdefault(stem, g)
    pq, info = evaluate(first, gt)
    print(f"ceiling (one annotator vs all readings, val fold): PQ={pq:.3f}", {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()})
    # 4) RLE round trip + validator incl. overlap rejection
    m1 = decode(encode(a)); assert (m1 == a.astype(bool)).all()
    path = "/tmp/_sub_test.csv" if not os.environ.get("SCRATCH") else os.path.join(os.environ["SCRATCH"], "sub_test.csv")
    write_submission(path, {"stemA": [a.astype(bool)], "stemB": [b.astype(bool)]})
    print("validator ok:", validate_submission(path, ["stemA", "stemB", "stemC"]))
    write_submission(path, {"stemA": [a.astype(bool), b.astype(bool)]})
    try:
        validate_submission(path, ["stemA"]); raise SystemExit("overlap not caught!")
    except AssertionError as e:
        print("overlap correctly rejected:", e)
    print("ALL OK")


if __name__ == "__main__":
    main()
