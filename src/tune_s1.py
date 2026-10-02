"""Tune the S1-only post-processing (fallback path) on validation probs with the fast PQ.

python src/tune_s1.py --run s1_r34_f0 --probs probs_tta
python src/tune_s1.py --run s1_r34_f0 --probs probs_tta --submit submissions/s1_only.csv   (uses tuned params)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from assemble import FastGT  # noqa: E402
from common import RUNS, TEST_IMG  # noqa: E402
from proposals import load_probs  # noqa: E402
from s1 import postprocess_s1, split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probs", default="probs_tta")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--submit", default=None)
    ap.add_argument("--eval", default=None, help="score fixed params (json) on val instead of tuning")
    a = ap.parse_args()
    _, va, _ = split(a.val_fold)
    pdir = os.path.join(RUNS, a.run, a.probs)
    if a.submit:
        return submit(pdir, a.submit)
    probs = {s: load_probs(os.path.join(pdir, s + ".npy")) for s in va}
    gt = FastGT(va)
    if a.eval:
        P = json.load(open(a.eval))
        P = {k: P[k] for k in ("t_hi", "t_lo", "gap", "min_area", "head")}
        print("EVAL", gt.pq({s: postprocess_s1(probs[s], s, **P) for s in va}), P)
        return

    def run(P):
        return gt.pq({s: postprocess_s1(probs[s], s, **P) for s in va})

    P = dict(t_hi=0.5, t_lo=0.35, gap=0, min_area=300, head=0)
    best, info = run(P)
    print("start", round(best, 4), info, P, flush=True)
    grid = dict(head=[0, 1], t_hi=[0.4, 0.5, 0.6, 0.7], t_lo=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                gap=[0, 4, 8, 12], min_area=[100, 200, 300, 400, 600, 800])
    for _ in range(2):
        for k, vals in grid.items():
            for v in vals:
                if v == P[k] or (k == "t_lo" and v > P["t_hi"]) or (k == "t_hi" and v < P["t_lo"]):
                    continue
                Q = dict(P, **{k: v})
                s, inf = run(Q)
                if s > best + 1e-5:
                    best, P, info = s, Q, inf
                    print(f"  {k}={v} -> {best:.4f} {info}", flush=True)
    print("BEST", round(best, 4), info, P)
    json.dump(dict(P, val_pq=best, **info), open(os.path.join(pdir, "s1_best_params.json"), "w"), indent=1)


def submit(pdir, out):
    from rle import validate_submission, write_submission
    P = json.load(open(os.path.join(pdir, "s1_best_params.json")))
    P = {k: P[k] for k in ("t_hi", "t_lo", "gap", "min_area", "head")}
    test = sorted(f.rsplit(".", 1)[0] for f in os.listdir(TEST_IMG) if f.endswith(".jpeg"))
    preds = {}
    for s in test:
        masks = postprocess_s1(load_probs(os.path.join(pdir, s + ".npy")), s, **P)
        if masks:
            preds[s] = masks
    n = write_submission(out, preds)
    print("wrote", n, "rows ->", out, validate_submission(out, test))


if __name__ == "__main__":
    main()
