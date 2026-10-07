"""Assemble out-of-fold S1 probability maps: each stem's map comes from the fold model that never trained on it.

python src/oof_probs.py --runs 0:s1_r34_f0_e40,1:s1_r34_f1_e40,... --out s1_oof   -> runs/s1_oof/probs_tta (symlinks)
Test stems are taken as the mean over all fold models (same as src/ens_probs.py).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import RUNS, load_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="fold:run,... e.g. 0:s1_r34_f0_e40,1:s1_r34_f1_e40")
    ap.add_argument("--probs", default="probs_tta")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {int(k): v for k, v in (x.split(":") for x in a.runs.split(","))}
    for k, r in runs.items():  # each fold map must come from a model that held that fold out
        vf = json.load(open(os.path.join(RUNS, r, "args.json")))["val_fold"]
        assert vf == k, f"{r} was trained with --val-fold {vf}, not {k}"
    meta = load_meta()
    fold_of = dict(zip(meta.file_name.str[:-5], meta.fold))
    odir = os.path.join(RUNS, a.out, a.probs)
    os.makedirs(odir, exist_ok=True)
    n_oof = 0
    for stem, k in fold_of.items():
        if k not in runs:
            continue
        src = os.path.abspath(os.path.join(RUNS, runs[k], a.probs, stem + ".npy"))
        assert os.path.exists(src), f"{src} missing (did that fold predict with --val-fold {k}?)"
        dst = os.path.join(odir, stem + ".npy")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        n_oof += 1
    # test stems: mean over the fold models
    test = set.intersection(*[{f[:-4] for f in os.listdir(os.path.join(RUNS, r, a.probs)) if f.endswith(".npy")}
                              for r in runs.values()]) - set(fold_of)
    assert len(test) == 180, f"expected 180 test stems common to all fold runs, got {len(test)}"
    for s in sorted(test):
        maps = [np.load(os.path.join(RUNS, r, a.probs, s + ".npy")).astype(np.float32) for r in runs.values()]
        np.save(os.path.join(odir, s + ".npy"), np.round(np.mean(maps, 0)).astype(np.uint8))
    print(f"{n_oof} out-of-fold stems linked, {len(test)} test stems averaged -> {odir}")


if __name__ == "__main__":
    main()
