#!/usr/bin/env bash
# Full pipeline: data cache -> stage 1 (blob finder) -> proposals -> stage 2 (blob refiner) -> assembly -> submission.
# Usage: bash scripts/run_pipeline.sh   (expects data/MAGFiLO_1.0_Kaggle_2026 downloaded)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONWARNINGS=ignore
S1=${S1:-s1_r34_f0}
S2=${S2:-s2_r34}

# 0. caches (rasterized GT, folds, images, disk geometry, stage-1 targets)
$PY src/prep.py
$PY src/cache_inst.py
$PY src/cache_images.py
$PY src/disk.py
$PY src/targets.py

# 1. stage 1: full-disk UNet, validation fold 0
$PY src/s1.py --name "$S1" --val-fold 0 --epochs 16
$PY src/s1.py --name "$S1" --predict train            # in-sample maps -> stage-2 training proposals
$PY src/s1.py --name "$S1" --predict all --tta        # val + test maps with D4 TTA
$PY src/tune_s1.py --run "$S1" --probs probs_tta      # S1-only fallback params

# 2. proposals (blobs)
$PY src/proposals.py --run "$S1" --probs probs_plain
$PY src/proposals.py --run "$S1" --probs probs_tta

# 3. stage 2: per-blob refiner
$PY src/s2.py --name "$S2" --s1 "$S1" --props plain --epochs 8
$PY src/s2.py --name "$S2" --s1 "$S1" --props tta --predict val
$PY src/s2.py --name "$S2" --s1 "$S1" --props tta --predict test

# 4. assembly (fixed config, lam = PQ/2): score on val, then write + validate the test submission
$PY src/assemble.py eval --cands "runs/$S2/cands_val_tta_last" --params configs/assemble_v2.json
$PY src/assemble.py submit --cands "runs/$S2/cands_test_tta_last" \
    --params configs/assemble_v2.json --out submissions/submission.csv
