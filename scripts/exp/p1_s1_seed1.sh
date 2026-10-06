#!/usr/bin/env bash
# P1 (parallel to Q2): second S1 at 40 epochs with seed 1 -> 2-model S1 ensemble with Q2's E2 (seed 0).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
S1=s1_r34_f0_e40_s1
$PY src/s1.py --name $S1 --val-fold 0 --epochs 40 --eval-every 4 --seed 1
$PY src/s1.py --name $S1 --predict all --tta
$PY src/tune_s1.py --run $S1 --probs probs_tta --eval configs/s1_postprocess.json || echo "EVAL FAILED"
$PY src/s1.py --name $S1 --predict train
echo "P1 DONE"
