#!/usr/bin/env bash
# P2: third S1 at 40 epochs (seed 2) for a 3-model S1 ensemble.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
S1=s1_r34_f0_e40_s2
$PY src/s1.py --name $S1 --val-fold 0 --epochs 40 --eval-every 4 --seed 2
$PY src/s1.py --name $S1 --predict all --tta
$PY src/tune_s1.py --run $S1 --probs probs_tta --eval configs/s1_postprocess.json || echo "EVAL FAILED"
$PY src/s1.py --name $S1 --predict train
echo "P2 DONE"
