#!/usr/bin/env bash
# E1: retrain stage 2 on the same kind of proposals it sees at test time (S1 D4-TTA maps on the
# training stems), instead of plain-inference proposals. Targets the 216 low-q scoring misses.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
$PY src/s1.py --name s1_r34_f0 --predict train --tta                 # train-stem TTA maps -> probs_tta/
$PY src/proposals.py --run s1_r34_f0 --probs probs_tta                # props_tta/ for all 707 stems
$PY src/s2.py --name s2_r34_tta --s1 s1_r34_f0 --props tta --epochs 8
$PY src/s2.py --name s2_r34_tta --s1 s1_r34_f0 --props tta --predict val
$PY src/s2.py --name s2_r34_tta --s1 s1_r34_f0 --props tta --predict test
$PY src/assemble.py eval --cands runs/s2_r34_tta/cands_val_tta_last --params configs/assemble_v2.json
echo "E1 DONE"
