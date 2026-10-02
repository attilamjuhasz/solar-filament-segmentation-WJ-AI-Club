#!/usr/bin/env bash
# Queue Q1 (runs after E1): E2 = S1 trained 40 epochs (it was still improving at 16), then
# E3 = S2 with the q label averaged over all annotator readings, trained on E2's TTA proposals.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
S1=s1_r34_f0_e40
# --- finish E1 if its 2 h background slot ran out before the predictions
n_val() { ls runs/s2_r34_tta/cands_val_tta_last 2>/dev/null | grep -c pkl || true; }
n_test() { ls runs/s2_r34_tta/cands_test_tta_last 2>/dev/null | grep -c pkl || true; }
if [ -f runs/s2_r34_tta/last.pt ] && [ "$(n_val)" -lt 141 ]; then
  $PY src/s2.py --name s2_r34_tta --s1 s1_r34_f0 --props tta --predict val
  $PY src/assemble.py eval --cands runs/s2_r34_tta/cands_val_tta_last --params configs/assemble_v2.json
fi
if [ -f runs/s2_r34_tta/last.pt ] && [ "$(n_test)" -lt 180 ]; then
  $PY src/s2.py --name s2_r34_tta --s1 s1_r34_f0 --props tta --predict test
fi
echo "E1 CHECKED"
# --- E2
$PY src/s1.py --name $S1 --val-fold 0 --epochs 40 --eval-every 4
$PY src/s1.py --name $S1 --predict all --tta
$PY src/tune_s1.py --run $S1 --probs probs_tta --eval configs/s1_postprocess.json          # S1-only, fixed params
$PY src/s1.py --name $S1 --predict train --tta
$PY src/proposals.py --run $S1 --probs probs_tta
for S2 in s2_r34 s2_r34_tta; do                                                          # existing S2s on E2 proposals
  [ -f runs/$S2/last.pt ] || continue
  $PY src/s2.py --name $S2 --s1 $S1 --props tta --predict val --tag _e40
  $PY src/assemble.py eval --cands runs/$S2/cands_val_tta_last_e40 --params configs/assemble_v2.json
done
echo "E2 DONE"
# --- E3
$PY src/s2.py --name s2_r34_qavg --s1 $S1 --props tta --epochs 16 --q-avg
$PY src/s2.py --name s2_r34_qavg --s1 $S1 --props tta --predict val
$PY src/assemble.py eval --cands runs/s2_r34_qavg/cands_val_tta_last --params configs/assemble_v2.json
echo "E3 DONE"
