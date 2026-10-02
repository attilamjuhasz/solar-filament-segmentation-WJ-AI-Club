#!/usr/bin/env bash
# Queue Q1 (runs after E1): E2 = S1 trained 40 epochs (it was still improving at 16), then
# E3 = S2 with the q label averaged over all annotator readings, trained on E2's TTA proposals.
# Eval-only steps never abort the queue; training steps do.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
S1=s1_r34_f0_e40
n_pkl() { ls "$1" 2>/dev/null | grep -c pkl || true; }
# --- finish E1 if its 2 h background slot ran out before the predictions (only if S2 trained fully)
if [ -f runs/s2_r34_tta/ep8.pt ]; then
  if [ "$(n_pkl runs/s2_r34_tta/cands_val_tta_last)" -lt 141 ]; then
    $PY src/s2.py --name s2_r34_tta --s1 s1_r34_f0 --props tta --predict val
    $PY src/assemble.py eval --cands runs/s2_r34_tta/cands_val_tta_last --params configs/assemble_v2.json || echo "EVAL FAILED"
  fi
  if [ "$(n_pkl runs/s2_r34_tta/cands_test_tta_last)" -lt 180 ]; then
    $PY src/s2.py --name s2_r34_tta --s1 s1_r34_f0 --props tta --predict test
  fi
else
  echo "E1 S2 did not finish 8 epochs - skipping its predictions"
fi
echo "E1 CHECKED"
# --- E2
$PY src/s1.py --name $S1 --val-fold 0 --epochs 40 --eval-every 4
$PY src/s1.py --name $S1 --predict all --tta
$PY src/tune_s1.py --run $S1 --probs probs_tta --eval configs/s1_postprocess.json || echo "EVAL FAILED"   # S1-only, fixed params
$PY src/s1.py --name $S1 --predict train --tta
$PY src/proposals.py --run $S1 --probs probs_tta
for S2 in s2_r34 s2_r34_tta; do                                                          # existing S2s on E2 proposals
  [ -f runs/$S2/ep8.pt ] || continue
  $PY src/s2.py --name $S2 --s1 $S1 --props tta --predict val --tag _e40
  $PY src/assemble.py eval --cands runs/$S2/cands_val_tta_last_e40 --params configs/assemble_v2.json || echo "EVAL FAILED"
  $PY src/s2.py --name $S2 --s1 $S1 --props tta --predict test --tag _e40
done
echo "E2 DONE"
# --- E3
$PY src/s2.py --name s2_r34_qavg --s1 $S1 --props tta --epochs 16 --q-avg
$PY src/s2.py --name s2_r34_qavg --s1 $S1 --props tta --predict val
$PY src/assemble.py eval --cands runs/s2_r34_qavg/cands_val_tta_last --params configs/assemble_v2.json || echo "EVAL FAILED"
$PY src/s2.py --name s2_r34_qavg --s1 $S1 --props tta --predict test
echo "E3 DONE"
