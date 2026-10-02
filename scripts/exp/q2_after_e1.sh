#!/usr/bin/env bash
# Queue Q2 (replaces Q1 after the score-optimizer showed E1 = S2 trained on TTA proposals loses -0.005):
#   E3a  S2 with ONLY the q target changed (averaged over annotator readings), plain proposals, 8 ep
#   SEED S2 v2 recipe with seed 1 -> run-to-run noise baseline (and a 2-seed q ensemble to test)
#   E2   S1 trained 40 epochs, then the S2 models re-scored on its proposals
# Eval-only steps never abort the queue; training steps do.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
V2=configs/assemble_v2.json
s2_eval_and_test() {  # $1 = S2 run, $2 = S1 run, $3 = tag
  $PY src/s2.py --name "$1" --s1 "$2" --props tta --predict val --tag "$3"
  $PY src/assemble.py eval --cands "runs/$1/cands_val_tta_last$3" --params $V2 || echo "EVAL FAILED"
  $PY src/s2.py --name "$1" --s1 "$2" --props tta --predict test --tag "$3"
}
# --- E3a
$PY src/s2.py --name s2_r34_qavg_plain --s1 s1_r34_f0 --props plain --epochs 8 --q-avg
s2_eval_and_test s2_r34_qavg_plain s1_r34_f0 ""
echo "E3a DONE"
# --- seed baseline
$PY src/s2.py --name s2_r34_seed1 --s1 s1_r34_f0 --props plain --epochs 8 --seed 1
s2_eval_and_test s2_r34_seed1 s1_r34_f0 ""
echo "SEED DONE"
# --- E2
S1=s1_r34_f0_e40
$PY src/s1.py --name $S1 --val-fold 0 --epochs 40 --eval-every 4
$PY src/s1.py --name $S1 --predict all --tta
$PY src/tune_s1.py --run $S1 --probs probs_tta --eval configs/s1_postprocess.json || echo "EVAL FAILED"
$PY src/s1.py --name $S1 --predict train                    # plain maps -> S2 training proposals later
$PY src/proposals.py --run $S1 --probs probs_plain
$PY src/proposals.py --run $S1 --probs probs_tta
for S2 in s2_r34 s2_r34_qavg_plain s2_r34_seed1; do
  [ -f runs/$S2/ep8.pt ] && s2_eval_and_test $S2 $S1 _e40
done
echo "E2 DONE"
