#!/usr/bin/env bash
# Q4 (after E4 = Q3 step D, and P2): bigger S1 ensembles, scored with S2 v2 and the v2+seed1 q-ensemble.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
V2=configs/assemble_v2.json
ev()  { echo "== EVAL $*"; $PY src/assemble.py eval --params $V2 "$@" || echo "EVAL FAILED"; }
sub() { echo "== SUBMIT $1"; shift; $PY src/assemble.py submit --params $V2 "$@" || echo "SUBMIT FAILED"; }
while pgrep -f "[q]3_after_q2_p1|[p]2_s1_seed2" > /dev/null; do sleep 60; done
ens() {  # $1 out name, $2 comma-separated S1 runs
  $PY src/ens_probs.py --runs "$2" --probs probs_tta --out "$1" || return 1
  $PY src/tune_s1.py --run "$1" --probs probs_tta --eval configs/s1_postprocess.json || echo "EVAL FAILED"
  $PY src/proposals.py --run "$1" --probs probs_tta || return 1
  for S2 in s2_r34 s2_r34_seed1; do
    $PY src/s2.py --name $S2 --s1 "$1" --props tta --predict val --tag "_$1" || return 1
    $PY src/s2.py --name $S2 --s1 "$1" --props tta --predict test --tag "_$1" || return 1
  done
  ev --cands "runs/s2_r34/cands_val_tta_last_$1"
  ev --cands "runs/s2_r34/cands_val_tta_last_$1" --extra-cands "runs/s2_r34_seed1/cands_val_tta_last_$1"
  sub "$1" --cands "runs/s2_r34/cands_test_tta_last_$1" --out "submissions/s2_r34_$1.csv"
  sub "$1 qens" --cands "runs/s2_r34/cands_test_tta_last_$1" --extra-cands "runs/s2_r34_seed1/cands_test_tta_last_$1" \
      --out "submissions/qens_$1.csv"
}
S1A=s1_r34_f0_e40; S1B=s1_r34_f0_e40_s1; S1C=s1_r34_f0_e40_s2; S1H=s1_r34_f0_1536
[ -f runs/$S1C/best.pt ] && { ens ens3_e40 "$S1A,$S1B,$S1C" || echo "ENS3 FAILED"; }
[ -d runs/$S1H/probs_tta ] && { ens ens2_1536 "$S1A,$S1B,$S1H" || echo "ENS+1536 FAILED"; }
[ -f runs/$S1C/best.pt ] && [ -d runs/$S1H/probs_tta ] && { ens ens4_1536 "$S1A,$S1B,$S1C,$S1H" || echo "ENS4 FAILED"; }
echo "Q4 DONE"
