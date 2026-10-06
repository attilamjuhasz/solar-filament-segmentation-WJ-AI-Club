#!/usr/bin/env bash
# Q3 (auto-starts after Q2; waits for P1 too). Writes a validated CSV for every candidate result into
# submissions/ and prints each val PQ, so the morning pick can be made from runs/q3.out + AGENTS.md.
#   A  submissions for Q2's S2 runs (E3a q-avg, SEED) and the v2+SEED q-ensemble
#   B  S1 2-seed ensemble (E2 seed0 + P1 seed1, 40 ep) -> proposals -> S2 v2 scoring
#   C  MaskQ S2 (mask-aware q head, q-loss x2, synthetic q weight .25, levels AP)
#   D  E4: S1 fine-tuned at 1536 from E2's best
# Eval/submit steps never abort the queue; training steps do.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
V2=configs/assemble_v2.json
ev()  { echo "== EVAL $*"; $PY src/assemble.py eval --params $V2 "$@" || echo "EVAL FAILED"; }
sub() { echo "== SUBMIT $1"; shift; $PY src/assemble.py submit --params $V2 "$@" || echo "SUBMIT FAILED"; }
s2_val_test() {  # $1 S2 run, $2 S1 run (proposals), $3 tag
  $PY src/s2.py --name "$1" --s1 "$2" --props tta --predict val --tag "$3" || return 1
  $PY src/s2.py --name "$1" --s1 "$2" --props tta --predict test --tag "$3" || return 1
  ev --cands "runs/$1/cands_val_tta_last$3"
  sub "$1$3" --cands "runs/$1/cands_test_tta_last$3" --out "submissions/$1$3.csv"
}
# --- A
for R in s2_r34_qavg_plain s2_r34_seed1; do
  [ -d runs/$R/cands_test_tta_last ] && sub $R --cands runs/$R/cands_test_tta_last --out submissions/$R.csv
done
if [ -d runs/s2_r34_seed1/cands_val_tta_last ]; then
  ev --cands runs/s2_r34/cands_val_tta_last --extra-cands runs/s2_r34_seed1/cands_val_tta_last
  sub qens_v2_seed1 --cands runs/s2_r34/cands_test_tta_last --extra-cands runs/s2_r34_seed1/cands_test_tta_last \
      --out submissions/qens_v2_seed1.csv
fi
echo "A DONE"
# --- wait for P1 (parallel S1 seed 1)
while pgrep -f "[p]1_s1_seed1.sh" > /dev/null; do sleep 60; done
# --- B
if [ -f runs/s1_r34_f0_e40/best.pt ] && [ -f runs/s1_r34_f0_e40_s1/best.pt ]; then
  $PY src/ens_probs.py --runs s1_r34_f0_e40,s1_r34_f0_e40_s1 --probs probs_tta --out s1_ens2_e40
  $PY src/tune_s1.py --run s1_ens2_e40 --probs probs_tta --eval configs/s1_postprocess.json || echo "EVAL FAILED"
  $PY src/proposals.py --run s1_ens2_e40 --probs probs_tta
  s2_val_test s2_r34 s1_ens2_e40 _ens2 || echo "S2 ON ENS FAILED"
fi
echo "B DONE"
# --- C
$PY src/s2.py --name s2_r34_mq --s1 s1_r34_f0 --props plain --epochs 8 --qhead mask --wq 2 --q-syn-w 0.25 --prop-levels AP \
  && s2_val_test s2_r34_mq s1_r34_f0 "" || echo "MASKQ FAILED"
echo "C DONE"
# --- D
if [ -f runs/s1_r34_f0_e40/best.pt ]; then
  $PY src/s1.py --name s1_r34_f0_1536 --val-fold 0 --res 1536 --init runs/s1_r34_f0_e40/best.pt --lr 1e-4 \
      --epochs 12 --eval-every 4 \
    && $PY src/s1.py --name s1_r34_f0_1536 --predict all --tta \
    && { $PY src/tune_s1.py --run s1_r34_f0_1536 --probs probs_tta --eval configs/s1_postprocess.json || true; } \
    && $PY src/proposals.py --run s1_r34_f0_1536 --probs probs_tta \
    && s2_val_test s2_r34 s1_r34_f0_1536 _1536 || echo "E4 FAILED"
fi
echo "Q3 DONE"
