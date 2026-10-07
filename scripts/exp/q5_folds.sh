#!/usr/bin/env bash
# Q5 (after Q3/E4): M1 fixed-epoch check (last.pt vs best.pt), then M2 = S1 on folds 1-4 (OOF over all 707 stems).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
V2=configs/assemble_v2.json
ev() { echo "== EVAL $*"; $PY src/assemble.py eval --params $V2 "$@" || echo "EVAL FAILED"; }
# --- M1: do the 40-epoch runs' LAST checkpoints match their val-selected BEST ones? (decides fixed-epoch training)
for R in s1_r34_f0_e40 s1_r34_f0_e40_s1; do
  mkdir -p runs/${R}_last && cp runs/$R/last.pt runs/${R}_last/best.pt && cp runs/$R/args.json runs/${R}_last/
  $PY src/s1.py --name ${R}_last --predict all --tta || echo "PRED FAILED $R"
done
for E in "ens2_last:s1_r34_f0_e40_last,s1_r34_f0_e40_s1_last" \
         "ens4_bl:s1_r34_f0_e40,s1_r34_f0_e40_s1,s1_r34_f0_e40_last,s1_r34_f0_e40_s1_last"; do
  N=${E%%:*}; RUNS_=${E#*:}
  $PY src/ens_probs.py --runs "$RUNS_" --probs probs_tta --out $N && $PY src/proposals.py --run $N --probs probs_tta \
    && $PY src/s2.py --name s2_r34 --s1 $N --props tta --predict val --tag _$N && ev --cands runs/s2_r34/cands_val_tta_last_$N
done
echo "M1 DONE"
# --- M2: folds 1-4 (predict `all` = that fold's held-out stems + test; NOTE --val-fold must be passed to predict)
for k in 1 2 3 4; do
  $PY src/s1.py --name s1_r34_f${k}_e40 --val-fold $k --epochs 40 --eval-every 4 --seed 0 \
    && $PY src/s1.py --name s1_r34_f${k}_e40 --predict all --tta --val-fold $k || echo "FOLD $k FAILED"
  echo "FOLD $k DONE"
done
echo "Q5 DONE"
