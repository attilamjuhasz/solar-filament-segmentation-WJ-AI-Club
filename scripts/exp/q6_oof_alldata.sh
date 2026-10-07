#!/usr/bin/env bash
# Q6 (after Q5 folds): M3 = S2 trained on out-of-fold S1 proposals; M4 = 3 all-data S1 seeds (fixed 40 ep, last.pt).
# ONE S1 training at a time (16 GB M1 swaps with two).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
V2=configs/assemble_v2.json
ev()  { echo "== EVAL $*"; $PY src/assemble.py eval --params $V2 "$@" || echo "EVAL FAILED"; }
# --- OOF S1 maps over all 707 stems (fold 0 = the seed-0 40-ep run) + CV number
FOLDS="0:s1_r34_f0_e40,1:s1_r34_f1_e40,2:s1_r34_f2_e40,3:s1_r34_f3_e40,4:s1_r34_f4_e40"
$PY src/oof_probs.py --runs $FOLDS --out s1_oof || echo "OOF FAILED"
for k in 0 1 2 3 4; do $PY src/tune_s1.py --run s1_oof --probs probs_tta --val-fold $k --eval configs/s1_postprocess.json \
  | sed "s/^EVAL/EVAL fold$k/" || true; done
$PY src/proposals.py --run s1_oof --probs probs_tta
# --- M3: S2 on OOF proposals, scored on fold-0 val with the ens2 proposals (paired vs v2 .4616)
$PY src/s2.py --name s2_oof --s1 s1_oof --props tta --epochs 8 \
  && $PY src/s2.py --name s2_oof --s1 s1_ens2_e40 --props tta --predict val --tag _ens2 \
  && ev --cands runs/s2_oof/cands_val_tta_last_ens2 \
  && ev --cands runs/s2_r34/cands_val_tta_last_ens2 --extra-cands runs/s2_oof/cands_val_tta_last_ens2 \
  || echo "M3 FAILED"
echo "M3 DONE"
# --- M4: all-data S1 seeds (no val; keep last.pt as best.pt), then test maps
for sd in 0 1 2; do
  R=s1_r34_all_e40_s$sd
  $PY src/s1.py --name $R --val-fold -1 --epochs 40 --seed $sd \
    && cp runs/$R/last.pt runs/$R/best.pt \
    && $PY src/s1.py --name $R --predict test --tta --val-fold -1 || echo "ALLDATA $sd FAILED"
  echo "ALLDATA $sd DONE"
done
echo "Q6 DONE"
