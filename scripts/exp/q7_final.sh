#!/usr/bin/env bash
# Q7 (after Q6): FINAL "aggressive" submission = mean of ALL S1 members (fold-0 seeds, folds 1-4 last.pt, all-data seeds)
# -> proposals -> S2 v2 -> assemble_v2 (lam .225), plus a guard check against the robust pick (s2_r34_ens2.csv).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONWARNINGS=ignore
M="s1_r34_f0_e40,s1_r34_f0_e40_s1"
for k in 1 2 3 4; do [ -d runs/s1_r34_f${k}_e40_last/probs_tta ] && M="$M,s1_r34_f${k}_e40_last"; done
for sd in 0 1 2; do [ -d runs/s1_r34_all_e40_s$sd/probs_tta ] && M="$M,s1_r34_all_e40_s$sd"; done
echo "members: $M"
$PY src/ens_probs.py --runs "$M" --probs probs_tta --out ens_final \
  && $PY src/proposals.py --run ens_final --probs probs_tta \
  && $PY src/s2.py --name s2_r34 --s1 ens_final --props tta --predict test --tag _final \
  && $PY src/assemble.py submit --cands runs/s2_r34/cands_test_tta_last_final --params configs/assemble_v2.json \
       --out submissions/final_aggressive.csv \
  && $PY scripts/exp/compare_subs.py submissions/s2_r34_ens2.csv submissions/final_aggressive.csv \
  || echo "FINAL FAILED"
echo "Q7 DONE"
