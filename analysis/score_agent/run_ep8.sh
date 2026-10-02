#!/bin/bash
D=/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle
cd /private/tmp/claude-503/-Volumes-Zaids-Nvme-zaidzamani-Desktop-Projects-temp-kaggle/91b796c3-671c-43b1-b3fa-06921ba1f96e/scratchpad/score
until [ "$(ls $D/runs/s2_r34/cands_val_tta_last/*.pkl 2>/dev/null | wc -l | tr -d ' ')" -ge 141 ]; do sleep 10; done
sleep 20  # let the last pickle finish writing
export CANDS=runs/s2_r34/cands_val_tta_last TAG=ep8 PYTHONPATH=$D/src
PY="nice -n 10 $D/.venv/bin/python"
echo "== q1"; $PY q1_calib.py > q1_ep8.log 2>&1; echo "q1 rc=$?"
echo "== feats"; $PY feats.py > feats_ep8.log 2>&1; echo "feats rc=$?"
echo "== q3 M0 M1"; $PY q3.py M0 M1 > q3_ep8.log 2>&1; echo "q3 rc=$?"
echo "== rescore"; $PY rescore.py > rescore_ep8.log 2>&1; echo "rescore rc=$?"
echo "== rescore2"; $PY rescore2.py > rescore2_ep8.log 2>&1; echo "rescore2 rc=$?"
echo "== final"; $PY final.py > final_ep8.log 2>&1; echo "final rc=$?"
echo ALLDONE
