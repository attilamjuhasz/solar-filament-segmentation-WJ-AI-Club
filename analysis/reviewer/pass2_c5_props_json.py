import os, sys
sys.path.insert(0, "src")
from proposals import load_probs
f = "s1_best_params.json"
print("stem would be:", repr(f[:-4]))
try:
    load_probs(os.path.join("runs/s1_r34_f0/probs_tta", f)); print("loaded?!")
except Exception as e:
    print("CRASH:", type(e).__name__, str(e)[:120])
