import os, sys, pickle, numpy as np, torch
sys.path.insert(0, "src")
import s2
torch.set_num_threads(8)
dev = torch.device("cpu")
model = s2.build_model(pretrained=False)
model.load_state_dict(torch.load("runs/s2_r34/last.pt", map_location="cpu"))
for split, d in [("val", "runs/s2_r34/cands_val_tta_last"), ("test", "runs/s2_r34/cands_test_tta_last")]:
    stems = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".pkl"))
    st = stems[len(stems) // 3]
    props = pickle.load(open(f"runs/s1_r34_f0/props_tta/{st}.pkl", "rb"))[:12]
    stored = [c for c in pickle.load(open(f"{d}/{st}.pkl", "rb")) if c["idx"] < 12]
    for tta in (True, False):
        new = s2.refine(model, st, props, dev, tta=tta)
        dq = max(abs(a["q"] - b["q"]) for a, b in zip(new, stored)) if len(new) == len(stored) else None
        same_idx = [c["idx"] for c in new] == [c["idx"] for c in stored]
        dsoft = max(int(np.abs(a["soft"].astype(int) - b["soft"].astype(int)).max()) if a["soft"].shape == b["soft"].shape else 999
                    for a, b in zip(new, stored)) if same_idx else None
        print(split, st, "tta", tta, "n", len(new), len(stored), "same idx", same_idx, "max|dq|", dq, "max|dsoft|", dsoft)
