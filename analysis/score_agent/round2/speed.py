import os, sys, time, torch
sys.path.insert(0, "/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle/src")
torch.set_num_threads(4)
from s2 import build_model
m = build_model(pretrained=False).eval()
m.load_state_dict(torch.load("/Volumes/Zaids_Nvme/zaidzamani/Desktop/Projects/temp-kaggle/runs/s2_r34/last.pt", map_location="cpu"))
x = torch.randn(32, 4, 256, 256)
with torch.no_grad():
    m(x)
    t = time.time(); m(x); print("per sample ms", (time.time() - t) / 32 * 1000)
