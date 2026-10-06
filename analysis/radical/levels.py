"""Do merged (B) / faint (C) proposals add matchable objects? Oracle with levels AP vs APBC; LOO oracle too."""
import decode as D
from decode import *
for lv in ("AP", "APB", "APBC"):
    D.P["levels"] = lv
    run(f"model q*mean_p levels={lv}", model_score, 0.225)
    run(f"realized oracle levels={lv}", real_score, 0.269)
    run(f"LOO oracle levels={lv} (multi stems)", loo_score, 0.22, stems_=MULTI, loo=True)
