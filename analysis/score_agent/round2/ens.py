"""Q4: ep4+ep8 soft-mask averaging (same proposal), with v2 selection and with q-avg selection; split-half."""
from base import *
import assemble as AS
ctx = load_ctx(load_s1=False)
E4 = os.path.join(ROOT, "runs/s2_r34/cands_val_tta_ep4")
for s in ctx.stems:
    c4 = {(c["idx"], c["level"]): c for c in pickle.load(open(os.path.join(E4, s + ".pkl"), "rb"))}
    for c in ctx.C[s]:
        d = c4.get((c["idx"], c["level"]))
        c["q4"] = d["q"] if d else c["q"]
        # average soft on the union box
        if d is None:
            c["soft_avg"] = c["soft"]; c["xa"], c["ya"] = c["x"], c["y"]; continue
        x0, y0 = min(c["x"], d["x"]), min(c["y"], d["y"])
        x1 = max(c["x"] + c["soft"].shape[1], d["x"] + d["soft"].shape[1]); y1 = max(c["y"] + c["soft"].shape[0], d["y"] + d["soft"].shape[0])
        a = np.zeros((y1 - y0, x1 - x0), np.float32)
        a[c["y"] - y0:c["y"] - y0 + c["soft"].shape[0], c["x"] - x0:c["x"] - x0 + c["soft"].shape[1]] += c["soft"]
        a[d["y"] - y0:d["y"] - y0 + d["soft"].shape[0], d["x"] - x0:d["x"] - x0 + d["soft"].shape[1]] += d["soft"]
        c["soft_avg"] = (a / 2).astype(np.uint8); c["xa"], c["ya"] = x0, y0


def run(use_avg, score):
    out = {}
    for s in ctx.stems:
        cs = []
        for c in ctx.C[s]:
            if c["level"] not in "AP":
                continue
            soft = c["soft_avg"] if use_avg else c["soft"]
            x, y = (c["xa"], c["ya"]) if use_avg else (c["x"], c["y"])
            m = soft > 127
            cs.append((score(c), x, y, m, int(m.sum())))
        cs.sort(key=lambda t: -t[0])
        owner = np.zeros((2048, 2048), bool); fin = []
        for sc, x, y, m, a in cs:
            if a < 100 or sc < 0.225:
                continue
            free = m & ~owner[y:y + m.shape[0], x:x + m.shape[1]]
            if free.sum() < 0.8 * a or free.sum() < 100:
                continue
            owner[y:y + m.shape[0], x:x + m.shape[1]] |= free
            fin.append((x, y, free))
        out[s] = fin
    return out


half_report(ctx, run(False, lambda c: c["q"] * c["mean_p"]), "ep8 mask, q8*mean_p (=v2)")
half_report(ctx, run(True, lambda c: c["q"] * c["mean_p"]), "avg(ep4,ep8) mask, q8*mean_p")
half_report(ctx, run(True, lambda c: 0.5 * (c["q"] + c["q4"]) * c["mean_p"]), "avg mask, avg q * mean_p")
half_report(ctx, run(False, lambda c: 0.5 * (c["q"] + c["q4"]) * c["mean_p"]), "ep8 mask, avg q * mean_p")
half_report(ctx, run(False, lambda c: np.sqrt(c["q"] * c["q4"]) * c["mean_p"]), "ep8 mask, geo-mean q * mean_p")
