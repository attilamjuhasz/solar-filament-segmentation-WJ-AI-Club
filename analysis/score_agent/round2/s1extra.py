"""Q4c: S1-only instances as an extra candidate source for objects the v2 assembly does not cover."""
from base import *
from rs3 import pav_fit
ctx = load_ctx(load_s1=True)
QMP = lambda c, m, a, mp: c["q"] * c["mean_p"]
PRESETS = {"best": dict(t_hi=0.6, t_lo=0.45, gap=8, min_area=400, head=0), "loose": dict(t_hi=0.5, t_lo=0.4, gap=8, min_area=200, head=0)}
for pn, PS in PRESETS.items():
    base_fin, extra = {}, {}
    for s in ctx.stems:
        fin, _ = assemble_prov(ctx.C[s], V2, score_fn=QMP)
        owner = np.zeros((2048, 2048), bool)
        for x, y, m in fin:
            owner[y:y + m.shape[0], x:x + m.shape[1]] |= m
        p0 = cv2.resize(ctx.probs[s][0].astype(np.float32) / 255, (2048, 2048))
        ex = []
        for x, y, m in ctx.s1_inst(s, **PS):
            a = int(m.sum())
            if a == 0 or (owner[y:y + m.shape[0], x:x + m.shape[1]] & m).sum() > 0.2 * a:
                continue
            free = m & ~owner[y:y + m.shape[0], x:x + m.shape[1]]
            oi = ctx.obj_ious(s, x, y, free)
            ious = np.array([b for b, _, _ in oi])
            ex.append(dict(x=x, y=y, m=free, mp=float(p0[y:y + m.shape[0], x:x + m.shape[1]][free].mean()), area=int(free.sum()),
                           yv=float((ious * (ious > .5)).mean())))
        base_fin[s], extra[s] = fin, ex
    allx = [(s, e) for s in ctx.stems for e in extra[s]]
    print(f"[{pn}] uncovered S1 instances: {len(allx)}; with y>.228: {sum(e['yv'] > .228 for _, e in allx)}; mean y {np.mean([e['yv'] for _, e in allx]):.3f}")
    # cross-fit isotonic on mean p0 -> y; add if calibrated > .228
    out = {}
    for tr, te in ((ctx.A, ctx.B), (ctx.B, ctx.A)):
        xs = np.array([e["mp"] for s in tr for e in extra[s]]); ys = np.array([e["yv"] for s in tr for e in extra[s]])
        w = np.array([len(ctx.G.by_stem[s]) for s in tr for e in extra[s]], float)
        kx, ky = pav_fit(xs, ys, w) if len(xs) else (np.array([0, 1]), np.array([0, 0]))
        for s in te:
            add = [(e["x"], e["y"], e["m"]) for e in extra[s] if np.interp(e["mp"], kx, ky) > 0.228 and e["area"] >= 100]
            out[s] = base_fin[s] + add
    half_report(ctx, out, f"v2 + uncovered S1-only [{pn}] gated by cross-fit isotonic(mean p0) > .228")
    for thr in (0.8, 0.9):
        half_report(ctx, {s: base_fin[s] + [(e["x"], e["y"], e["m"]) for e in extra[s] if e["mp"] > thr] for s in ctx.stems}, f"v2 + uncovered S1-only [{pn}] mean p0 > {thr}")
