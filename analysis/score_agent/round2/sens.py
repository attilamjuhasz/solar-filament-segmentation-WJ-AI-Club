from base import *
ctx = load_ctx(load_s1=False)
for k, vals in (("a_min", (50, 100, 200, 300)), ("own_frac", (0.5, 0.6, 0.7, 0.8, 0.9)), ("levels", ("A", "AP", "APB", "APBC"))):
    for v in vals:
        half_report(ctx, {s: assemble(ctx.C[s], dict(V2, **{k: v})) for s in ctx.stems}, f"v2 with {k}={v}")
