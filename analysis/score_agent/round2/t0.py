from base import *
ctx = load_ctx(load_s1=False)
fin = {s: assemble(ctx.C[s], V2) for s in ctx.stems}
half_report(ctx, fin, "v2 baseline")
print(len(ctx.A), len(ctx.B), sum(len(ctx.G.by_stem[s]) for s in ctx.A), sum(len(ctx.G.by_stem[s]) for s in ctx.B))
