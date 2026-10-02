from base import *
from scipy.stats import spearmanr
import pandas as pd
ctx = load_ctx(load_s1=False)
rows = []
for s in ctx.stems:
    fin, prov = assemble_prov(ctx.C[s], V2, score_fn=lambda c, m, a, mp: c["q"] * c["mean_p"])
    ngt = np.mean([int((a[1:] > 0).sum()) for _, a in ctx.G.by_stem[s]])
    for (x, y, m), (c, sc) in zip(fin, prov):
        oi = ctx.obj_ious(s, x, y, m)
        best = max(b for b, _, _ in oi)
        anyov = any(inter[1:].sum() > 0 for _, _, inter in oi)
        rows.append(dict(stem=s, y=c["yv"], best=best, anyov=anyov, sc=sc, q=c["q"], mp=c["mean_p"], area=int(m.sum()), ngt=ngt, nkept=len(fin), nr=len(oi)))
d = pd.DataFrame(rows)
kb = d[d.y < 0.05]
print(f"kept {len(d)}, kept-bad (y<.05) {len(kb)}: zero overlap with every reading {(~kb.anyov).mean():.2f}, best IoU .3-.5 {((kb.best >= .3) & (kb.best <= .5)).mean():.2f}, "
      f"best IoU in (0,.3) {((kb.best > 0) & (kb.best < .3)).mean():.2f}")
print(f"kept-bad score median {kb.sc.median():.3f} vs kept-good {d[d.y > .228].sc.median():.3f}; area median {kb.area.median():.0f} vs {d[d.y > .228].area.median():.0f}")
im = d.groupby("stem").agg(nkept=("nkept", "first"), ngt=("ngt", "first"), nbad=("y", lambda v: (v < .05).sum()), nr=("nr", "first"))
im["excess"] = im.nkept - im.ngt
print(f"per image: spearman(nkept, mean #GT/reading) {spearmanr(im.nkept, im.ngt)[0]:.3f}; share of kept-bad in images where nkept > 1.5*#GT: "
      f"{im[im.nkept > 1.5 * im.ngt].nbad.sum() / im.nbad.sum():.2f} (those images = {(im.nkept > 1.5 * im.ngt).mean():.2f} of images)")
print("kept-bad per image distribution:", im.nbad.value_counts().sort_index().to_dict())
# top-10 images by kept-bad
print(im.sort_values("nbad", ascending=False).head(8))
