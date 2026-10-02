"""Q1: anatomy of near-misses (IoU .3-.5) for the v2 assembly vs TPs. Pred-vs-GT shape diagnostics."""
from base import *
from skimage.morphology import skeletonize
from collections import Counter

ctx = load_ctx(load_s1=False)
fin, prov = {}, {}
for s in ctx.stems:
    f, p = assemble_prov(ctx.C[s], V2, score_fn=lambda c, m, a, mp: c["q"] * c["mean_p"])
    fin[s], prov[s] = f, p
half_report(ctx, fin, "v2 (assemble_prov)")


def skel_len(m):
    if m.sum() == 0:
        return 0.0
    sk = skeletonize(m)
    # length with diagonal weighting: count 4-neighbour links + sqrt2 diagonal links / 2 per pixel approx
    n = sk.sum()
    return float(max(n, 1))


def crop_pair(x, y, m, gfull):
    """local pred mask (x,y,m) + full-frame GT bool -> crops (P, G) on union bbox + pad."""
    ys, xs = np.nonzero(gfull)
    h, w = m.shape
    x0 = min(x, xs.min()) - 6; y0 = min(y, ys.min()) - 6
    x1 = max(x + w, xs.max() + 1) + 6; y1 = max(y + h, ys.max() + 1) + 6
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, 2048), min(y1, 2048)
    P = np.zeros((y1 - y0, x1 - x0), bool)
    P[y - y0:y - y0 + h, x - x0:x - x0 + w] = m
    return P, gfull[y0:y1, x0:x1], (x0, y0, x1, y1)


def pair_stats(P, G):
    inter = (P & G).sum(); uni = (P | G).sum()
    iou = inter / max(uni, 1)
    LP, LG = skel_len(P), skel_len(G)
    dP = cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_L2, 3)  # distance to P
    dG = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 3)
    M = G & ~P; E = P & ~G
    return dict(iou=iou, aP=P.sum(), aG=G.sum(), LP=LP, LG=LG, WP=P.sum() / LP, WG=G.sum() / max(LG, 1),
                miss_near=(M & (dP <= 2.5)).sum(), miss_far=(M & (dP > 2.5)).sum(),
                extra_near=(E & (dG <= 2.5)).sum(), extra_far=(E & (dG > 2.5)).sum(), uni=uni, inter=inter)


recs = []  # kind, half, stem, reading, stats
for s in ctx.stems:
    half = "A" if s in set(ctx.A) else "B"
    readings = ctx.G.by_stem[s]
    F = fin[s]
    for ri, (lab, areas) in enumerate(readings):
        labs = np.nonzero(areas[1:] > 0)[0] + 1
        # IoU matrix pred x label
        I = np.zeros((len(F), 256)); INT = np.zeros((len(F), 256))
        for k, (x, y, m) in enumerate(F):
            h, w = m.shape
            v = np.bincount(lab[y:y + h, x:x + w][m], minlength=256).astype(float); v[0] = 0
            INT[k] = v
            I[k] = v / np.maximum(m.sum() + areas - v, 1)
        for k, (x, y, m) in enumerate(F):
            j = int(I[k].argmax()); bi = I[k, j]
            if bi > 0.5:
                kind = "TP"
            elif bi >= 0.3:
                kind = "FPnear"
            else:
                continue
            g = lab == j
            P, G, box = crop_pair(x, y, m, g)
            st = pair_stats(P, G)
            # merge: pred covers >=30% of another GT label
            other = [l for l in labs if l != j and INT[k, l] >= 0.3 * areas[l]]
            # split: GT j covered >=20% by another pred
            oth_pred = [kk for kk in range(len(F)) if kk != k and INT[kk, j] >= 0.2 * areas[j]]
            st.update(merge=len(other), split=len(oth_pred), q=prov[s][k][0].get("q"), score=prov[s][k][1])
            recs.append((kind, half, s, ri, j, k, st))
        for j in labs:
            col = I[:, j] if len(F) else np.zeros(0)
            if len(col) and col.max() > 0.5:
                continue
            if not len(col) or col.max() < 0.3:
                continue
            k = int(col.argmax())
            x, y, m = F[k]
            P, G, box = crop_pair(x, y, m, lab == j)
            st = pair_stats(P, G)
            other = [l for l in labs if l != j and INT[k, l] >= 0.3 * areas[l]]
            oth_pred = [kk for kk in range(len(F)) if kk != k and INT[kk, j] >= 0.2 * areas[j]]
            st.update(merge=len(other), split=len(oth_pred), q=prov[s][k][0].get("q"), score=prov[s][k][1])
            recs.append(("FNnear", half, s, ri, j, k, st))

pickle.dump(recs, open("near_recs.pkl", "wb"))


def summarize(kind):
    R = [r[6] for r in recs if r[0] == kind]
    n = len(R)
    g = lambda k: np.array([r[k] for r in R], float)
    iou, aP, aG, LP, LG, WP, WG = g("iou"), g("aP"), g("aG"), g("LP"), g("LG"), g("WP"), g("WG")
    mn, mf, en, ef, uni = g("miss_near"), g("miss_far"), g("extra_near"), g("extra_far"), g("uni")
    print(f"\n== {kind}: n={n}  mean IoU {iou.mean():.3f}")
    print(f"  area ratio P/G median {np.median(aP / aG):.2f}  [q25 {np.quantile(aP / aG, .25):.2f}, q75 {np.quantile(aP / aG, .75):.2f}]")
    print(f"  length ratio LP/LG median {np.median(LP / LG):.2f} [q25 {np.quantile(LP / LG, .25):.2f}, q75 {np.quantile(LP / LG, .75):.2f}]")
    print(f"  width ratio WP/WG median {np.median(WP / WG):.2f} [q25 {np.quantile(WP / WG, .25):.2f}, q75 {np.quantile(WP / WG, .75):.2f}];  WG median {np.median(WG):.1f}px, WP median {np.median(WP):.1f}px")
    tot = mn + mf + en + ef
    print(f"  share of union-minus-inter error: missing-near(width) {mn.sum() / tot.sum():.2f}, missing-far(length/branch) {mf.sum() / tot.sum():.2f}, "
          f"extra-near(width) {en.sum() / tot.sum():.2f}, extra-far(length/branch) {ef.sum() / tot.sum():.2f}")
    # dominant error per pair
    dom = Counter()
    for r in R:
        parts = dict(short=r["miss_far"], thin=r["miss_near"], wide=r["extra_near"], long_or_branch=r["extra_far"])
        if r["merge"]:
            dom["merge(pred covers other GT)"] += 1
        elif r["split"]:
            dom["split(GT covered by another pred)"] += 1
        else:
            dom[max(parts, key=parts.get)] += 1
    print("  dominant error:", dict(dom.most_common()))
    # oracle: IoU after removing far-extra only / adding far-missing only
    if kind != "TP":
        inter, aPv = g("inter"), aP
        iou_noextra = inter / (uni - ef)
        iou_addmiss = (inter + mf) / uni
        iou_width = (inter + mn) / (uni - en)
        print(f"  oracle fix share reaching IoU>.5: drop far-extra {np.mean(iou_noextra > .5):.2f}, add far-missing {np.mean(iou_addmiss > .5):.2f}, fix width only {np.mean(iou_width > .5):.2f}")
    return R


for k in ("TP", "FPnear", "FNnear"):
    summarize(k)
