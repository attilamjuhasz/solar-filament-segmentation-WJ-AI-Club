"""Q2: FP / FN anatomy of the best S2 assembly vs the tuned S1-only instances (per reading, official counting)."""
from collections import Counter, defaultdict

import numpy as np

from ana import BEST, S1BEST, SIZE_BINS, Ctx, assemble_prov, cand_mask, cnt, pq

ctx = Ctx()


def qpu(c, m, area, mprob):
    return c["q"] * (0.5 + 0.5 * c["peak_u"])


def size_bin(a):
    i = int(np.digitize(a, SIZE_BINS) - 1)
    return f"<{SIZE_BINS[i + 1]}" if i < len(SIZE_BINS) - 2 else f">={SIZE_BINS[i]}"


# candidate coverage of every GT instance (any level), mask at best thr
cand_best = {}  # (stem, reading_idx) -> (best iou per label, q of that cand)
for s in ctx.stems:
    for ri, (lab, areas) in enumerate(ctx.G.by_stem[s]):
        cand_best[(s, ri)] = (np.zeros(256), np.zeros(256))
    for c in ctx.C[s]:
        m, area, _ = cand_mask(c, BEST)
        if area == 0:
            continue
        for ri, (lab, areas) in enumerate(ctx.G.by_stem[s]):
            h, w = m.shape
            inter = np.bincount(lab[c["y"]:c["y"] + h, c["x"]:c["x"] + w][m], minlength=256)
            inter[0] = 0
            iou = inter / np.maximum(area + areas - inter, 1)
            bi, bq = cand_best[(s, ri)]
            upd = iou > bi
            bi[upd] = iou[upd]
            bq[upd] = c["q"]


def analyse(name, finals, levels=None):
    fp_cat, fn_cat = Counter(), Counter()
    fp_size, fn_size, tp_size = Counter(), Counter(), Counter()
    fp_level, tp_level = Counter(), Counter()
    fn_cand_q = []
    t = np.zeros(4)
    for s in ctx.stems:
        fin = finals[s]
        lv = levels[s] if levels else ["S1"] * len(fin)
        t += ctx.G.stats(s, fin)
        readings = ctx.G.by_stem[s]
        # per pred, per reading inter vectors
        inters = []
        for x, y, m in fin:
            h, w = m.shape
            row = []
            for lab, areas in readings:
                v = np.bincount(lab[y:y + h, x:x + w][m], minlength=256)
                v[0] = 0
                row.append(v)
            inters.append(row)
        for ri, (lab, areas) in enumerate(readings):
            n_lab = np.nonzero(areas[1:] > 0)[0] + 1
            ious = np.array([[inters[k][ri][j] / max(fin[k][2].sum() + areas[j] - inters[k][ri][j], 1)
                              for j in range(256)] for k in range(len(fin))]) if fin else np.zeros((0, 256))
            cover = np.array([inters[k][ri] for k in range(len(fin))]) if fin else np.zeros((0, 256))
            matched_any_other = []
            for k in range(len(fin)):
                pa = fin[k][2].sum()
                bi = ious[k].max()
                if bi > 0.5:
                    tp_size[size_bin(pa)] += 1
                    tp_level[lv[k]] += 1
                    continue
                n_cov = int(((cover[k] >= 0.3 * np.maximum(areas, 1)) & (np.arange(256) > 0)).sum())
                j = int(cover[k].argmax())
                frag = j > 0 and int((cover[:, j] >= 0.2 * areas[j]).sum()) >= 2
                if n_cov >= 2:
                    cat = "merge(covers>=2 GT)"
                elif frag:
                    cat = "fragment(GT split over >=2 preds)"
                elif bi >= 0.3:
                    cat = "near-miss IoU .3-.5"
                elif bi > 0:
                    cat = "low overlap IoU<.3"
                else:
                    other = any(ctx.obj_ious(s, fin[k][0], fin[k][1], fin[k][2])[r2][0] > 0.5
                                for r2 in range(len(readings)) if r2 != ri)
                    cat = "no overlap, matched in another reading" if other else "no overlap with any reading"
                fp_cat[cat] += 1
                fp_size[size_bin(pa)] += 1
                fp_level[lv[k]] += 1
            bi_c, bq_c = cand_best[(s, ri)]
            for j in n_lab:
                col = ious[:, j] if len(fin) else np.zeros(0)
                if len(col) and col.max() > 0.5:
                    continue
                fn_size[size_bin(areas[j])] += 1
                best = col.max() if len(col) else 0.0
                if len(col) and (cover[:, j] >= 0.2 * areas[j]).sum() >= 2:
                    cat = "fragmented over >=2 preds"
                elif len(col) and best > 0 and ((cover[int(col.argmax())] >= 0.3 * np.maximum(areas, 1))
                                                 & (np.arange(256) > 0)).sum() >= 2:
                    cat = "merged into a pred covering >=2 GT"
                elif best >= 0.3:
                    cat = "near-miss IoU .3-.5"
                elif best > 0:
                    cat = "low overlap IoU<.3"
                else:
                    cat = "no pred overlap"
                if bi_c[j] > 0.5:
                    cat += " | a cand had IoU>.5"
                    fn_cand_q.append(bq_c[j])
                elif bi_c[j] > 0:
                    cat += " | cands only IoU<=.5"
                else:
                    cat += " | no cand overlaps"
                fn_cat[cat] += 1
    print(f"\n===== {name}: PQ={pq(t):.4f} {cnt(t)}")
    print(" FP by cause:")
    for k, v in fp_cat.most_common():
        print(f"   {v:5d}  {k}")
    print(" FN by cause (pred-side | candidate-side):")
    for k, v in sorted(fn_cat.items(), key=lambda kv: -kv[1]):
        print(f"   {v:5d}  {k}")
    if fn_cand_q:
        fq = np.array(fn_cand_q)
        print(f" FN where some cand had IoU>.5: n={len(fq)} q of that cand: median {np.median(fq):.3f}, "
              f"share q>=.35: {(fq >= .35).mean():.2f}")
    print(" size bin |  TP  |  FP  | FN(GT size)")
    for b in [size_bin(a) for a in (100, 300, 700, 2000, 5000, 20000)]:
        print(f"  {b:>7} | {tp_size[b]:4d} | {fp_size[b]:4d} | {fn_size[b]:4d}")
    if levels:
        print(" level TP/FP:", {k: (tp_level[k], fp_level[k]) for k in sorted(set(tp_level) | set(fp_level))})


import os
if os.environ.get("Q2CFG") == "qmp":
    P = dict(score="qmp", thr=float(os.environ.get("Q2THR", 0.5)), ring=0.0, rel=0.0, lam=float(os.environ.get("Q2LAM", 0.3)),
             lam_small=0.0, small=0, a_min=int(os.environ.get("Q2AMIN", 300)), own_frac=0.8,
             levels=os.environ.get("Q2LV", "AP"), grow=0)
    sfn, nm = (lambda c, m, a, mp: c["q"] * c["mean_p"]), f"S2 q*mean_p {P}"
else:
    P, sfn, nm = dict(BEST), qpu, "S2 best (ep4 params, qpu thr.6 lam.35 a_min400 own.8 levels A)"
fin2, lv2 = {}, {}
for s in ctx.stems:
    f, prov = assemble_prov(ctx.C[s], P, score_fn=sfn)
    fin2[s], lv2[s] = f, [c["level"] for c, _ in prov]
analyse(nm, fin2, lv2)
fin1 = {s: ctx.s1_inst(s, **S1BEST) for s in ctx.stems}
analyse("S1-only tuned (t_hi.6 t_lo.45 gap8 min400 head0)", fin1)
for nm, fin in (("S2best", fin2), ("S1only", fin1)):
    tA, tB = ctx.pq_counts(fin, ctx.A), ctx.pq_counts(fin, ctx.B)
    print(f"{nm}: half A {pq(tA):.4f} ({len(ctx.A)} stems)  half B {pq(tB):.4f} ({len(ctx.B)} stems)")
