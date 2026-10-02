"""Q1 fixes: post-assembly shape changes on the v2 selection (selection fixed, growth only into unowned px)."""
from base import *
from common import disk_info, disk_mask

ctx = load_ctx(load_s1=True)
QMP = lambda c, m, a, mp: c["q"] * c["mean_p"]
FP = {}
for s in ctx.stems:
    FP[s] = assemble_prov(ctx.C[s], V2, score_fn=QMP)
half_report(ctx, {s: FP[s][0] for s in ctx.stems}, "v2")
K3 = np.ones((3, 3), np.uint8)
KX = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def up(s):
    p = ctx.probs[s].astype(np.float32) / 255.0
    return [cv2.resize(p[i], (2048, 2048), interpolation=cv2.INTER_LINEAR) for i in range(2)]


UP = {}
DM = {}


def get_up(s):
    if s not in UP:
        UP.clear()
        UP[s] = up(s)
        DM.clear()
        DM[s] = disk_mask(disk_info(s))
    return UP[s], DM[s]


def to_box(x, y, m, margin):
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + m.shape[1] + margin, 2048), min(y + m.shape[0] + margin, 2048)
    big = np.zeros((y1 - y0, x1 - x0), np.uint8)
    big[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] = m
    return x0, y0, x1, y1, big


def geo_grow(big, region, iters):
    cur = big.copy()
    allowed = (region | big.astype(bool)).astype(np.uint8)
    for _ in range(iters):
        nxt = cv2.dilate(cur, K3) & allowed
        if (nxt == cur).all():
            break
        cur = nxt
    return cur.astype(bool)


def post(s, fn, order_owner=True):
    """fn(x0,y0,big,c, maps) -> new full-box mask (bool). Growth only into px not owned by any original mask
    nor already claimed by an earlier (higher-score) grown mask."""
    F, prov = FP[s]
    if not F:
        return []
    owner = np.zeros((2048, 2048), bool)
    for x, y, m in F:
        owner[y:y + m.shape[0], x:x + m.shape[1]] |= m
    out = []
    for (x, y, m), (c, sc) in zip(F, prov):
        x0, y0, x1, y1, big = to_box(x, y, m, 60)
        new = fn(s, x0, y0, x1, y1, big, c)
        add = new & ~big.astype(bool) & ~owner[y0:y1, x0:x1]
        res = big.astype(bool) | add
        owner[y0:y1, x0:x1] |= add
        out.append((x0, y0, res))
    return out


def shrink(s, fn):
    F, prov = FP[s]
    return [(x, y, fn(m, c)) for (x, y, m), (c, sc) in zip(F, prov)]


def f_recon(head, t, iters):
    def fn(s, x0, y0, x1, y1, big, c):
        (p0, p1), dm = get_up(s)
        src = (p0 if head == 0 else p1)[y0:y1, x0:x1]
        reg = (src > t) & dm[y0:y1, x0:x1]
        return geo_grow(big, reg, iters)
    return fn


def f_close(k):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    def fn(s, x0, y0, x1, y1, big, c):
        return cv2.morphologyEx(big, cv2.MORPH_CLOSE, ker).astype(bool)
    return fn


PROPS = {}


def f_prior(s, x0, y0, x1, y1, big, c):
    if s not in PROPS:
        PROPS.clear(); PROPS[s] = props_of(s)
    pr = mu.decode(PROPS[s][c["idx"]]["rle"]).astype(bool)
    return pr[y0:y1, x0:x1]


def f_soft(tlo):
    def fn(s, x0, y0, x1, y1, big, c):
        h, w = c["soft"].shape
        soft = np.zeros((y1 - y0, x1 - x0), np.uint8)
        soft[c["y"] - y0:c["y"] - y0 + h, c["x"] - x0:c["x"] - x0 + w] = c["soft"]
        return geo_grow(big, soft > int(tlo * 255), 1000)
    return fn


def f_lenonly(head, t, d):
    """geodesic growth, but drop added px within d px of the original mask unless they connect a far part."""
    def fn(s, x0, y0, x1, y1, big, c):
        (p0, p1), dm = get_up(s)
        src = (p0 if head == 0 else p1)[y0:y1, x0:x1]
        g = geo_grow(big, (src > t) & dm[y0:y1, x0:x1], 1000)
        add = g & ~big.astype(bool)
        dist = cv2.distanceTransform((1 - big).astype(np.uint8), cv2.DIST_L2, 3)
        far = add & (dist > d)
        if not far.any():
            return big.astype(bool)
        # keep far parts + the near px that lie within d px of a far px (connector band)
        dfar = cv2.distanceTransform((~far).astype(np.uint8), cv2.DIST_L2, 3)
        keep = far | (add & (dfar <= d + 1))
        return big.astype(bool) | keep
    return fn


tests = [
    ("erode1 cross (4-nbr)", lambda: {s: shrink(s, lambda m, c: cv2.erode(m.astype(np.uint8), KX).astype(bool)) for s in ctx.stems}),
    ("grow1 3x3 (assemble grow=1)", lambda: {s: post(s, lambda s_, x0, y0, x1, y1, big, c: cv2.dilate(big, K3).astype(bool)) for s in ctx.stems}),
    ("grow1 cross", lambda: {s: post(s, lambda s_, x0, y0, x1, y1, big, c: cv2.dilate(big, KX).astype(bool)) for s in ctx.stems}),
]
for k in (3, 5, 9):
    tests.append((f"closing ellipse {k}", (lambda k=k: {s: post(s, f_close(k)) for s in ctx.stems})))
for tlo in (0.3, 0.4):
    tests.append((f"S2 soft hysteresis .5 -> {tlo}", (lambda tlo=tlo: {s: post(s, f_soft(tlo)) for s in ctx.stems})))
tests.append(("union with S1 proposal blob", lambda: {s: post(s, f_prior) for s in ctx.stems}))
for head in (1, 0):
    for t in (0.5, 0.4, 0.3):
        for it in (3, 1000):
            tests.append((f"geo-grow S1 head{head} > {t} iters {it}", (lambda head=head, t=t, it=it: {s: post(s, f_recon(head, t, it)) for s in ctx.stems})))
for head, t, d in ((1, 0.5, 3), (1, 0.4, 3), (0, 0.5, 3), (0, 0.4, 3), (1, 0.4, 5)):
    tests.append((f"length-only grow head{head} > {t} d{d}", (lambda head=head, t=t, d=d: {s: post(s, f_lenonly(head, t, d)) for s in ctx.stems})))

import sys
sel = sys.argv[1:]
for nm, f in tests:
    if sel and not any(x in nm for x in sel):
        continue
    t0 = time.time()
    half_report(ctx, f(), nm)
