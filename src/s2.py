"""Stage 2: per-blob refiner at native resolution.

For every blob proposal: square window around it (native px), resampled to S x S, input =
[intensity, limb-flattened contrast, r/R, prior blob mask] (+ optional S1 precise/union probability
crops with --s1-ch) -> UNet-ResNet34 predicts
  * the exact mask of the filament the blob belongs to
  * q = P(predicted mask matches an annotator's filament at IoU > .5)   (the keep/reject score)

Train:   python src/s2.py --name s2_r34 --s1 s1_r34_f0 --val-fold 0
Infer:   python src/s2.py --name s2_r34 --s1 s1_r34_f0 --predict val|test --props tta
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mu
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import segmentation_models_pytorch as smp  # noqa: E402

from common import CACHE, RUNS, device, disk_info, load_inst, make_planes  # noqa: E402

S = 256
MAX_READ = 3  # max annotator readings per stem
MIN_SIDE, MAX_SIDE = 128, 1536


class MaskQ(nn.Module):
    """Mask-aware quality head: encoder features + the DETACHED predicted mask + the prior (+ S1 channels).

    Mask-pooled deep features, a small conv over [stride-16 features, max-pooled mask, prior] and cheap mask
    statistics -> one logit for E[IoU * 1(IoU > .5)]. The mask input is detached, so the q loss never reaches the
    decoder/segmentation head; it does train the shared encoder (like the aux head), scaled by --wq.
    """

    def __init__(self, n_extra=0):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(256 + 2 + n_extra, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.mlp = nn.Sequential(nn.Linear(512 * 2 + 256 * 2 + 128 + 6, 256), nn.ReLU(True), nn.Dropout(0.2),
                                 nn.Linear(256, 1))

    def forward(self, f, m, x):
        p = torch.sigmoid(m).detach()
        pr = x[:, 3:4]
        a16, a8 = F.avg_pool2d(p, 16), F.avg_pool2d(p, 32)
        ring16 = (F.max_pool2d(a16, 3, 1, 1) - a16).clamp(min=0)

        def mpool(t, a):
            return (t * a).sum((2, 3)) / a.sum((2, 3)).clamp(min=1e-3)

        ex = [F.max_pool2d(x[:, 4:], 16)] if x.shape[1] > 4 else []
        c = self.conv(torch.cat([f[4], F.max_pool2d(p, 16), F.max_pool2d(pr, 16)] + ex, 1)).mean((2, 3))
        h = (p > 0.5).float()
        ha = h.sum((2, 3)).clamp(min=1)
        edge = torch.cat([h[..., 0, :], h[..., -1, :], h[..., :, 0], h[..., :, -1]], -1).amax(-1)
        unsure = (((p > 0.3) & (p < 0.7)).float().sum((2, 3)) / ha).clamp(max=4.0)  # bounded when the mask is empty
        st = torch.cat([torch.log1p(ha) / 10, (p * h).sum((2, 3)) / ha, unsure,
                        edge, (h * pr).sum((2, 3)) / pr.sum((2, 3)).clamp(min=1), (h * pr).sum((2, 3)) / ha], 1)
        return self.mlp(torch.cat([f[5].mean((2, 3)), mpool(f[5], a8), mpool(f[4], a16), mpool(f[4], ring16), c, st], 1))


class S2Net(nn.Module):
    """UNet-ResNet34 mask path + MaskQ quality head; returns (mask_logits, q_logits) like smp's aux_params model."""

    def __init__(self, in_ch=4, pretrained=True):
        super().__init__()
        self.net = smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=in_ch, classes=1)
        self.q = MaskQ(in_ch - 4)

    def forward(self, x):
        f = self.net.encoder(x)
        m = self.net.segmentation_head(self.net.decoder(f))
        return m, self.q(f, m, x)


def build_model(pretrained=True, in_ch=4, qhead="aux"):
    if qhead == "mask":
        return S2Net(in_ch, pretrained)
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=in_ch, classes=1,
                    aux_params=dict(classes=1, pooling="avg", dropout=0.2))


def window(box, rng=None):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = float(np.clip(1.5 * max(w, h) + 48, MIN_SIDE, MAX_SIDE))
    if rng is not None:
        side = float(np.clip(side * rng.uniform(0.8, 1.25), MIN_SIDE, MAX_SIDE))
        cx += rng.uniform(-0.1, 0.1) * side
        cy += rng.uniform(-0.1, 0.1) * side
    side = int(round(side))
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def crop_pad(arr, x0, y0, side):
    """Native crop [y0:y0+side, x0:x0+side] with zero padding outside the frame."""
    H, W = arr.shape[:2]
    out = np.zeros((side, side), arr.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = arr[ya:yb, xa:xb]
    return out


def resize_to(a, n, area=True):
    if a.shape[0] == n:
        return a
    interp = cv2.INTER_AREA if (area and a.shape[0] > n) else cv2.INTER_LINEAR
    return cv2.resize(a, (n, n), interpolation=interp)


def s1_crop(s1p, x0, y0, side):
    """S1 probability maps (C, 1024, 1024) uint8 -> (C, S, S) float in [0, 1] for the native window.

    Exact pixel-centre mapping: S-pixel j sits at native x0 + (j + .5) * step - .5, i.e. at
    1024-coordinate (x0 + (j + .5) * step) / 2 - .5 (bilinear, zero outside the frame).
    """
    step = side / S
    M = np.array([[step / 2, 0, (x0 + 0.5 * step) / 2 - 0.5],
                  [0, step / 2, (y0 + 0.5 * step) / 2 - 0.5]], np.float64)
    out = [cv2.warpAffine(np.ascontiguousarray(c), M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in s1p]
    return np.stack(out).astype(np.float32) / 255.0


def make_input(img2048, stem, prior_full, x0, y0, side, s1p=None):
    crop = crop_pad(img2048, x0, y0, side)
    step = side / S
    planes = make_planes(resize_to(crop, S), disk_info(stem), x0, y0, step)
    pr = resize_to(crop_pad(prior_full, x0, y0, side).astype(np.float32), S)
    parts = [planes, pr[None]]
    if s1p is not None:
        parts.append(s1_crop(s1p, x0, y0, side))
    return np.concatenate(parts, 0).astype(np.float32)


def load_s1(s1_dir, stem):
    if not s1_dir:
        return None
    p = np.load(os.path.join(s1_dir, stem + ".npy"), mmap_mode="r")
    assert p.dtype == np.uint8 and p.shape[0] == 2, (s1_dir, stem, p.dtype, p.shape)
    return p


def pick_target(prior, inst, areas):
    """Index of the GT instance (in one reading) the prior blob belongs to, or 0."""
    labs, inter = np.unique(inst[prior], return_counts=True)
    best, best_l = 0.0, 0
    pa = prior.sum()
    for l, it in zip(labs, inter):
        if l == 0:
            continue
        iou = it / (pa + areas[l] - it)
        ok = iou > 0.1 or it / areas[l] > 0.5 or it / pa > 0.5
        if ok and iou > best:
            best, best_l = iou, int(l)
    return best_l


# ----------------------------------------------------------------------------------------- data
def corrupt(m, inst, lab, rng):
    """Synthetic proposal from a GT mask: grow/shrink, cut, merge with a neighbour, fragment."""
    m = m.astype(np.uint8)
    k = int(rng.integers(0, 5))
    if k:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        m2 = cv2.erode(m, ker) if rng.random() < 0.5 else cv2.dilate(m, ker)
        if m2.sum() >= 20:
            m = m2
    ys, xs = np.nonzero(m)
    if rng.random() < 0.3 and len(xs) > 50:  # cut along a random line, keep one side
        a = rng.uniform(0, np.pi)
        proj = (xs - xs.mean()) * np.cos(a) + (ys - ys.mean()) * np.sin(a)
        t = np.quantile(proj, rng.uniform(0.3, 0.7))
        keep = proj > t if rng.random() < 0.5 else proj <= t
        m = np.zeros_like(m)
        m[ys[keep], xs[keep]] = 1
    if rng.random() < 0.2:  # merge with the nearest other instance within 60 px
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))
        y0, y1, x0, x1 = max(ys.min() - 70, 0), ys.max() + 70, max(xs.min() - 70, 0), xs.max() + 70
        near = cv2.dilate(m[y0:y1, x0:x1], ker) > 0
        others = np.unique(inst[y0:y1, x0:x1][near])
        others = others[(others != 0) & (others != lab)]
        if len(others):
            m = (m.astype(bool) | (inst == rng.choice(others))).astype(np.uint8)
    if rng.random() < 0.2:  # fragment
        noise = cv2.GaussianBlur(rng.random(m.shape).astype(np.float32), (0, 0), 6)
        m2 = m & (noise > np.quantile(noise, 0.3)).astype(np.uint8)
        if m2.sum() >= 20:
            m = m2
    return m.astype(bool)


class S2Train(Dataset):
    def __init__(self, stems, readings, props_dir, n_samples, p_prop=0.6, seed=0, q_avg=False, s1_dir=None,
                 levels=None, flag_prop=False):
        self.stems, self.readings, self.props_dir, self.s1_dir = list(stems), readings, props_dir, s1_dir
        self.levels, self.flag_prop = levels, flag_prop
        self.n, self.p_prop, self.seed, self.q_avg = n_samples, p_prop, seed, q_avg
        w = np.array([len(readings[s]) for s in self.stems], np.float64)
        self.p_stem = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2**31))
        while True:
            stem = self.stems[rng.choice(len(self.stems), p=self.p_stem)]
            rds = self.readings[stem]
            is_prop = rng.random() < self.p_prop
            if is_prop:
                props = pickle.load(open(os.path.join(self.props_dir, stem + ".pkl"), "rb"))
                if self.levels:
                    props = [d for d in props if d["level"] in self.levels]
                if not props:
                    continue
                d = props[rng.integers(len(props))]
                prior = mu.decode(d["rle"]).astype(bool)
            else:
                src = load_inst(rds[rng.integers(len(rds))])
                labs = np.unique(src)[1:]
                if not len(labs):
                    continue
                lab = int(rng.choice(labs))
                prior = corrupt(src == lab, src, lab, rng)
            if prior.sum() >= 20:
                break
        r_mask = int(rng.integers(len(rds)))
        insts = [load_inst(r) for r in rds] if self.q_avg else [load_inst(rds[r_mask])]
        if not self.q_avg:
            r_mask = 0
        inst = insts[r_mask]
        areas = np.bincount(inst.ravel())
        t = pick_target(prior, inst, areas)
        ys, xs = np.nonzero(prior)
        box = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        x0, y0, side = window(box, rng)
        img = np.load(os.path.join(CACHE, "img2048", stem + ".npy"), mmap_mode="r")
        x = make_input(img, stem, prior, x0, y0, side, load_s1(self.s1_dir, stem))
        tgt = (inst == t) if t else np.zeros_like(prior)
        y = resize_to(crop_pad(tgt, x0, y0, side).astype(np.float32), S)[None]
        gt_area = float(areas[t]) if t else 0.0
        if self.q_avg:  # every reading's target, so the q label can average over annotators
            y_all = np.zeros((MAX_READ, S, S), np.float32)
            r_areas = np.zeros(MAX_READ, np.float32)
            r_valid = np.zeros(MAX_READ, np.float32)
            for j, ins in enumerate(insts[:MAX_READ]):
                ar = areas if j == r_mask else np.bincount(ins.ravel())
                tj = t if j == r_mask else pick_target(prior, ins, ar)
                r_valid[j] = 1.0
                if tj:
                    y_all[j] = resize_to(crop_pad(ins == tj, x0, y0, side).astype(np.float32), S)
                    r_areas[j] = float(ar[tj])

        g = rng.uniform(0.9, 1.1)
        x[:2] = x[:2] * g + rng.uniform(-0.1, 0.1)
        if rng.random() < 0.3:
            x[:2] += rng.normal(0, 0.05, x[:2].shape).astype(np.float32)
        k = int(rng.integers(4))
        flip = rng.random() < 0.5
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        if flip:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        meta = torch.tensor([gt_area, (side / S) ** 2] + ([float(is_prop)] if self.flag_prop else []),
                            dtype=torch.float32)
        out = [torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), meta]
        if self.q_avg:
            y_all = np.rot90(y_all, k, (1, 2))
            if flip:
                y_all = y_all[:, :, ::-1]
            out += [torch.from_numpy(np.ascontiguousarray(y_all)), torch.from_numpy(np.stack([r_areas, r_valid]))]
        return tuple(out)


def s2_loss(mask_logits, q_logits, y, meta, wq, y_all=None, rmeta=None, qw=None):
    """Mask loss only on samples whose blob belongs to a GT filament (existence is the q head's job,
    otherwise the mask shrinks toward the annotators' intersection). q head: soft target
    IoU * 1[IoU > .5] so sigmoid(q) estimates the blob's expected PQ numerator contribution.
    With y_all/rmeta (--q-avg) that target is averaged over every annotator reading of the stem,
    which is exactly the quantity pooled PQ rewards, instead of one random reading."""
    has = (y.sum((1, 2, 3)) > 0).float()
    nh = has.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(mask_logits, y, reduction="none").mean((1, 2, 3)) * has).sum() / nh
    p = torch.sigmoid(mask_logits)
    inter = (p * y).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    dice = ((1 - (2 * inter + 1) / (denom + 1)) * has).sum() / nh
    with torch.no_grad():  # q target: native-scale IoU of the hard prediction vs the full GT instance
        hard = (p > 0.5).float()
        tgt = (y > 0.5).float()
        it = (hard * tgt).sum((1, 2, 3)) * meta[:, 1]
        pa = hard.sum((1, 2, 3)) * meta[:, 1]
        if y_all is None:
            iou = it / (pa + meta[:, 0] - it).clamp(min=1)
            q = iou * ((iou > 0.5) & (meta[:, 0] > 0)).float()
        else:
            areas, valid = rmeta[:, 0], rmeta[:, 1]  # (B, R)
            it_r = (hard * (y_all > 0.5).float()).sum((2, 3)) * meta[:, 1:2]
            iou_r = it_r / (pa[:, None] + areas - it_r).clamp(min=1)
            contrib = iou_r * ((iou_r > 0.5) & (areas > 0)).float() * valid
            q = contrib.sum(1) / valid.sum(1).clamp(min=1)
        q = q.clamp(max=1.0)  # S-scale area can exceed the native area by ~1% -> IoU proxy slightly > 1
    if qw is None:
        lq = F.binary_cross_entropy_with_logits(q_logits[:, 0], q)
    else:  # e.g. down-weight synthetic blobs so q is learned mostly on real S1 proposals
        lq = (F.binary_cross_entropy_with_logits(q_logits[:, 0], q, reduction="none") * qw).sum() / qw.sum().clamp(min=1e-6)
    return bce + dice + wq * lq, q.mean().item()


# ------------------------------------------------------------------------------------ inference
FLIPS = [(), (3,), (2,), (2, 3)]


@torch.no_grad()
def refine(model, stem, props, dev, tta=True, bs=32, thr=0.3, s1_dir=None):
    """-> list of candidates {x, y, soft, q, level, idx, ...}.

    `soft` is the uint8 (0..255) native-res probability crop at (x, y), zeroed outside the
    components of (prob > thr) that touch the prior blob -- so any final threshold >= thr can be
    applied later in assembly. Candidates may overlap; assembly resolves ownership.
    """
    model.eval()
    if not props:
        return []
    img = np.load(os.path.join(CACHE, "img2048", stem + ".npy"), mmap_mode="r")
    s1p = load_s1(s1_dir, stem)
    wins, xs, priors = [], [], []
    for d in props:
        prior = mu.decode(d["rle"]).astype(bool)
        x0, y0, side = window(d["box"])
        wins.append((x0, y0, side))
        priors.append(prior)
        xs.append(make_input(img, stem, prior, x0, y0, side, s1p))
    probs, qs = [], []
    for i in range(0, len(xs), bs):
        xb = torch.from_numpy(np.stack(xs[i:i + bs])).to(dev)
        acc_m = torch.zeros(len(xb), 1, S, S, device=dev)
        acc_q = torch.zeros(len(xb), device=dev)
        views = FLIPS if tta else [()]
        for f in views:
            xv = torch.flip(xb, f) if f else xb
            m, q = model(xv)
            m = torch.sigmoid(m)
            acc_m += torch.flip(m, f) if f else m
            acc_q += torch.sigmoid(q[:, 0])
        probs += list((acc_m / len(views))[:, 0].cpu().numpy())
        qs += list((acc_q / len(views)).cpu().numpy())
    out = []
    for k, (d, (x0, y0, side), prior, pr, q) in enumerate(zip(props, wins, priors, probs, qs)):
        big = resize_to(pr, side, area=False)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, 2048), min(y0 + side, 2048)
        loc = big[ya - y0:yb - y0, xa - x0:xb - x0]
        m = loc > thr
        if not m.any():
            continue
        # keep only components touching the (slightly dilated) prior blob
        _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        pd = cv2.dilate(prior[ya:yb, xa:xb].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        keep = np.unique(lab[pd & m])
        keep = keep[keep > 0]
        if not len(keep):
            continue
        m = np.isin(lab, keep)
        ys, xs = np.nonzero(m)
        by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        soft = np.where(m, np.round(loc * 255), 0)[by0:by1, bx0:bx1].astype(np.uint8)
        out.append(dict(x=int(xa + bx0), y=int(ya + by0), soft=soft, q=float(q), level=d["level"], idx=k,
                        area50=int((soft > 127).sum()), peak_u=d["peak_u"], mean_p=d["mean_p"]))
    return out


# --------------------------------------------------------------------------------------- train
def split(val_fold):
    from s1 import split as s1_split
    return s1_split(val_fold)


def train(args):
    torch.manual_seed(args.seed)
    dev = device()
    out_dir = os.path.join(RUNS, args.name)
    os.makedirs(out_dir, exist_ok=True)
    tr, va, readings = split(args.val_fold)
    props_dir = os.path.join(RUNS, args.s1, "props_" + args.props)
    args.s1_probs = args.s1_probs or "probs_plain"  # resolved value is recorded in args.json
    s1_dir = os.path.join(RUNS, args.s1, args.s1_probs) if args.s1_ch else None
    assert not args.prop_levels or set(args.prop_levels) <= set("APBC"), f"bad --prop-levels {args.prop_levels}"
    ds = S2Train(tr, readings, props_dir, args.samples, p_prop=args.p_prop, seed=args.seed, q_avg=args.q_avg,
                 s1_dir=s1_dir, levels=args.prop_levels, flag_prop=args.q_syn_w != 1.0)
    extra = dict(persistent_workers=True, prefetch_factor=4) if args.workers > 0 else {}
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, drop_last=True, **extra)
    model = build_model(in_ch=6 if args.s1_ch else 4, qhead=args.qhead).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(dl)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"))
    log = open(os.path.join(out_dir, "log.txt"), "a")
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, runq = time.time(), 0.0, 0.0
        for batch in dl:
            x, y, meta = (b.to(dev) for b in batch[:3])
            extra = [b.to(dev) for b in batch[3:]]
            wq = args.wq * min(1.0, step / max(1, 2 * len(dl)))
            m, q = model(x)
            qw = torch.where(meta[:, 2] > 0, 1.0, args.q_syn_w) if args.q_syn_w != 1.0 else None
            loss, qm = s2_loss(m, q, y, meta, wq, *extra, qw=qw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run += loss.item()
            runq += qm
        torch.save(model.state_dict(), os.path.join(out_dir, f"ep{ep}.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
        msg = f"ep {ep} loss {run / len(dl):.4f} q-rate {runq / len(dl):.3f} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()


def predict(args):
    dev = device()
    cfg = json.load(open(os.path.join(RUNS, args.name, "args.json")))
    s1_ch = cfg.get("s1_ch", False)
    model = build_model(pretrained=False, in_ch=6 if s1_ch else 4, qhead=cfg.get("qhead", "aux")).to(dev)
    model.load_state_dict(torch.load(os.path.join(RUNS, args.name, args.ckpt), map_location=dev))
    s1_dir = None
    if s1_ch:  # S1 input maps default to exactly what the model was trained with; override explicitly
        s1_dir = os.path.join(RUNS, args.s1_maps or cfg["s1"], args.s1_probs or cfg.get("s1_probs", "probs_plain"))
        print("S1 input maps:", s1_dir, flush=True)
    props_dir = os.path.join(RUNS, args.s1, "props_" + args.props)
    tr, va, _ = split(args.val_fold)
    test = sorted(f[:-4] for f in os.listdir(os.path.join(CACHE, "img1024"))
                  if f.endswith(".npy") and f[:-4] not in set(tr) | set(va))
    stems = {"val": va, "test": test}[args.predict]
    if s1_dir:
        missing = [s for s in stems if not os.path.exists(os.path.join(s1_dir, s + ".npy"))]
        if missing:
            raise SystemExit(f"{len(missing)} {args.predict} stems lack S1 maps in {s1_dir} "
                             f"(e.g. {missing[0]}); run src/s1.py --predict {args.predict} for that S1 first")
    odir = os.path.join(RUNS, args.name, f"cands_{args.predict}_{args.props}_{args.ckpt[:-3]}{args.tag}")
    os.makedirs(odir, exist_ok=True)
    t0 = time.time()
    for i, s in enumerate(stems):
        props = pickle.load(open(os.path.join(props_dir, s + ".pkl"), "rb"))
        pickle.dump(refine(model, s, props, dev, tta=not args.no_tta, s1_dir=s1_dir),
                    open(os.path.join(odir, s + ".pkl"), "wb"))
        if i % 20 == 0:
            print(f"{i}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    print("done", odir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--props", default="plain")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p-prop", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--q-avg", action="store_true", help="q label averaged over all annotator readings")
    ap.add_argument("--tag", default="", help="suffix for the candidates folder (e.g. which S1 made the proposals)")
    ap.add_argument("--s1-ch", action="store_true", help="add S1 precise/union probability crops as input channels")
    ap.add_argument("--s1-probs", default=None,
                    help="S1 probability folder for --s1-ch inputs (train default probs_plain; predict default = training's)")
    ap.add_argument("--s1-maps", default=None, help="predict: S1 run providing the input maps (default = training's --s1)")
    ap.add_argument("--qhead", default="aux", choices=["aux", "mask"], help="aux = smp pooled head; mask = MaskQ head")
    ap.add_argument("--wq", type=float, default=1.0, help="weight of the q loss (after its warm-up ramp)")
    ap.add_argument("--q-syn-w", type=float, default=1.0, help="q-loss weight of synthetic (non-proposal) blobs")
    ap.add_argument("--prop-levels", default=None, help="only sample S1 proposals of these levels, e.g. AP")
    a = ap.parse_args()
    predict(a) if a.predict else train(a)
