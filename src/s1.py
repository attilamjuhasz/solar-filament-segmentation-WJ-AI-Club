"""Stage 1: full-disk proposal network (UNet-ResNet34 on 1024 downsampled disk, 2 heads).

head 0: one annotator's foreground (precision / scoring)
head 1: union of annotators (recall / proposals)

Train:   python src/s1.py --name s1_r34_f0 --val-fold 0
Infer:   python src/s1.py --name s1_r34_f0 --predict val|test|all [--tta]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import segmentation_models_pytorch as smp  # noqa: E402

from assemble import FastGT  # noqa: E402
from common import CACHE, DATA, RUNS, device, disk_info, disk_mask, load_meta, make_planes  # noqa: E402

ANN = os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CROP = 512  # training crop, in pixels of the working resolution (--res, default 1024)


def build_model(encoder="resnet34", pretrained=True):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=2)


def load_img(stem, res=1024):
    return np.load(os.path.join(CACHE, f"img{res}", stem + ".npy"))


def full_planes(stem, res=1024):
    return make_planes(load_img(stem, res), disk_info(stem), 0.0, 0.0, 2048 / res)


# ----------------------------------------------------------------------------------------- data
class S1Train(Dataset):
    def __init__(self, stems, readings, n_samples, seed=0, res=1024):
        self.res = res
        self.stems = list(stems)
        self.readings = readings  # stem -> [reading_id]
        self.n = n_samples
        self.boxes = json.load(open(os.path.join(CACHE, "boxes.json")))
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2**31))
        stem = self.stems[rng.integers(len(self.stems))]
        rid = self.readings[stem][rng.integers(len(self.readings[stem]))]
        info = disk_info(stem)
        res, step = self.res, 2048 / self.res
        if rng.random() < 0.75 and self.boxes.get(stem):
            cx, cy, _, _ = self.boxes[stem][rng.integers(len(self.boxes[stem]))]
            cx, cy = (cx + rng.normal(0, 96)) / step, (cy + rng.normal(0, 96)) / step
        else:
            a, rr = rng.uniform(0, 2 * np.pi), info["r"] * np.sqrt(rng.uniform(0, 0.9))
            cx, cy = (info["cx"] + rr * np.cos(a)) / step, (info["cy"] + rr * np.sin(a)) / step
        ox = int(np.clip(round(cx - CROP / 2), 0, res - CROP))
        oy = int(np.clip(round(cy - CROP / 2), 0, res - CROP))
        sl = (slice(oy, oy + CROP), slice(ox, ox + CROP))
        img = np.load(os.path.join(CACHE, f"img{res}", stem + ".npy"), mmap_mode="r")[sl]
        x = make_planes(np.ascontiguousarray(img), info, step * ox, step * oy, step)
        y0 = np.load(os.path.join(CACHE, f"fg{res}", rid + ".npy"), mmap_mode="r")[sl]
        y1 = np.load(os.path.join(CACHE, f"un{res}", stem + ".npy"), mmap_mode="r")[sl]
        y = np.stack([y0, y1]).astype(np.float32) / 255.0

        # photometric on intensity planes only
        g = rng.uniform(0.9, 1.1)
        x[:2] = x[:2] * g + rng.uniform(-0.1, 0.1)
        if rng.random() < 0.2:
            s = rng.uniform(0.3, 1.0)
            for c in (0, 1):
                x[c] = cv2.GaussianBlur(x[c], (0, 0), s)
        if rng.random() < 0.3:
            x[:2] += rng.normal(0, 0.05, x[:2].shape).astype(np.float32)
        # D4
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


def s1_loss(logits, y):
    l0 = F.binary_cross_entropy_with_logits(logits[:, 0], y[:, 0]) + dice_loss(logits[:, 0], y[:, 0])
    l1 = F.binary_cross_entropy_with_logits(logits[:, 1], y[:, 1]) + dice_loss(logits[:, 1], y[:, 1])
    return l0 + 0.5 * l1


# ------------------------------------------------------------------------------------ inference
D4 = [(k, f) for k in range(4) for f in (False, True)]


@torch.no_grad()
def predict_probs(model, stems, dev, tta=False, bs=2, res=1024):
    """-> {stem: float16 (2, res, res) sigmoid probs}"""
    model.eval()
    out = {}
    views = D4 if tta else [(0, False)]
    for i in range(0, len(stems), bs):
        chunk = stems[i:i + bs]
        x = torch.from_numpy(np.stack([full_planes(s, res) for s in chunk])).to(dev)
        acc = torch.zeros(len(chunk), 2, res, res, device=dev)
        for k, f in views:
            xv = torch.rot90(x, k, (2, 3))
            if f:
                xv = torch.flip(xv, (3,))
            p = torch.sigmoid(model(xv.contiguous()))
            if f:
                p = torch.flip(p, (3,))
            acc += torch.rot90(p, -k, (2, 3))
        acc = (acc / len(views)).cpu().numpy().astype(np.float16)
        for s, p in zip(chunk, acc):
            out[s] = p
    return out


def upsample(p1024):
    return cv2.resize(p1024.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)


def postprocess_s1(prob1024, stem, t_hi=0.5, t_lo=0.35, gap=0, min_area=300, head=0):
    """S1-only instances: hysteresis threshold at 2048, optional gap rejoin, min area.

    Returns disjoint instances as local crops [(x0, y0, bool mask)] to keep memory small.
    """
    if prob1024.dtype == np.uint8:
        prob1024 = prob1024.astype(np.float32) / 255.0
    assert prob1024.max() <= 1.0 + 1e-3, "expects probabilities in [0, 1]"
    p = upsample(prob1024[head])
    p[~disk_mask(disk_info(stem))] = 0
    lo = (p > t_lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(lo, connectivity=8)
    if n <= 1:
        return []
    seeded = np.zeros(n, bool)
    seeded[np.unique(lab[p > t_hi])] = True
    seeded[0] = False
    fg = seeded[lab]
    if gap > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
        _, glab = cv2.connectedComponents(cv2.dilate(fg.astype(np.uint8), k), connectivity=8)
        lab = np.where(fg, glab, 0).astype(np.int32)
    else:
        _, lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    # stats on the (possibly gap-merged) labels: bbox + pixel count per id
    n = int(lab.max()) + 1
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    cnt = np.bincount(ids, minlength=n)
    x0 = np.full(n, 1 << 30)
    y0 = np.full(n, 1 << 30)
    x1 = np.full(n, -1)
    y1 = np.full(n, -1)
    np.minimum.at(x0, ids, xs)
    np.minimum.at(y0, ids, ys)
    np.maximum.at(x1, ids, xs)
    np.maximum.at(y1, ids, ys)
    out = []
    for i in np.nonzero(cnt >= min_area)[0]:
        if i == 0:
            continue
        sl = (slice(y0[i], y1[i] + 1), slice(x0[i], x1[i] + 1))
        out.append((int(x0[i]), int(y0[i]), lab[sl] == i))
    return out


# --------------------------------------------------------------------------------------- train
def split(val_fold):
    meta = load_meta()
    readings = meta.groupby(meta.file_name.str[:-5]).image_id.apply(list).to_dict()
    stems = meta.assign(stem=meta.file_name.str[:-5]).drop_duplicates("stem")
    tr = stems[stems.fold != val_fold].stem.tolist() if val_fold >= 0 else stems.stem.tolist()
    va = stems[stems.fold == val_fold].stem.tolist() if val_fold >= 0 else []
    return tr, va, readings


def validate(model, va, gt, dev, res=1024):
    """gt: assemble.FastGT over the val stems. Post-processes stem by stem (bounded memory)."""
    finals = {}
    for i in range(0, len(va), 8):
        probs = predict_probs(model, va[i:i + 8], dev, res=res)
        for s, p in probs.items():
            finals[s] = postprocess_s1(p.astype(np.float32), s)
    return gt.pq(finals)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = device()
    out_dir = os.path.join(RUNS, args.name)
    os.makedirs(out_dir, exist_ok=True)
    tr, va, readings = split(args.val_fold)
    gt = FastGT(va) if va else None
    ds = S1Train(tr, readings, args.samples, seed=args.seed, res=args.res)
    extra = dict(persistent_workers=True, prefetch_factor=4) if args.workers > 0 else {}
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, drop_last=True, **extra)
    model = build_model(args.encoder).to(dev)
    if args.init:  # fine-tune from an existing checkpoint (e.g. a 1024 model continued at 1536)
        model.load_state_dict(torch.load(args.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(dl)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    log = open(os.path.join(out_dir, "log.txt"), "a")
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"))
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for it, (x, y) in enumerate(dl):
            x, y = x.to(dev), y.to(dev)
            loss = s1_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
        msg = f"ep {ep} loss {run / len(dl):.4f} {time.time() - t0:.0f}s"
        if va and (ep % args.eval_every == 0 or ep == args.epochs):
            pq, info = validate(model, va, gt, dev, args.res)
            msg += f" | val PQ {pq:.4f} tp {info['tp']} fp {info['fp']} fn {info['fn']} sq {info['sq']:.3f}"
            if pq > best:
                best = pq
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
                msg += " *"
        torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    if not va:
        torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))


def predict(args):
    dev = device()
    out_dir = os.path.join(RUNS, args.name)
    cfg = json.load(open(os.path.join(out_dir, "args.json")))
    encoder, res = cfg.get("encoder", args.encoder), cfg.get("res", 1024)
    model = build_model(encoder, pretrained=False).to(dev)
    model.load_state_dict(torch.load(os.path.join(out_dir, args.ckpt), map_location=dev))
    tr, va, _ = split(args.val_fold)
    test = sorted(f[:-4] for f in os.listdir(os.path.join(CACHE, "img1024"))
                  if f.endswith(".npy") and f[:-4] not in set(tr) | set(va))
    stems = {"val": va, "test": test, "train": tr, "all": va + test}[args.predict]
    tag = "tta" if args.tta else "plain"
    pdir = os.path.join(out_dir, f"probs_{tag}")
    os.makedirs(pdir, exist_ok=True)
    for i in range(0, len(stems), 16):
        for s, p in predict_probs(model, stems[i:i + 16], dev, tta=args.tta, res=res).items():
            np.save(os.path.join(pdir, s + ".npy"), (p.astype(np.float32) * 255).round().astype(np.uint8))
        print(f"{min(i + 16, len(stems))}/{len(stems)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1536)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--res", type=int, default=1024, help="working resolution (caches img/fg/un<res> must exist)")
    ap.add_argument("--init", default=None, help="checkpoint to start training from (fine-tuning)")
    a = ap.parse_args()
    predict(a) if a.predict else train(a)
