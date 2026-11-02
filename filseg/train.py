"""Train the U-Net and keep the checkpoint with the best validation PQ.

    python -m filseg.train --data /path/to/MAGFiLO_1.0_Kaggle_2026 --out weights/model.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DataPaths, find_data_root, index_images, load_annotations, split_by_image
from .dataset import FilamentDataset
from .metrics import format_summary, summarize
from .model import bce_dice_loss, build_model, save_checkpoint
from .postprocess import PostParams
from .validate import predict_val_probs, score_probs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder containing train/ and test/")
    p.add_argument("--out", default="weights/model.pt")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--no-pretrained", action="store_true", help="don't download ImageNet weights")
    p.add_argument("--size", type=int, default=1024, help="resolution the network sees (2048 is downscaled)")
    p.add_argument("--crop", type=int, default=512, help="random crop size for training")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--limit", type=int, default=0, help="use only N annotation sets (quick tests)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    paths = DataPaths(find_data_root(args.data))
    items = load_annotations(paths.train_json)
    if args.limit:
        items = items[: args.limit]
    image_paths = index_images(paths.train_images)
    items = [i for i in items if i.stem in image_paths]
    train_items, val_items = split_by_image(items, args.val_frac, args.seed)
    print(f"{len(train_items)} train / {len(val_items)} val annotation sets")

    train_ds = FilamentDataset(train_items, image_paths, args.size, args.crop, True, args.cache_dir)
    val_ds = FilamentDataset(val_items, image_paths, args.size, None, False, args.cache_dir)
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=len(train_ds) > args.batch, pin_memory=device.type == "cuda",
    )

    model = build_model(args.encoder, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.1
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    meta = {
        "encoder": args.encoder,
        "size": args.size,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "val_stems": sorted({i.stem for i in val_items}),
    }
    best_pq, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = bce_dice_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        row = {"epoch": epoch, "loss": total / len(loader), "sec": round(time.time() - t0, 1)}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            probs = predict_val_probs(model, val_ds, device)
            summary = summarize(score_probs(probs, val_items, PostParams()))
            row["val_pq"] = summary["pq"]
            print(f"  val: {format_summary(summary)}")
            if summary["pq"] > best_pq:
                best_pq = summary["pq"]
                save_checkpoint(model, args.out, {**meta, "epoch": epoch, "val": summary})
                print(f"  saved {args.out} (best PQ {best_pq:.4f})")
        history.append(row)
        print(json.dumps(row))

    Path(args.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val PQ {best_pq:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
