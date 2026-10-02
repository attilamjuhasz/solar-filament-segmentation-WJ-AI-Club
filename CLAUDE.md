# temp-kaggle — Solar Filament Segmentation Challenge 2026

Kaggle: https://www.kaggle.com/competitions/filament-segmentation-2026
Python env: `.venv/` (torch + MPS on Apple M1 16GB). Run with `.venv/bin/python`.

## Task
Class-agnostic instance segmentation of solar filaments in 2048×2048 8-bit grayscale GONG
H-alpha full-disk JPEGs. Filaments are thin, dark, elongated structures, often only a few px wide.

## Data (data/MAGFiLO_1.0_Kaggle_2026, CC BY-NC 4.0)
- 707 train JPEGs, 180 test JPEGs, 1 COCO JSON. No sample_submission.
- JSON `images[]` are *readings*: `id = "<annotator>-<stem>"`, so 1,154 readings for 707 files
  (411 files ×1, 145 ×2, 151 ×3). 8,199 polygons, plus `spine` centerline and chirality category (ignored).
- Inter-annotator agreement is low (pairwise PQ ≈ 0.33). Masks are thin, so 1–2 px boundary shifts kill IoU.
- Filaments never touch within one reading (CC of the union recovers the instances at PQ 0.997).
- Disk radius ≈ 900 px. All GT lies inside the disk.
- Same-date near-duplicate frames come from different sites, so validation must be grouped by date.

## Metric (official self-eval notebook)
- PQ = ΣIoU(TP) / (TP + 0.5·FP + 0.5·FN), match at IoU > 0.5, pooled (micro) over **all readings**.
  The same predictions are scored against each annotator's reading separately.
- Predictions for images without GT are ignored.

## Submission
- CSV `filament_id,segmentation_rle`. `filament_id = <stem>_<n>`.
- `segmentation_rle` = pycocotools compressed `counts` string of the Fortran-order 2048×2048 uint8 mask.
- Images with no detections get no rows.
- **Kaggle rejects any submission with overlapping masks.** Enforce pixel ownership.
- 5 submissions/day, 2 final picks. Deadline 2026-11-15 06:00 UTC. Final eligibility also needs a public
  repo, a public Kaggle notebook, a 4-page ACM report, and the Google form.
- Forbidden: the public MAGFiLO 1.0 release (contains the test labels), filament-pretrained weights.
  ImageNet weights are fine.

## Reference scores (other teams, val → LB; LB runs ~0.04–0.06 below val)
- Classical CC 0.18 → 0.15. ResNet34 UNet@1024 0.376 → 0.32. 5-fold + D4 TTA 0.40 → 0.35.
  Mask R-CNN@2048 0.41 → 0.37.
- Lessons from those teams: per-reading targets beat vote-share/union targets; stride-2 encoders (ResNet)
  beat ConvNeXt/MiT on thin structures; D4 TTA ≈ +0.02; choose checkpoints by val PQ, not by loss;
  small filaments (<400 px) are mostly missed.

## M1/MPS notes
- fp16 autocast gives no speedup; channels_last crashes in backward. Use fp32 and contiguous tensors.
- ResNet34-UNet trains at ≈8 img/s at 512 with batch 8.

## Pipeline & results (fold 0 = 141 val stems; see scripts/run_pipeline.sh)
- S1 `src/s1.py` UNet-R34 @1024, 2 heads (one-annotator fg, union). 16 ep ≈ 80 min on M1.
  Val PQ 0.402 plain → 0.418 D4 TTA → 0.426 tuned S1-only postprocess. Public LB 0.35 (val→LB gap ≈ 0.076).
- Proposals `src/proposals.py` levels A/P/B/C. S2 `src/s2.py` per-blob refiner @native res (window→256)
  + q head (E[IoU·1(IoU>.5)]). 8 ep ≈ 45 min.
- Assembly `src/assemble.py` with configs/assemble_v2.json (score q×S1 mean_p, lam 0.225 = PQ/2, levels AP):
  val 0.456, split-half 0.454/0.459. S2's gain comes from its keep/reject score, not sharper masks.
- Don't re-run coordinate-ascent tuning on the full val set: it overfits by about 0.003–0.01. Use split-half or a principled lam ≈ PQ/2.
- Next ideas: train S2 on TTA/OOF proposals (216 FNs are low-q misses), 5-fold S1 ensemble, retrain on all data.
