# AGENTS.md: how we work on this repo

Competition facts and environment notes are in `CLAUDE.md`. This file covers the agent team, the current pipeline state, and the improvement loop.

## Status: running (resumed 2026-10-06 18:15, overnight run)
Queue Q2 restarted (`runs/q2.out`): E3a → SEED → E2. Research agents: radical-researcher, kaggle-planner (scorer), code-reviewer (`--s1-ch`).
If paused again: kill `q2_after_e1.sh` and `src/s2.py`/`src/s1.py`, then relaunch the queue with `nohup bash scripts/exp/q2_after_e1.sh > runs/q2.out 2>&1 &`
(a stopped training run restarts from scratch; move its partial `runs/<name>/` aside first).

## The agent team (`.claude/agents/`)
| Agent | Use it when | Writes code? |
|---|---|---|
| `kaggle-planner` | before building a new model or pipeline variant: research + milestone plan | no |
| `code-reviewer` | after changing `src/`, before trusting a score or uploading | no (reports bugs) |
| `score-optimizer` | new val predictions/candidates exist, or to check a change really optimizes pooled PQ | no (reports params/snippets) |
| `radical-researcher` | incremental tuning has plateaued: proposes radical ideas and falsifies them cheaply on CPU first | no (reports tested ideas) |

The main session owns all edits to `src/`. Agents run in the background, CPU-only while the GPU trains. Relay their findings and verify them before adopting.
Ask for them by name, e.g. "use the code-reviewer agent on src/s2.py".

Previous agent work is saved in `analysis/`:
- `score_agent/official.py`: verbatim port of the official PQ scorer (parity with `src/metric.py` verified)
- `score_agent/*.py, *.log`: calibration, error anatomy, split-half harness, rescorer
- `reviewer/pass{1,2}_*.py`: bug-reproduction checks
- `figures/`: data/annotator overlays, S1 predictions, S2 samples, blob vs final mask

## Pipeline (all PyTorch, `scripts/run_pipeline.sh` runs every step)
```
2048 image → disk fit + 3 planes (intensity, limb-flattened contrast, r/R)
→ S1  UNet-ResNet34 @1024, heads: one-annotator fg / union fg, D4 TTA         (src/s1.py)
→ blobs: hysteresis + connected components, levels A/P (B/C exist but hurt)  (src/proposals.py)
→ S2  per-blob UNet-ResNet34 @native-res window→256, input planes + blob prior;
      outputs exact mask + classifier q ≈ E[IoU·1(IoU>.5)]                     (src/s2.py)
→ assembly: keep_score = q × S1 mean prob, keep ≥ 0.225 (≈PQ/2), area ≥ 100,
      pixel ownership (no overlaps)                                            (src/assemble.py, configs/assemble_v2.json)
→ submissions/*.csv (validated by src/rle.py validate_submission)
```
Weights live on local disk (gitignored): `runs/s1_r34_f0/best.pt` and `runs/s2_r34/last.pt`.
Caches are in `data/cache/`. Rebuild everything with `bash scripts/run_pipeline.sh`.

## Results ledger (fold 0 = 141 val stems / 232 readings)
| Date | Submission | Val PQ | Public LB |
|---|---|---|---|
| 2026-10-01 | `s1_only.csv`: S1 + tuned postprocess | 0.426 | 0.35 |
| 2026-10-01 | `two_stage_v2.csv`: S1 + S2 classifier, q×mean_p, lam .225 | 0.456 (split-half .454/.459) | 0.37 |

Remaining error anatomy (v2 on val):
- FN 584 in total:
  - 216 where some candidate had IoU > .5 but a low score (median q .21);
  - 120 near-misses (IoU .3–.5);
  - 65 with no candidate at all.
- FP 483 in total:
  - 156 near-misses;
  - 121 matching no reading.
- Small GT (<400 px) is break-even.
- S2 masks ≈ S1 masks (pair IoU .88). The S2 gain comes from keep/reject, not shape.

## Research findings (2026-10-02, kaggle-planner)
- **The 0.55 leaderboard cluster is leaked, not modelled.** 54 teams sit at exactly 0.55. A public notebook ("Solar Filament Unet
  Segmentation | 0.55+") embeds a CSV byte-identical to hdjojo/solar-filament-seg-inference, a YOLOv8l-seg @2048 whose
  private weights were probably trained on the forbidden public MAGFiLO 1.0 labels (they include test GT).
  **Never use those notebooks or their outputs.** The honest frontier is ≈0.40–0.41 LB; we are at 0.37.
- Honest competitor numbers:
  - Anon Tokyo: ConvNeXt-T UNet 1024→1536 fine-tune + native crop refiner, 5-fold OOF 0.443 → LB 0.38.
  - YOLO-seg replications: LB 0.35–0.36.
- Our val→LB gap (0.086) is normal for the field (0.05–0.09). Val and test candidate statistics match, and fold 0 is not easier.
- **S1 is under-trained.** It scores the same on train and val stems (0.434 vs 0.427) and val PQ was still rising at epoch 16.
- Measured negatives elsewhere (don't repeat):
  - native 2048 is worse than 1536;
  - dilating masks hurts (−0.02 at 1 px);
  - pseudo-labelling the test set;
  - Mask R-CNN / Mask2Former;
  - clDice and boundary losses give ≈0 here.
- Yardstick on fold 0: +0.01 PQ ≈ 38 recovered FNs, or 70 removed FPs, or 25 fixed near-miss pairs.

## Score-optimizer findings, round 2 (2026-10-02; scripts in analysis/score_agent/round2/)
- **v2 is robust, not over-tuned.** Keep it unchanged:
  - lam peaks at .22–.23 on both halves; a marginal check confirms lam = PQ/2;
  - a_min 50–200 and own_frac .6–.9 are flat.
- **Near-misses are annotator ambiguity.** On those filaments the other annotator's drawing overlaps the GT at median IoU .37, against .59 on TPs.
  - Every shape fix failed on both halves: thresholds .4–.65 are flat; geodesic growth gives −.015 to −.04; union with the S1 blob −.016.
  - grow=1 is **−.05** on our masks, and erosion −.027. **Skip all shape work.**
- **The headroom is the keep/reject scorer.** Perfect keep/reject on the existing candidates reaches .538.
  - Raising the scorer's Spearman from .63 to .72 is worth about +.007; to .79 about +.024.
  - Rescorers with richer features (logistic, isotonic, a numpy GBM, 34 features incl. shape/context/site/year) do **not** beat q×mean_p, so the ranking signal has to come from better training.
  - 29% of the single-reading q-target variance is annotator noise. That motivates `--q-avg`.
- **The val→LB gap is not our pipeline.**
  - Test inputs match val (KS p ≥ .25 on 13 statistics).
  - Reweighting val to the test year/site/annotator mix changes ≤ .002.
  - About .06 of the gap is unexplained on our side: likely the test annotator pool, plus LB noise (SE ≈ .016 on ~90 images).
- Ensembling S2 epochs 4 and 8 hurts. Adding S1-only instances where S2 has no candidate is ±0.

## Experiment queue (scripts/exp/, logs in runs/*.out)
| Id | What | Status |
|---|---|---|
| E1 | S2 retrained on TTA proposals (`s2_r34_tta`) | **rejected**: .4510 vs .4564 (−.005 ± .002, worse on both halves; its q is miscalibrated at the margin) |
| E3a | S2 with ONLY the q target averaged over readings (`--q-avg`, plain props, 8 ep, `s2_r34_qavg_plain`) | queued (Q2) |
| SEED | v2 S2 recipe with seed 1 (`s2_r34_seed1`): run-to-run noise baseline + 2-seed q ensemble | queued (Q2) |
| E2 | S1 40 epochs (`s1_r34_f0_e40`), then existing S2s scored on its proposals | queued (Q2) |
| E4 | S1 fine-tune at 1536 from the best S1 | planned |
| E5 | 5-fold S1 ensemble + OOF (lets lam be re-derived on 707 stems) | planned |
| E6 | OOF stacker (gradient boosting on candidate features) | planned |
| E7 | all-data final retrain | planned |
| E8/E9 | spine aux head split/merge; temporal-neighbour prior | optional |

## Next steps (ranked by expected gain)
1. Retrain S2 on TTA or out-of-fold S1 proposals (it was trained on plain-inference proposals and run on TTA ones). Targets the 216 low-q misses.
2. Ensemble S1 over folds or seeds (other teams saw +0.02 LB from a 5-fold probability ensemble).
3. Retrain S1/S2 on all 707 images with fixed epochs for the final submission.
4. Optional, closer to the original design: make S1 a genuinely rough detector (coarse blobs or boxes) so S2 does the real segmentation. Probably not a score gain, but a cleaner story for the report.
5. Final deliverables by 2026-11-15 06:00 UTC: public repo, public Kaggle notebook running the pipeline, 4-page ACM report, Google form. Pick 2 final submissions.

## Improvement loop
Use `/loop` (self-paced) with a prompt like:
```
/loop Improve the filament pipeline toward higher val PQ. Each iteration: pick the top item from AGENTS.md "Next steps",
plan it with the kaggle-planner agent if it is non-trivial, implement it, train in the background, have the code-reviewer
agent check the diff, have the score-optimizer agent evaluate with split-half, adopt only gains > 0.005 on both halves,
write a validated submission to submissions/, and update the AGENTS.md results ledger. Never upload to Kaggle without asking.
```
Ground rules for every iteration:
- Validate on fold 0. Report the split-half numbers.
- Never tune on the leaderboard (5 uploads/day; it shows 2 decimals and uses about 50% of test).
- Run `validate_submission` before handing over any CSV.
- Keep training on the MPS GPU and agents on the CPU.
