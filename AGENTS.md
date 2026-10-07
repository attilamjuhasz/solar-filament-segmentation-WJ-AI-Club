# AGENTS.md: how we work on this repo

Competition facts and environment notes are in `CLAUDE.md`. This file covers the agent team, the current pipeline state, and the improvement loop.

## Status: running unattended (2026-10-06 ~19:40) — RESUME HERE
Everything below runs DETACHED (nohup) and survives Claude exiting. Nothing needs Claude except judging results.
- **Q2** (`runs/q2.out`): E3a (S2 q-avg) → SEED (S2 seed 1) → E2 (S1 40 ep seed 0, `s1_r34_f0_e40`) → existing S2s on E2 proposals.
- **P1** (`runs/p1.out`, parallel): S1 40 ep seed 1 (`s1_r34_f0_e40_s1`).
- **Q3** (`runs/q3.out`, auto-starts after Q2, then waits for P1), `scripts/exp/q3_after_q2_p1.sh`:
  - A: CSVs for E3a/SEED + the v2+SEED q-ensemble;
  - B: 2-seed S1 ensemble (`src/ens_probs.py`) → S2 v2;
  - C: MaskQ S2 (`s2_r34_mq`);
  - D: E4 1536 fine-tune (`s1_r34_f0_1536`).
  Every result gets a validated CSV in `submissions/` and an `== EVAL` val-PQ line in `runs/q3.out`.
- **2026-10-07 10:30:** two concurrent S1 trainings swapped heavily (8 GB swap full, ~5 MB/s in+out), which likely also caused the overnight E4 stall. **Run at most ONE S1 training at a time on this 16 GB M1.** P2 (3rd seed, ep 13) and Q4 were stopped (low value: +.002 predicted); partial run in `runs/_partial_s1_r34_f0_e40_s2_ep13`. Q5 (folds 1–4) continues alone.
- **2026-10-07 08:30, running:**
  - E4 (Q3 step D, 1536 fine-tune);
  - **P2** (`runs/p2.out`, S1 seed 2, 40 ep, `s1_r34_f0_e40_s2`);
  - **Q4** (`runs/q4.out`, auto after E4 and P2, `scripts/exp/q4_ensembles.sh`): ensembles `ens3_e40` (3 seeds), `ens2_1536` and `ens4_1536`, each scored with S2 v2 and the v2+seed1 q-ensemble, with CSVs.
  - Best so far: `submissions/s2_r34_ens2.csv` (val .4616, LB .37).
- **Picking the best:** grep `EVAL\|^(np.float` in runs/q2.out and runs/q3.out. Compare to v2 = **0.4564** (current best, LB 0.37). Adopt only if better on BOTH split halves: `analysis/score_agent/round2/base.py` / `ana.py` harness. Then update the ledger.
- **7:00 AM Kaggle upload + text** was a session-only cron. If this Claude session died, do it manually in a new session: "read AGENTS.md, pick the best validated CSV, upload it, text me the score" (texting: `bash scripts/local/notify.sh "msg"`).
- **Git:** commit locally on `zaid`. **Do NOT push** until the user says so.
- The Mac is kept awake by `caffeinate` until about 09:30 on 2026-10-07.

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
| 2026-10-01 | `two_stage_v1.csv`: S1 + S2 classifier (full-val tuned params) | 0.456 | 0.37 |
| — | `two_stage_v2.csv`: same S2, assemble_v2 (q×mean_p, lam .225) | 0.4564 (split-half .454/.459) | never uploaded (the LB .37 was v1's) |
| 2026-10-07 | `s2_r34_ens2.csv`: **2-seed S1 ensemble (40 ep, seeds 0+1)** → proposals → S2 v2, assemble_v2 | **0.4616** (halves .4496/.4751 vs v2 .4410/.4739) | 0.37 |

Overnight 2026-10-06/07 results (val PQ, fixed assemble_v2):
- E3a (S2 q-avg) .4524 ✗
- SEED (S2 seed 1) .4558 (≈ v2: the run-to-run noise floor is about ±.001)
- q-ensemble v2+seed .4549 ✗
- MaskQ S2 .4521 ✗
- E2 S1-40ep: S1-only .4289 (vs .4262); S2 on its proposals .4557 / seed1 .4576
- P1 S1-40ep seed 1: S1-only .4302
- **2-seed S1 ensemble: S1-only .4317; with S2 v2 .4616 ✓ (best)**
- M1 fixed-epoch check: ens2 of last.pt .4597 vs ens2 of best.pt .4616 (−.002 = the best.pt selection inflation). best+last 4-model mix .4613 (no gain: same-seed checkpoints are near-identical). → Train folds/all-data for a fixed 40 epochs and use last.pt.
- E4 (1536 fine-tune from E2): S1-only .4296; with S2 v2 .4568 (TP 1149 but FP 575). Neutral vs v2, below ens2. `submissions/s2_r34_1536.csv`.
- Lesson: S1 ensembling is the lever that works; S2/scorer variants are all within noise.

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

## Scorer research (2026-10-06, kaggle-planner; competitor + literature survey)
- **Noise ceiling.** Two annotator readings of the same candidate agree only at Spearman .49, so a perfect E[y] scorer
  reaches ≈ .75 Spearman against realised y on the fold-0 mix. We are at .63. The realistic scorer gain is therefore
  about **+.003–.006 PQ** (perfect ≈ +.01). Bigger levers are S1 (E2, ensembles) and recall.
- The round-2 "29% annotator noise" figure was biased low (ddof=0). Corrected, about **51%** of single-reading q-target
  variance is annotator noise, which makes `--q-avg` more valuable than estimated.
- Built from this:
  - `--qhead mask` (MaskQ: Mask Scoring R-CNN / GFLv2-DGQP style head fed the detached predicted mask);
  - `--wq` (q-loss weight; biconcavelens gained LB +.01 from upweighting their confidence loss);
  - `--q-syn-w`, `--prop-levels`;
  - `assemble.py --extra-cands`: q-ensemble across S2 runs that share proposals. Zero GPU cost. Needs same-recipe runs:
    v2 + E1 averaged gave .4525 < .4564, because E1's q is miscalibrated.
- Competitors:
  - Anon Tokyo's refiner = the same smp aux head as ours, binary quality target, LB .38.
  - biconcavelens (LB .40): biggest lever was upweighting the classification loss; all post-hoc rescorers failed.
- Scripts are in the session scratchpad (`ana/errtype.py`, `ceiling.py`, `context.py`).

## Radical-researcher findings (2026-10-06; scripts in analysis/radical/) — all radical ideas KILLED
- **Expected-PQ decoding gives nothing.** Every kept mask adds exactly +0.5 per reading to the denominator, so "keep iff E[y] > PQ/2" (v2's gate) is optimal. An exact max-weight independent set over overlapping candidates scores .4559 vs greedy .4564. Choosing mask extent per candidate gains nothing either.
- **Annotator-noise ceiling.** Scoring candidates with OTHER annotators' real verdicts (leave-one-reading-out) gives .4408 < model .4533. The ceiling with infinitely many annotators is ≈ +.017 over v2. Max Spearman ≈ .68.
  **The model already predicts a random annotator better than another human does** (human vs human PQ .32–.36).
- **Temporal / cross-site neighbours are negative.** Images are solar-north-up and P-corrected, with east on the left; differential rotation was verified. Labels are frame-specific (seeing), so warped neighbour GT gives PQ ≤ .085, and every score feature built from it lowered PQ.
- **Multi-frame TTA from the public GONG archive is negative.** Kaggle JPEGs = gong2.nso.edu/HA/hag archive frames with overlays removed. A frame 1 min later scores .338 vs .400 S1-only, and fusion is ±0 or worse.
- **Direction:** spend GPU on variance reduction that preserves geometry (S1 seed/fold ensembles, like D4 TTA's +.016) and on candidate quality (E2, E4, E5, E7). Scorer/decoder research is near its ceiling.
- **v2 PQ by annotator group ranges .42–.51.** The unknown test annotator mix likely explains part of the val→LB gap, and that part can't be fixed by modelling.

## Ensemble mechanism and final strategy (2026-10-07, kaggle-planner)
- **The 2-seed S1 ensemble gain is real:** +.0059 vs a single 40-ep seed (bootstrap P(>0) ≈ .99).
  - About 55% comes from better mean_p in the keep score, about 45% from smoother proposals. Recall of candidates is unchanged.
  - S2 does NOT need retraining for ensembles. Retraining on in-sample smooth proposals hurt (E1).
- **More of the same is weak.** Member errors correlate .95–.97: a 3rd seed adds about +.002, 4–5 members about +.003–.0035.
  1536 members are neutral/negative. A single 40-ep S1 alone gives no full-pipeline gain over 16 ep.
- **Plan:**
  - M1: last.pt vs best.pt check, which decides fixed-epoch training.
  - M2: S1 folds 1–4 (13 GPU-h; OOF over 707 stems).
  - M3: S2 on OOF proposals (adopt only if +.003 on both halves).
  - M4: 3 all-data S1 seeds.
  - M5: finals.
- **Finals:**
  - robust = best fold-0 ensemble + S2 v2;
  - aggressive = mean of all S1 members, lam .225 with a guard (test kept/img within 5% of the robust pick, about 85% instance agreement).
- **Kaggle notebook (required):** inference-only, needs a GPU (CPU ≈ 3 h per S1 member), about 30–45 min on T4 for about 11 members. Start building it early.
- **LB is only a bug detector:** 2 decimals, paired SD .003–.005. Pick finals by val/OOF.
- The external NVMe stalled E4 for about 3 h overnight (disk wait). `caffeinate -m` now prevents disk idle sleep.

## Experiment queue (scripts/exp/, logs in runs/*.out)
| Id | What | Status |
|---|---|---|
| E1 | S2 retrained on TTA proposals (`s2_r34_tta`) | **rejected**: .4510 vs .4564 (−.005 ± .002, worse on both halves; its q is miscalibrated at the margin) |
| E3a | S2 with ONLY the q target averaged over readings (`--q-avg`, plain props, 8 ep, `s2_r34_qavg_plain`) | queued (Q2) |
| SEED | v2 S2 recipe with seed 1 (`s2_r34_seed1`): run-to-run noise baseline + 2-seed q ensemble | queued (Q2) |
| E2 | S1 40 epochs (`s1_r34_f0_e40`), then existing S2s scored on its proposals | queued (Q2) |
| Q3 | A: Q2 CSVs + q-ensemble; B: 2-seed S1 ensemble; C: MaskQ S2; D: E4 1536 fine-tune | auto-queued (`runs/q3.out`) |
| Q5 | M1 last-vs-best check, then M2 = S1 folds 1–4 (`s1_r34_f{k}_e40`) | auto after Q3 (`runs/q5.out`) |
| Q6 | M3: OOF S1 maps (`src/oof_probs.py` → `s1_oof`, per-fold CV) → S2 on OOF proposals (`s2_oof`) scored on ens2 val; M4: 3 all-data S1 seeds (`s1_r34_all_e40_s{0,1,2}`, test maps) | auto after Q5 (`runs/q6.out`) |
| P1 | S1 40 epochs, **seed 1** (`s1_r34_f0_e40_s1`), run in PARALLEL with Q2 (user freed the machine): with E2 gives a 2-model S1 probability ensemble | running (`runs/p1.out`) |
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
