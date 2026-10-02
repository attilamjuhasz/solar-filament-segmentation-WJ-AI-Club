# AGENTS.md: how we work on this repo

Competition facts and environment notes are in `CLAUDE.md`. This file covers the agent team, the current pipeline state, and the improvement loop.

## The agent team (`.claude/agents/`)
| Agent | Use it when | Writes code? |
|---|---|---|
| `kaggle-planner` | before building a new model or pipeline variant: research + milestone plan | no |
| `code-reviewer` | after changing `src/`, before trusting a score or uploading | no (reports bugs) |
| `score-optimizer` | new val predictions/candidates exist, or to check a change really optimizes pooled PQ | no (reports params/snippets) |

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
