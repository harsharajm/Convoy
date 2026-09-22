# results/analysis

Analysis of the first batch of successful seed-42 runs, for the results slides.

**Start here for deck generation: `RESULTS_BRIEF.md`** — one self-contained file with
every number, every figure described, a slide plan, and the list of claims the data
does not support. Upload it together with `figures/`.

```
RESULTS_BRIEF.md    self-contained brief for generating the deck — upload this
analyse_runs.py     everything: loads the 4 run JSONs, writes every figure and table
vizstyle.py         palette + matplotlib style shared by the figures
SLIDES.md           slide-by-slide narrative: what to say, and what not to claim
summary_tables.md   every number, paste-ready markdown (generated)
figures/*.png       generated, 200 dpi, light background, 16:9-ish
```

## Running it

```bash
python3 analyse_runs.py
```

Needs `numpy` + `matplotlib`. The system python here is PEP-668 externally
managed, so if `pip install` refuses:

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib && .venv/bin/python analyse_runs.py
```

Reads only `../runs/*_seed42.json` and `../../bdd100k_manifest.csv`. Nothing is
recomputed from weights — no GPU, no dataset needed, runs in about two seconds.

## Figures

| File | Slide | Says |
|---|---|---|
| `fig01_run_design.png` | 1 | the 4 runs × FL / gossip / replay / continual |
| `fig02_partition.png` | 1 backup | 5×3 cell sizes; the 11:1 imbalance and the 44/72-image cells |
| `fig02b_training_budget.png` | 1 backup | sequential trains 34% fewer optimiser steps — the runs are not all step-matched |
| `fig03_headline_map.png` | 2 | the headline mAP, split into the two metrics that actually exist |
| `fig04_detection_metrics.png` | 2 | mAP@[.5:.95] / mAP50 / mAR@100 — the ~1/3 recall ceiling |
| `fig05_per_task.png` | 2 | night is the hard task; the top two night scores sit on 18 and 10 images |
| `fig06_retention.png` | 2 | per-task accuracy after each later task — nothing forgets, in any mode |
| `fig07_forgetting_bwt.png` | 2 | forgetting/BWT are ≤2.8% of the score, shown against the score's own scale |
| `fig08_cross_matrix.png` | 3 | the two 5×5 model-vs-test-set matrices on one shared scale |
| `fig09_matrix_structure.png` | 3 | disagreement-between-clients vs test-set-difficulty, separated |
| `fig10_own_vs_other.png` | 3 | why `global_own_vs_other_gap` is a signed-mean artefact |

## The one thing to get right before reading any chart

The four runs do not all report the same number:

- **M1** task-averaged final mAP — mean of the 3 per-task cell scores.
  gossip, local_only, sequential. **Not joint** (no task boundaries).
- **M2** pooled own-weather mAP — the 3 cells pooled into one loader,
  image-weighted. gossip, local_only, **joint**. **Not sequential**
  (`_collect()` skips the 5×5 for it).

There is no single number that ranks all four. `fig03` is two panels for that
reason. Details and the valid comparisons are in `SLIDES.md`.

## Colour convention

Each run keeps one colour everywhere in the deck, assigned by entity and never
by rank: gossip `#2a78d6` blue, local_only `#eb6834` orange, sequential
`#1baf7a` aqua, joint `#4a3aa7` violet. The 5×5 matrices use a single-hue blue
ramp (sequential encoding — magnitude, so one hue light→dark, never a rainbow).
Red `#e34948` is reserved for status: below-floor test cells and the recall
ceiling.

## Related

- `../metrics_diagnosis.md` — why gossip underperforms local_only, and why
  forgetting stays at 1e-3. The figures here are the visual form of that.
- `../metrics_diagnosis_handoff.md` — the original questions, plus the gate
  measurements from the seed-7 re-run.
