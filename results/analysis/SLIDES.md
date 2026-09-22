# Results slides — seed 42, first successful batch

Three slides as scoped, plus two backup slides you will want in your pocket
because the first question from the room lands on them.

All figures are in `figures/`, all numbers in `summary_tables.md`. Regenerate
both with `python3 analyse_runs.py`.

---

## Slide 1 — What the four runs are

**Figure:** `fig01_run_design.png`

| Run | FL (cross-client) | Gossip mixing | Replay buffer | Continual (3 tasks) |
|---|---|---|---|---|
| **gossip** | ✓ | ✓ | ✓ | ✓ |
| **local_only** | – | – | ✓ | ✓ |
| **sequential** | – | – | – | ✓ |
| **joint** | – | – | n/a | – |

Two things to correct from the original framing of this slide:

1. **There are four runs, not three.** The one not in the original list is
   `local_only` — replay, no gossip. It is the most important row on the
   slide: it is the *only* baseline that differs from `gossip` by exactly one
   switch, so it is the only thing gossip's number can be read against.
   Sequential differs by two switches (and by training budget, see backup A).

2. **Joint does not use replay — your instinct was right, and for the reason
   you gave.** `config.py`: `MODES["joint"] = {"gossip": False, "replay": False}`.
   `run_joint()` pools all 15 (weather × time-of-day) cells into one shuffled
   dataset with no task boundaries. There is no "past task" to replay because
   the shuffle already gives i.i.d. exposure to everything. Mark it `n/a`, not
   ✗ — ✗ would imply a capability that was switched off.

One thing worth saying out loud so the columns don't look redundant: **in this
system FL *is* gossip.** There is no server and no FedAvg, so the two columns
can never disagree. Keep both if the audience expects "FL" as a category; say
the sentence.

---

## Slide 2 — The metric comparison

**Figures:** `fig03_headline_map.png` (headline), then
`fig06_retention.png` + `fig07_forgetting_bwt.png` (the CL metrics).

### Which metric — this has to be the first thing on the slide

The four runs do **not** all report the same number. There are two:

| | **M1** task-averaged final mAP | **M2** pooled own-weather mAP |
|---|---|---|
| what | mean of the client's 3 per-task cell scores | the client's 3 cells pooled into one loader |
| weighting | per **task** (a 10-image cell counts as much as a 2,843-image one) | per **image** |
| where | `per_client.final_mAP` / `system.mAP.mean` | `cross_client.diagonal[c]` / `per_client_pooled_mAP` |
| exists for | gossip, local_only, sequential | gossip, local_only, **joint** |

So there is no single number that ranks all four runs. **Do not put joint's
0.2460 next to sequential's 0.2306** — those are different metrics. The two
defensible statements:

```
M1   local_only 0.2311  ≈  sequential 0.2306  >  gossip 0.2244
M2   joint      0.2460  >  local_only 0.2320  >  gossip 0.2235
```

Headline metric for the deck: **mAP@[.5:.95]**, COCO-style, macro-averaged over
the 10 BDD100K classes. `mAP50` and `mAR@100` are on `fig04` as supporting.

### What the numbers say

- **Gossip loses to local_only**, by −0.0084 mAP (−3.6%) on M2, −0.0067 on M1.
  This is the one clean A/B in the batch — same seed, same schedule, same
  8,043 optimiser steps, one switch different — and the mechanism under test
  is on the wrong side of it.
- **The joint ceiling is barely a ceiling.** +0.0140 mAP (+6.1%) over
  local_only for 11× the pooled data in one model. When an 11× data change
  moves the metric by 0.014, the metric is not resolving what you are asking
  it to resolve.
- **Forgetting and BWT are 1e-3 in every mode** — at most 2.8% of the score
  (`fig07`, right panel puts that on the same axis). `sequential` has **no
  replay buffer at all** and reports forgetting 0.00064 / BWT +0.00251, within
  noise of gossip's 0.00061 / +0.00279.

### The claim NOT to make

Do not say "replay prevented forgetting" or "gossip produced backward
transfer." `fig06` is the reason: the retention curves are flat or rising in
every mode including sequential. **Nothing forgot, so nothing was prevented.**

The mechanism: only `roi_heads.box_predictor` is trainable, and the 3 tasks are
time-of-day splits over the *same 10 classes with the same label semantics*.
"ROI feature → class + box delta" is the same function for daytime, night and
dawn/dusk. Training on night moves the head toward that shared optimum, not
away from it. There is no task-specific mapping to overwrite.

What the ~+0.002 BWT is actually measuring: a few thousand more SGD steps on a
shared head nudging everything up slightly.

### Caveats that belong on this slide, not in the appendix

1. **Two reported cells are noise.** overcast/night = 72 train + **18 test**
   images; partly cloudy/night = 44 train + **10 test**. That is 15 and 9
   optimiser steps of "task". Both are in the top two night scores
   (`fig05`, right). `CLASS_FLOOR = 50` exists in `evaluate_detection.py` to
   catch exactly this, but only drops classes with `N == 0`, so both cells feed
   straight into M1 and into BWT.
2. **Night is the whole story of M1, not the method.** Task 2 averages
   0.199–0.209 against 0.244–0.250 for daytime, in every mode (`fig05`, left).
   The tasks are separated by far more than the runs are.
3. **Recall caps at ~1/3** (`fig04`). mAP50 is 0.44–0.51 while mAR@100 is
   0.30–0.33 — the frozen COCO RPN does not propose BDD100K's small objects
   (55.5% of boxes are under 32px). Same ceiling in all four runs, and it caps
   how far apart any two of them can land.

---

## Slide 3 — Cross-client

**Figures:** `fig08_cross_matrix.png` (the 5×5s), then
`fig09_matrix_structure.png`, then `fig10_own_vs_other.png` if you have room.

Cross-client evaluation only ran for `gossip` and `local_only` — `train.py`
`_collect()` skips the 5×5 for the other two modes.

### The headline: gossip homogenised the clients, downward

Read **down a column** of `fig08`. In gossip, every model scores the same on a
given test set to three decimals. The five clients have become one model. In
local_only the columns spread, and `clear`'s model — 11× the data — is the best
model on *all five* test sets, including the other four clients' own data.

`fig09` left panel is the quantitative version: spread across the five models
collapses from **0.0077 to 0.0010**, a 7.8× reduction. That is gossip doing
exactly what gossip is supposed to do.

The problem is where the consensus landed. **Gossip's consensus scores 0.2009
on clear's test set — the level of local_only's *worst* client (partly cloudy,
0.1974), not its best (clear, 0.2253).** The averaging pulled the best client
down rather than pulling the weak ones up.

Why (from `metrics_diagnosis.md`, one line for the slide): `clear` needs 5,340
steps while `partly cloudy` needs 480, so **223 of clear's 268 gossip
operations (83%) mix it halfway into peers that had already frozen** — nodes
that only give and never take are absorbing boundaries in a pairwise-averaging
network, and the consensus is dragged onto them.

### The correction to make: own_vs_other_gap is not evidence of anything

`fig10`. The JSON's `global_own_vs_other_gap` reads gossip **0.0006** vs
local_only **0.0045** and invites "gossip homogenised the clients." It is a
signed-mean cancellation artefact. Per-client gaps are ±2–3% in **both** runs,
and the mean *absolute* gap is **0.0200 vs 0.0192** — indistinguishable.

What the sign actually tracks is whether your own pooled test set is harder
than average (`fig09` right: every model scores ~0.20 on clear's test set and
~0.24 on overcast's, whichever model it is). That is a property of the
partition, not of gossip. Report `mean |gap|` alongside the signed mean, and
read homogenisation off the column spread instead.

---

## Backup A — the runs are not all step-matched

**Figure:** `fig02b_training_budget.png`

gossip and local_only run **8,043** optimiser steps each. `sequential` runs
**5,304** — 34% fewer, and 40% fewer for `clear`. Cause: with no buffer,
`use_replay` is false, so `cur_bs` stays at `BATCH_SIZE=32` instead of dropping
to 16 for tasks 1–2 (`train.py:146`). Same images, half the optimiser steps.

Consequence: "sequential 0.2306 ≈ local_only 0.2311, so replay bought nothing"
is **not** a controlled read — sequential matched it on a third fewer steps.
Have this ready; it is the first thing a careful listener will ask.

## Backup B — the partition

**Figure:** `fig02_partition.png`

`clear` holds 61% of all training images. Two night cells are 72 and 44 train
images. This is upstream of every anomaly on slide 2 and of the 83% figure on
slide 3.

---

## One thing this batch cannot tell you

**The gate was never recorded.** `_gossip()` in `train.py` discarded
`gossip_round()`'s return value, so these four runs contain zero evidence of
what the gate actually did — 406 gossip operations, no alpha logged. The
instrumentation landed after this batch. The seed-7 re-run measures
`alpha_mean = 0.4882` against `lambda = 0.5`, i.e. the gate passes essentially
everything at near-full strength and the run is functionally plain pairwise
averaging.

If a slide claims the gate contributed anything, it is claiming something this
batch did not measure.
