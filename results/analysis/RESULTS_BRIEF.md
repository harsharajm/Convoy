# Results brief — Decentralised Gossip FCL for Object Detection (seed 42, first successful batch)

**This file is self-contained.** Every number needed to build a deck is inline below;
nothing has to be looked up elsewhere. The 11 PNGs in `figures/` are referenced by
exact filename at the point where each one belongs.

**Purpose:** source material for generating a results presentation. A proposed slide
plan is in §9. Sections 1–8 are the content; §10 lists the claims that the data does
**not** support, which must not appear on any slide.

---

## 0. Setup in one paragraph

Decentralised gossip-based federated **continual** learning for object detection on
BDD100K. Faster R-CNN with a COCO-pretrained ResNet50+FPN backbone; **backbone and RPN
are frozen**, and only `roi_heads.box_predictor` trains — `cls_score` (1024→11) and
`bbox_pred` (1024→44), for 10 BDD100K classes plus background. There is **no server and
no FedAvg**: clients exchange head weights pairwise, gated by how well a neighbour's
weights score on the client's own replay memory.

- **Clients = 5 weather conditions:** clear, overcast, rainy, snowy, partly cloudy
- **Tasks = 3 time-of-day splits,** trained sequentially per client: daytime, night, dawn/dusk
- **Data:** 56,347 train / 14,079 test images across the 5×3 grid
- **Key hyperparameters:** `GOSSIP_LAMBDA=0.5`, `GATE_SENSITIVITY=2.0`,
  `GOSSIP_EVERY_N_STEPS=20`, `EPOCHS_PER_TASK=3`, `BATCH_SIZE=32`, `LR_HEAD=0.01`,
  `MOMENTUM=0.9`, `BUFFER_CAPACITY=2000`, `REPLAY_RATIO=1.0` (1:1 current:replay),
  no LR warm-up, no LR schedule, no augmentation
- **Source of every number below:** `results/runs/{mode}_seed42.json`, regenerated into
  figures and tables by `results/analysis/analyse_runs.py`

---

## 1. The four runs

There are **four** runs, not three. Defined in `config.py`:

```python
MODES = {
    "gossip":     {"gossip": True,  "replay": True},
    "local_only": {"gossip": False, "replay": True},
    "sequential": {"gossip": False, "replay": False},
    "joint":      {"gossip": False, "replay": False},   # all data, no CL
}
```

| Run | FL (cross-client) | Gossip mixing | Replay buffer | Continual (3 tasks) | What it isolates |
|---|---|---|---|---|---|
| **gossip** | Yes | Yes | Yes | Yes | the full method |
| **local_only** | No | No | Yes | Yes | replay alone — the baseline gossip must beat |
| **sequential** | No | No | No | Yes | naive CL — what forgetting looks like unprotected |
| **joint** | No | No | **n/a** | No | ceiling: all data pooled and shuffled, one model |

**Three points that must be made on this slide:**

1. **Joint uses no replay, and the reason matters.** `run_joint()` pools all 15
   (weather × time-of-day) cells into a single shuffled dataset with no task
   boundaries. There is no "past task" to replay because the shuffle already gives
   i.i.d. exposure to everything. Mark the cell **n/a**, not ✗ — ✗ would imply a
   capability that was deliberately switched off.

2. **`local_only` is the row that matters.** It is the only baseline that differs from
   `gossip` by exactly one switch, so it is the only thing gossip's number can honestly
   be read against. `sequential` differs by two switches *and* by training budget (§3).

3. **In this system, FL *is* gossip.** There is no server and no FedAvg, so the "FL"
   and "Gossip mixing" columns can never disagree. Keep both columns if the audience
   expects "FL" as a category, but say this sentence out loud so they don't look like
   redundant filler.

**Figure:** `figures/fig01_run_design.png` — the same table rendered as an image
(4 rows × ✓/– columns + a "what it isolates" column). Use the markdown table above if
the deck wants a native editable table; use the PNG if it wants a picture.

---

## 2. The data partition

| train images | daytime | night | dawn/dusk | **total** |
|---|---|---|---|---|
| clear | 11,375 | 20,927 | 1,849 | **34,151** |
| overcast | 6,872 | **72** | 1,064 | **8,008** |
| rainy | 2,335 | 1,996 | 307 | **4,638** |
| snowy | 2,628 | 2,018 | 408 | **5,054** |
| partly cloudy | 3,920 | **44** | 532 | **4,496** |

| test images | daytime | night | dawn/dusk | **total** |
|---|---|---|---|---|
| clear | 2,843 | 5,231 | 462 | **8,536** |
| overcast | 1,718 | **18** | 265 | **2,001** |
| rainy | 583 | 498 | **76** | **1,157** |
| snowy | 656 | 504 | 102 | **1,262** |
| partly cloudy | 980 | **10** | 133 | **1,123** |

- `clear` holds **61%** of all training images — an 11:1 imbalance against the smallest client.
- **overcast/night = 72 train + 18 test.  partly cloudy/night = 44 train + 10 test.**
  These are the origin of almost every anomaly downstream.

**Figure:** `figures/fig02_partition.png` — two heatmaps (train, test) of the 5×3 grid
with counts in each cell; cells under 100 images outlined in red.

---

## 3. Training budget — the runs are NOT all step-matched

| client | train imgs | gossip steps | local_only steps | sequential steps | gossip ops |
|---|---|---|---|---|---|
| clear | 34,151 | 5,340 | 5,340 | 3,204 | 268 |
| overcast | 8,008 | 861 | 861 | 756 | 44 |
| rainy | 4,638 | 654 | 654 | 438 | 33 |
| snowy | 5,054 | 708 | 708 | 480 | 36 |
| partly cloudy | 4,496 | 480 | 480 | 426 | 25 |
| **total** | **56,347** | **8,043** | **8,043** | **5,304** | **406** |

`joint` runs **5,283** steps as a single model over the pooled set (not a sum over clients).
Scheduler slices: gossip 269, local_only 269, sequential 162.

**gossip and local_only are step-matched (8,043 each). `sequential` is not — 5,304, or
34% fewer, and 40% fewer for `clear`.** Cause: with no buffer, `use_replay` is false, so
`cur_bs` stays at `BATCH_SIZE=32` instead of dropping to 16 for tasks 1–2
(`train.py:146`). Same images, half the optimiser steps.

**Consequence:** "sequential 0.2306 ≈ local_only 0.2311, therefore replay bought
nothing" is **not** a controlled read — sequential matched it on a third fewer steps.
The one clean A/B in this batch is **gossip vs local_only**.

**Figure:** `figures/fig02b_training_budget.png` — grouped bars of steps per client per
mode, plus a horizontal bar of total budget per run.

---

## 4. Which metric — read this before any results slide

The four runs do **not** all report the same number. There are two, and they are not
interchangeable.

| | **M1 — task-averaged final mAP** | **M2 — pooled own-weather mAP** |
|---|---|---|
| What | Mean of the client's 3 per-task cell scores, measured after all 3 tasks finish | The client's 3 test cells pooled into one loader |
| Weighting | **Per task** — a 10-image cell counts as much as a 2,843-image one | **Per image** |
| In the JSON | `per_client.final_mAP` / `system.mAP.mean` | `cross_client.diagonal[c]` / `per_client_pooled_mAP` |
| Exists for | gossip, local_only, sequential — **not joint** (no task boundaries) | gossip, local_only, joint — **not sequential** (`_collect()` skips the 5×5 for it) |

**There is therefore no single number that ranks all four runs.** The two defensible
statements are:

```
M1    local_only 0.2311  ≈  sequential 0.2306  >  gossip 0.2244
M2    joint      0.2460  >  local_only 0.2320  >  gossip 0.2235
```

**Headline metric for the deck:** mAP@[.5:.95], COCO-style, macro-averaged over the 10
BDD100K classes. mAP50 and mAR@100 are supporting (§5.3).

---

## 5. Results

### 5.1 M1 — task-averaged final mAP@[.5:.95]

| client | gossip | local_only | sequential |
|---|---|---|---|
| clear | 0.2098 | 0.2315 | 0.2272 |
| overcast | 0.2538 | 0.2570 | 0.2607 |
| rainy | 0.2101 | 0.2139 | 0.2102 |
| snowy | 0.2215 | 0.2262 | 0.2253 |
| partly cloudy | 0.2266 | 0.2269 | 0.2298 |
| **MEAN** | **0.2244** | **0.2311** | **0.2306** |
| std across clients | 0.0161 | 0.0142 | 0.0165 |

### 5.2 M2 — pooled own-weather mAP@[.5:.95]

| client | gossip | local_only | joint |
|---|---|---|---|
| clear | 0.2009 | 0.2253 | 0.2281 |
| overcast | 0.2440 | 0.2506 | 0.2628 |
| rainy | 0.2089 | 0.2135 | 0.2320 |
| snowy | 0.2213 | 0.2269 | 0.2447 |
| partly cloudy | 0.2425 | 0.2435 | 0.2624 |
| **MEAN** | **0.2235** | **0.2320** | **0.2460** |

**The two findings on this slide:**

- **Gossip loses to local_only** — by **−0.0084 mAP (−3.6%)** on M2, and −0.0067 on M1.
  This is the one clean A/B in the batch: same seed, same schedule, same 8,043
  optimiser steps, one switch different. The mechanism under test is on the wrong side
  of its own control.
- **The joint ceiling is barely a ceiling** — **+0.0140 mAP (+6.1%)** over local_only,
  for 11× the pooled data trained in a single model. When an 11× data change moves the
  metric by 0.014, the metric is not resolving what we are asking it to resolve.

**Figure:** `figures/fig03_headline_map.png` — two side-by-side grouped bar panels (M1
on the left over 3 runs, M2 on the right over 3 runs), per client plus a MEAN group.
**This is the single most important figure in the deck.**

### 5.3 mAP50 and mAR@100 — the recall ceiling

Recorded for gossip and local_only only (M2 split).

| metric | gossip | local_only |
|---|---|---|
| mAP@[.5:.95] | 0.2235 | 0.2320 |
| mAP50 | 0.4644 | 0.4857 |
| mAR@100 | 0.3181 | 0.3214 |

**mAP50 is 0.44–0.51 while mAR@100 is 0.30–0.33.** The head localises what it is shown,
but the **frozen COCO RPN proposes only about one third of BDD100K's boxes** — 55.5% of
them are under 32px, well below what COCO's anchors were tuned for. That ceiling is
identical in all four runs, and it caps how far apart any two of them can possibly land.

**Figure:** `figures/fig04_detection_metrics.png` — three panels (mAP@[.5:.95], mAP50,
mAR@100), gossip vs local_only per client, with a red line at 1/3 on the mAR panel.

### 5.4 Per-task breakdown — night is the story, not the method

Final mAP per task, averaged over the 5 clients:

| task | gossip | local_only | sequential |
|---|---|---|---|
| task 1 · daytime | 0.244 | 0.250 | 0.247 |
| task 2 · night | 0.199 | 0.207 | 0.209 |
| task 3 · dawn/dusk | 0.230 | 0.236 | 0.236 |

**The tasks are separated by ~0.042 mAP; the runs are separated by ~0.007.** M1 mostly
reports how much night is in the mix.

Per-client night scores (local_only), against the cell sizes they were measured on:

| client | night mAP | train imgs | test imgs |
|---|---|---|---|
| **overcast** | **0.275** | **72** | **18** ⚠ |
| **partly cloudy** | **0.211** | **44** | **10** ⚠ |
| clear | 0.199 | 20,927 | 5,231 |
| snowy | 0.194 | 2,018 | 504 |
| rainy | 0.157 | 1,996 | 498 |

**The two best night scores in the entire run are measured on 18 and 10 images**, from
tasks that were 15 and 9 optimiser steps long. `CLASS_FLOOR = 50` exists in
`evaluate_detection.py` to catch exactly this, but it only filters classes where
`N == 0` — it does not exclude below-floor classes from the macro average. So both cells
feed straight into M1 and into BWT. **Treat them as noise, not as results.**

**Figure:** `figures/fig05_per_task.png` — left: grouped bars of final mAP per task per
mode; right: horizontal bars of night mAP per client with cell sizes labelled, the two
sub-50 cells in red.

### 5.5 Continual-learning metrics — nothing forgot, in any run

| client | gossip F | gossip BWT | local_only F | local_only BWT | sequential F | sequential BWT |
|---|---|---|---|---|---|---|
| clear | 0.00203 | −0.00174 | 0.00000 | +0.00349 | 0.00311 | −0.00311 |
| overcast | 0.00030 | −0.00022 | 0.00243 | −0.00243 | 0.00000 | +0.00558 |
| rainy | 0.00006 | +0.00632 | 0.00000 | +0.00438 | 0.00000 | +0.00258 |
| snowy | 0.00064 | +0.00516 | 0.00157 | +0.00222 | 0.00007 | +0.00195 |
| partly cloudy | 0.00000 | +0.00446 | 0.00000 | +0.00312 | 0.00000 | +0.00557 |
| **MEAN** | **0.00061** | **+0.00279** | **0.00080** | **+0.00215** | **0.00064** | **+0.00251** |

**The largest |forgetting| or |BWT| anywhere in any run is 0.0063 — 2.8% of the score
itself**, and within the run-to-run noise of a 10-to-18-image test cell.

**`sequential` has no replay buffer at all** and reports forgetting 0.00064 / BWT
+0.00251 — statistically indistinguishable from gossip's 0.00061 / +0.00279. So this
number is not measuring backward transfer. It is measuring *"a few thousand more SGD
steps on a shared head nudge everything up slightly."*

**Why there is no forgetting to measure:** only `roi_heads.box_predictor` trains, and
the three tasks are time-of-day splits over the **same 10 classes with the same label
semantics**. The mapping "ROI feature → class + box delta" is the *same function* for
daytime, night and dawn/dusk. Training on night moves the head toward that shared
optimum, not away from it. There is no task-specific mapping to overwrite — so there is
nothing for replay to prevent and nothing for gossip's BWT to repair.

Retention data (task 1 = daytime, mAP re-measured after each later task, mean over clients):

| | after task 1 | after task 2 | after task 3 |
|---|---|---|---|
| gossip | 0.2382 | 0.2429 | 0.2436 |
| local_only | 0.2454 | 0.2489 | 0.2501 |
| sequential | 0.2452 | 0.2393 | 0.2466 |

Every mode ends **at or above** where it started. A forgetting curve slopes down; none
of these do.

**Figures:**
- `figures/fig06_retention.png` — three small-multiple panels (one per task), thin lines
  per client and a thick mean line per mode, showing the flat/rising curves.
- `figures/fig07_forgetting_bwt.png` — dot plots of forgetting and BWT per client per
  mode, plus a third panel putting the largest value on the same axis as the score
  itself (0.0063 against 0.2287).

---

## 6. Cross-client evaluation

Cross-client evaluation ran for **gossip and local_only only** — `train.py::_collect()`
skips the 5×5 for the other two modes. Each cell is model *i* scored on client *j*'s
pooled test set.

### 6.1 gossip — 5×5 mAP@[.5:.95] (row = model, column = test set)

| model \ test | clear | overcast | rainy | snowy | partly cloudy |
|---|---|---|---|---|---|
| **clear** | *0.2009* | 0.2416 | 0.2071 | 0.2196 | 0.2414 |
| **overcast** | 0.2007 | *0.2440* | 0.2071 | 0.2200 | 0.2447 |
| **rainy** | 0.2022 | 0.2444 | *0.2089* | 0.2214 | 0.2433 |
| **snowy** | 0.2014 | 0.2437 | 0.2083 | *0.2213* | 0.2437 |
| **partly cloudy** | 0.2000 | 0.2424 | 0.2061 | 0.2188 | *0.2425* |
| *spread across models* | *0.0007* | *0.0011* | *0.0010* | *0.0010* | *0.0011* |
| *test-set difficulty (col mean)* | *0.2010* | *0.2432* | *0.2075* | *0.2202* | *0.2431* |

Global cross-client mAP: **0.2229**

### 6.2 local_only — 5×5 mAP@[.5:.95]

| model \ test | clear | overcast | rainy | snowy | partly cloudy |
|---|---|---|---|---|---|
| **clear** | *0.2253* | 0.2591 | 0.2271 | 0.2395 | 0.2588 |
| **overcast** | 0.2055 | *0.2506* | 0.2112 | 0.2240 | 0.2523 |
| **rainy** | 0.2056 | 0.2443 | *0.2135* | 0.2243 | 0.2431 |
| **snowy** | 0.2033 | 0.2433 | 0.2092 | *0.2269* | 0.2394 |
| **partly cloudy** | 0.1974 | 0.2428 | 0.2025 | 0.2165 | *0.2435* |
| *spread across models* | *0.0094* | *0.0062* | *0.0081* | *0.0075* | *0.0071* |
| *test-set difficulty (col mean)* | *0.2074* | *0.2480* | *0.2127* | *0.2262* | *0.2474* |

Global cross-client mAP: **0.2275**

### 6.3 What the matrices say

**Read DOWN a column.**

- **In gossip, every model scores the same on a given test set to three decimals.** The
  five clients have become one model. Spread across models collapses from **0.0077 to
  0.0010 — a 7.8× reduction.** That part of the mechanism demonstrably worked.
- **In local_only the columns spread**, and `clear`'s model — 11× the data — is the best
  model on *all five* test sets, including the other four clients' own data.
- **But the consensus landed low.** Gossip scores **0.2009** on clear's test set — the
  level of local_only's **worst** client (partly cloudy, 0.1974), not its best (clear,
  0.2253). The averaging pulled the best client **down** instead of lifting the weak
  ones up.
- **Cause:** `clear` needs 5,340 optimiser steps while `partly cloudy` needs 480, so
  **223 of clear's 268 gossip operations (83%) mix it halfway into peers that had
  already frozen.** In a pairwise-averaging network, nodes that only give and never take
  are absorbing boundary conditions — the consensus is dragged onto them. The gate
  cannot stop it, because `gate_weight` measures *quality*, not *staleness*: a frozen
  peer that completed its whole curriculum is a decent model, so it passes at near-full
  strength precisely because it is decent.

**Figures:**
- `figures/fig08_cross_matrix.png` — both 5×5 matrices side by side on one shared colour
  scale, diagonal cells outlined. **The headline cross-client figure.**
- `figures/fig09_matrix_structure.png` — left: spread across models per test set (the
  homogenisation evidence, 0.0077 → 0.0010); right: column means (test-set difficulty,
  nearly identical in both runs).

### 6.4 Correction — `own_vs_other_gap` is not evidence of anything

| | clear | overcast | rainy | snowy | partly cloudy | signed mean | **mean abs** |
|---|---|---|---|---|---|---|---|
| gossip | −0.0265 | +0.0259 | −0.0189 | −0.0030 | +0.0257 | +0.00063 | **0.0200** |
| local_only | −0.0208 | +0.0273 | −0.0158 | +0.0031 | +0.0287 | +0.00450 | **0.0192** |

The JSON's `global_own_vs_other_gap` reads **gossip 0.0006 vs local_only 0.0045** and
invites the conclusion "gossip homogenised the clients." **It is a signed-mean
cancellation artefact.** The per-client gaps are ±2–3% in *both* runs, and the mean
**absolute** gap is **0.0200 vs 0.0192 — indistinguishable.**

What the sign actually tracks is whether your own pooled test set is harder than average
(clear and rainy negative, overcast and partly cloudy positive) — a property of the
**data partition**, not of gossip. Note also that **no client's model is best on its own
data**: the diagonal is not special in either run.

**Recommendation:** report `mean |gap|` alongside the signed mean, and read
homogenisation off the column spread (§6.3) instead.

**Figure:** `figures/fig10_own_vs_other.png` — left: per-client signed gaps for both
runs; right: signed mean vs mean absolute, showing the artefact.

---

## 7. Caveats that must appear in the deck

1. **Two reported cells are noise.** overcast/night = 72 train + 18 test images;
   partly cloudy/night = 44 train + 10 test — 15 and 9 optimiser steps of "task".
   `CLASS_FLOOR = 50` only drops classes with `N == 0`, so both feed into M1 and BWT.
2. **M1 and M2 are not interchangeable.** joint 0.2460 vs sequential 0.2306 is not a
   comparison. No metric in this batch ranks all four runs.
3. **Only gossip vs local_only is step-matched.** sequential trains 5,304 steps against
   their 8,043 (34% fewer). joint is a different code path entirely.
4. **Nothing forgot.** Forgetting and BWT are ≤2.8% of the score in every mode,
   sequential included. Forgetting requires convergence to something task-specific
   first; this head has no task-specific function to overwrite.
5. **Recall caps at ~1/3 in every run.** The frozen COCO RPN does not propose BDD100K's
   small objects. Same ceiling in all four runs.
6. **The head never converged.** From the joint run's training log, mean loss by
   1000-step window: 1.261 → 1.169 → 1.177 → 1.254 → 1.251. **The loss does not descend
   at all after ~step 100.** `LR_HEAD=0.01` with `MOMENTUM=0.9` and no warm-up and no
   schedule gives an effective LR of ~0.1 on a randomly-initialised head; torchvision's
   own Faster R-CNN recipe is lr=0.02 at batch 16 *with* a 1000-iteration linear warm-up
   and step decay. A model sitting at a noise-dominated equilibrium has nothing to
   forget, and all four configurations converge to the same number for that reason.
7. **The gate was never recorded in this batch.** `_gossip()` in `train.py` discarded
   `gossip_round()`'s return value — 406 gossip operations, zero alphas logged. These
   four runs contain no evidence of what their own central mechanism did.
   Instrumentation landed *after* this batch; the seed-7 re-run measures
   `alpha_mean = 0.4882` against `lambda = 0.5` (`alpha_std 0.036`, `frac_at_lam 0.33`),
   i.e. the gate passes essentially everything at near-full strength and the run is
   functionally **plain pairwise averaging**. Root cause: `gate_loss()` sums the entire
   Faster R-CNN loss dict including `loss_objectness` and `loss_rpn_box_reg`, which come
   from the frozen client-identical RPN and are a shared constant — they inflate the
   denominator of `excess = (loss_neighbour − loss_own)/loss_own` while contributing
   zero signal, leaving a mean neighbour-vs-own loss difference of **0.62%**.

---

## 8. Figure manifest

All in `figures/`, 200 dpi PNG, white background, sized for 16:9.

| File | Section | What it shows |
|---|---|---|
| `fig01_run_design.png` | §1 | The 4 runs × FL / gossip / replay / continual, + "what it isolates" |
| `fig02_partition.png` | §2 | 5×3 train and test cell sizes; sub-100 cells outlined red |
| `fig02b_training_budget.png` | §3 | Optimiser steps per client per mode + total budget per run |
| `fig03_headline_map.png` | §5.1–5.2 | **Headline.** M1 and M2 as two grouped-bar panels, per client + MEAN |
| `fig04_detection_metrics.png` | §5.3 | mAP@[.5:.95] / mAP50 / mAR@100, with the 1/3 recall ceiling marked |
| `fig05_per_task.png` | §5.4 | Final mAP per task per mode; night mAP vs cell size, tiny cells in red |
| `fig06_retention.png` | §5.5 | Per-task accuracy re-measured after each later task — the flat curves |
| `fig07_forgetting_bwt.png` | §5.5 | Forgetting and BWT dot plots + the same values against the score's scale |
| `fig08_cross_matrix.png` | §6.1–6.3 | **Headline.** Both 5×5 matrices on one shared colour scale |
| `fig09_matrix_structure.png` | §6.3 | Spread-across-models vs test-set-difficulty, separated |
| `fig10_own_vs_other.png` | §6.4 | Why `global_own_vs_other_gap` is a signed-mean artefact |

**Colour convention used throughout** (each run keeps one colour in every figure,
assigned by entity and never by rank):

| Run | Hex |
|---|---|
| gossip | `#2A78D6` blue |
| local_only | `#EB6834` orange |
| sequential | `#1BAF7A` aqua |
| joint | `#4A3AA7` violet |

The 5×5 matrices use a single-hue blue ramp (`#CDE2FB` → `#0D366B`). Red `#E34948` is
reserved for status only: below-floor test cells and the recall ceiling.

---

## 9. Proposed slide plan

| # | Slide | Content | Figure |
|---|---|---|---|
| 1 | Title | Setup from §0 | — |
| 2 | The four runs | §1 table + the three points | `fig01_run_design.png` (optional) |
| 3 | Which metric? | §4 table + the two defensible readings | — |
| 4 | Results at a glance | §5.1 and §5.2 tables side by side | — |
| 5 | **Headline comparison** | §5.2 two findings | **`fig03_headline_map.png`** |
| 6 | The recall ceiling | §5.3 | `fig04_detection_metrics.png` |
| 7 | Per-task breakdown | §5.4 | `fig05_per_task.png` |
| 8 | Nothing forgets | §5.5 retention + the mechanism | `fig06_retention.png` |
| 9 | Forgetting & BWT are noise | §5.5 table + the sequential comparison | `fig07_forgetting_bwt.png` |
| 10 | **Cross-client 5×5** | §6.3 | **`fig08_cross_matrix.png`** |
| 11 | What the matrix is made of | §6.3 | `fig09_matrix_structure.png` |
| 12 | Correction: own-vs-other gap | §6.4 | `fig10_own_vs_other.png` |
| 13 | Caveats | §7 | — |
| 14 | Backup: training budget | §3 | `fig02b_training_budget.png` |
| 15 | Backup: the partition | §2 | `fig02_partition.png` |

Slides 2–4 are the "explain the runs and the metric" block; 5–9 are the metric
comparison; 10–12 are cross-client.

---

## 10. Claims the data does NOT support — do not put these on a slide

- ❌ "Replay prevented forgetting." — sequential has no buffer and forgets no more than
  gossip. Nothing forgot, so nothing was prevented.
- ❌ "Gossip produced backward transfer." — sequential's BWT (+0.0025) and gossip's
  (+0.0028) are indistinguishable, and sequential has neither gossip nor replay.
- ❌ "Gossip homogenised the clients, as shown by the smaller own_vs_other gap." — that
  metric is a signed-mean artefact (§6.4). Gossip *did* homogenise the clients, but the
  evidence is the column spread, not this number.
- ❌ "Joint is the upper bound at 0.2460, sequential reaches 0.2306." — different
  metrics (§4).
- ❌ "Sequential ≈ local_only, so replay adds nothing." — not step-matched (§3).
- ❌ Anything attributing an effect to the **gate**. This batch logged zero gate
  measurements (§7.7).
- ❌ Any conclusion about continual learning drawn from overcast/night or partly
  cloudy/night. Those cells are 18 and 10 test images (§5.4).

---

## 11. Reproducing everything here

```bash
cd OBJ_DET/results/analysis
python3 analyse_runs.py          # regenerates figures/*.png and summary_tables.md
```

Reads only `../runs/*_seed42.json` and `../../bdd100k_manifest.csv`. No GPU, no dataset,
~2 seconds. Needs numpy + matplotlib; if the system python is PEP-668 externally managed:

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib && .venv/bin/python analyse_runs.py
```

Companion files: `summary_tables.md` (the same tables, generated), `SLIDES.md` (narrative
form), `README.md` (orientation), `../metrics_diagnosis.md` (the full root-cause analysis
these results rest on).
