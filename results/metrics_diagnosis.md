# Diagnosis: seed-42 baseline metrics

Answers to the two questions in `metrics_diagnosis_handoff.md`. Every claim
below is checked against `results/runs/*.json` and `results/logs/*.log`, not
inferred from the code alone.

---

## (a) Why `gossip` underperforms `local_only`

### Primary cause: 83% of `clear`'s gossip operations mix it halfway into permanently frozen, under-trained peers

The cells are 11:1 imbalanced (from `bdd100k_manifest.csv`):

| client | daytime | night | dawn/dusk | total train |
|---|---|---|---|---|
| clear | 11,375 | 20,927 | 1,849 | 34,151 |
| overcast | 6,872 | **72** | 1,064 | 8,008 |
| rainy | 2,335 | 1,996 | 307 | 4,638 |
| snowy | 2,628 | 2,018 | 408 | 5,054 |
| partly cloudy | 3,920 | **44** | 532 | 4,496 |

`run()` round-robins over *active* clients, so `clear` needs 5,340 optimiser
steps while `partly cloudy` needs 480. From `gossip_seed42_*.log`:

| event | log line |
|---|---|
| partly cloudy finishes all 3 tasks | 420 |
| rainy finishes all 3 tasks | 524 |
| snowy finishes all 3 tasks | 556 |
| overcast finishes all 3 tasks | 610 |
| **clear finishes its FIRST task** | **645** |
| clear finishes all 3 tasks | 1508 |

`clear` trains tasks 1 and 2 — 4,272 of its 5,340 steps — with all four peers
permanently frozen. `sample_partner()` keeps finished clients eligible as
sources (`gossip.py:133`, per design_choices.pdf 2.2.1) and `gossip_round()`
still mixes into the active side (`if not client_a["finished"]`), so every
20 steps `clear` executes:

```
clear ← 0.5 · clear + 0.5 · frozen_peer
```

Counted directly from the log: **223 of clear's 268 gossip ops (83%) happened
after every peer had frozen.**

That is a hard anchor. `clear` can never drift further than ~20 steps of
progress from the mean of four models that stopped learning hours earlier and
were under-trained to begin with (480–861 steps each).

**The 5×5 matrix is the proof** (row = model, col = test set, mAP@[.5:.95]):

```
gossip                                      local_only
           clear  overc  rainy  snowy  pcl             clear  overc  rainy  snowy  pcl
clear     0.2009 0.2416 0.2071 0.2196 0.2414  clear    0.2253 0.2591 0.2271 0.2395 0.2588
overcast  0.2007 0.2440 0.2071 0.2200 0.2447  overcast 0.2055 0.2506 0.2112 0.2240 0.2523
rainy     0.2022 0.2444 0.2089 0.2214 0.2433  rainy    0.2056 0.2443 0.2135 0.2243 0.2431
snowy     0.2014 0.2437 0.2083 0.2213 0.2437  snowy    0.2033 0.2433 0.2092 0.2269 0.2394
pcloudy   0.2000 0.2424 0.2061 0.2188 0.2425  pcloudy  0.1974 0.2428 0.2025 0.2165 0.2435
```

- **gossip**: every row is identical to three decimals. The five clients are
  one model.
- **local_only**: `clear`'s model is the best model on *every* test set,
  including the other four clients' — it had 11× the data. `partly cloudy`
  (480 steps) is the worst everywhere.
- The gossip consensus lands at **0.2010** on clear's test set, i.e. at the
  level of local_only's *worst* client (partly cloudy, 0.1974), not its best
  (clear, 0.2253).

So this is not "averaging already-similar clients." It is the one client with
11× the data being dragged down to the level of the least-trained one, for
83% of its training run.

### Contributing cause: the gate is a no-op

Two independent reasons `gate_weight()` returns ≈1 always:

1. **The RPN terms pollute the denominator.** `gate_loss()` returns
   `sum(losses.values())`, which includes `loss_objectness` and
   `loss_rpn_box_reg`. The backbone and RPN are frozen and bit-identical
   across clients, so those two terms are a shared constant. In
   `excess = (L_n − L_o) / L_o` the constant cancels in the numerator but
   *inflates the denominator*, systematically shrinking `excess` and pushing
   the gate toward 1.
2. **The clients are too similar for `GATE_SENSITIVITY=2.0` to bite.** Five
   heads trained on the same 11 classes over the same frozen features differ
   by a few percent in loss. `exp(−2 · 0.03) ≈ 0.94` → α ≈ 0.47 essentially
   always. To halve α the neighbour's *total* loss would have to be 35%
   higher.

Empirical confirmation: the gossip matrix above has zero row variance.
Nothing was ever gated.

**And this was structurally unobservable.** `gossip_round()` returns the gate
record specifically so that "the gate is not silently passing everything at
full strength" is checkable (`gossip.py:98`) — and `train.py:_gossip()`
**discards the return value**. 406 gossip ops, zero gate values recorded:
`grep -ci "alpha\|gate" gossip_seed42_*.log` → 0. Layer F was dropped for the
same reason; this was the only remaining hook, and it is unwired.

### Contributing cause (smaller): momentum is wiped every 20 steps

`reset_momentum()` fires on every gossip op. With `MOMENTUM=0.9` the buffer
needs ~10 steps to reach steady state, so a gossip client runs at roughly 60%
of local_only's effective learning rate throughout. The rule itself is correct
(the buffer was measured at weights that no longer exist), but its interaction
with `GOSSIP_EVERY_N_STEPS=20` is a real handicap that `local_only` never pays.

### Minor: the gate batch is a fixed 32 images during task 0

`Client.gate_batch()` falls back to `current_names[:GATE_BATCH_SIZE]` while
the buffer is empty. `current_names` is unshuffled, so the gate sees the
*same 32 images* on every call for the whole of task 0.

---

## (b) Why forgetting/BWT stay at 1e-3 in every mode

### Primary cause: for 2 of 5 clients, "task 1" is 9–15 optimiser steps

Task 0 runs at `BATCH_SIZE=32` (buffer empty, so `use_replay` is false);
tasks 1–2 run at `cur_bs = 16` with replay filling the other half. So:

- overcast / night: `ceil(72/16) = 5` batches × 3 epochs = **15 steps**
- partly cloudy / night: `3 × 3` = **9 steps**

Verified exactly against the recorded `local_steps`:

| client | task 0 | task 1 | task 2 | sum | logged |
|---|---|---|---|---|---|
| clear | 1068 | 3924 | 348 | 5340 | 5340 ✓ |
| overcast | 645 | **15** | 201 | 861 | 861 ✓ |
| rainy | 219 | 375 | 60 | 654 | 654 ✓ |
| snowy | 249 | 381 | 78 | 708 | 708 ✓ |
| partly cloudy | 369 | **9** | 102 | 480 | 480 ✓ |

You cannot measure catastrophic forgetting of task 0 when the intervening
task is 9 gradient steps. Both of those clients report forgetting of exactly
`0.00000` / `0.00030`.

**Corollary — two of the reported numbers are noise.** overcast's task-1 mAP
of 0.2786 is the *highest* night score anywhere in the run, and it is computed
on an **18-image test cell**. partly cloudy's task-1 test cell is **10 images**.
`CLASS_FLOOR = 50` GT instances exists in `evaluate_detection.py` precisely to
catch this, but `core_present` only filters classes with `N == 0` — it does not
exclude below-floor classes from the macro. Those two mAPs feed straight into
BWT; overcast's task-1 term (+0.0104) is the single largest contribution in
sequential mode.

### Primary cause: the head has no task-specific function to overwrite

Only `roi_heads.box_predictor` is trainable — `cls_score` (1024→11) and
`bbox_pred` (1024→44), two linear layers over a frozen, shared backbone and
RPN. The three tasks are time-of-day splits over the **same 10 classes with
the same label semantics**. The mapping "ROI feature → class + box delta" is
the *same function* for daytime, night and dawn/dusk. Training on night moves
the head toward that shared optimum, not away from it. There is no
interference to create — hence nothing for replay to prevent, and nothing for
gossip's BWT to repair.

`sequential` settles this. It has **no replay buffer at all**, and it is where
forgetting would have to appear:

```
sequential acc_history, task 0 (daytime), evaluated after tasks 0/1/2
  clear          0.2567 → 0.2459 → 0.2546
  overcast       0.2507 → 0.2452 → 0.2514
  rainy          0.2359 → 0.2328 → 0.2389
  snowy          0.2369 → 0.2321 → 0.2410
  partly cloudy  0.2459 → 0.2406 → 0.2473
```

Four of five clients end *above* where they started, giving forgetting of
exactly 0.0. The setup does not produce forgetting, with or without replay.

And therefore: `sequential` BWT (+0.00251) ≈ `gossip` BWT (+0.00279).
Sequential has no gossip and no replay. The BWT number is measuring "a few
thousand more SGD steps on a shared head nudge everything up slightly," not
backward transfer.

### Underlying cause: the head never converges — mAP is pinned at ~0.24 regardless

From `joint_seed42_*.log` (all 17k images pooled, 5,283 steps), training loss
by 1000-step window:

| steps | mean | min | max | sd |
|---|---|---|---|---|
| 0–1000 | 1.2608 | 1.1235 | 1.4105 | 0.089 |
| 1000–2000 | 1.1687 | 1.0339 | 1.3208 | 0.075 |
| 2000–3000 | 1.1766 | 1.0913 | 1.3514 | 0.077 |
| 3000–4000 | 1.2541 | 1.0878 | 1.4413 | 0.091 |
| 4000–5300 | 1.2510 | 1.1249 | 1.4108 | 0.095 |

**The loss does not descend at all after ~step 100.** `LR_HEAD=0.01` with
`MOMENTUM=0.9`, `LR_WARMUP=None` and no schedule gives an effective LR of
~0.1 on a randomly-initialised head. torchvision's own reference Faster R-CNN
recipe is lr=0.02 at batch 16 *with a 1000-iteration linear warm-up and step
decay*; here there is neither.

The consequence shows up across the entire results table:

- 480 steps (partly cloudy, local_only) → 0.2269
- 5,340 steps (clear, local_only) → 0.2315
- 5,283 steps on 11× the pooled data (joint) → 0.2460

An 11× change in training data moves mAP by 0.02. The "ceiling" baseline sits
barely above a single well-trained client, and *below* local_only's clear on
its own daytime cell (0.2626).

A model sitting at a noise-dominated equilibrium has nothing to forget.
Forgetting requires convergence to something task-specific first.

**Secondary ceiling — the frozen RPN.** `mAR@100` is 0.30–0.33 for every
client in every mode while `mAP50` is 0.44–0.51. Recall caps at roughly 1/3,
consistent with the frozen COCO RPN not proposing BDD100K's small objects —
`model.py`'s docstring already flags this (55.5% of boxes under 32px). That is
a hard ceiling the head cannot cross however it is trained, and a second
reason every configuration converges to the same number.

---

## One correction to the handoff

> gossip's `own_vs_other_gap` (0.00063) is much smaller than local_only's
> (0.0045) … a plausible fingerprint of gossip actually homogenizing the
> clients' weights

Gossip *did* homogenize the clients — but `global_own_vs_other_gap` is not
what shows it, and the comparison above is a signed-mean cancellation
artifact. Per-client gaps:

| | clear | overcast | rainy | snowy | partly cloudy | signed mean | **mean abs** |
|---|---|---|---|---|---|---|---|
| gossip | −0.0265 | +0.0259 | −0.0189 | −0.0030 | +0.0257 | +0.00063 | **0.0200** |
| local_only | −0.0208 | +0.0273 | −0.0158 | +0.0031 | +0.0287 | +0.00450 | **0.0192** |

The mean *absolute* gap is indistinguishable between the two modes. The
individual gaps are ±2–3%, not ~0.

What the metric is actually measuring is **test-set difficulty**, not
specialization. Read the matrix column means: every model scores ~0.201 on
clear's test set and ~0.243 on overcast's, whichever model it is. No client's
model is best on its own data — the diagonal is not special. Under local_only,
`clear`'s model is the best model on all five test sets including the other
four clients'. `own_vs_other_gap` is therefore dominated by "is my own pooled
test set harder than average", which is a property of the data partition, not
of gossip.

The real evidence of homogenization is the zero row-variance in the gossip
matrix.

---

## What to change, in order of impact

1. **Stop mixing into a client whose peers have all frozen**, or decay λ by
   staleness. Minimum fix: have `sample_partner()` prefer unfinished clients
   and skip gossip entirely when none exist. 83% of the mechanism's operations
   are currently pure drag on the only client still learning.

2. **Fix the partition.** overcast/night = 72 images and partly cloudy/night
   = 44 are not tasks. Either merge weather groups, drop those cells and
   report 2 tasks for those clients, or cap `clear` so the round-robin isn't
   11:1. Separately, enforce the `CLASS_FLOOR = 50` that already exists — an
   18-image test cell should not produce a reportable mAP.

3. **Get the head to converge before drawing any CL conclusion.** Add warm-up
   plus step/cosine decay, or drop `LR_HEAD` to ~0.002–0.005. Until the joint
   loss actually descends, forgetting and BWT are unmeasurable in principle,
   and the gossip comparison is being run on models that all sit at the same
   noise floor.

4. **Wire up the gate record.** `_gossip()` discards `gossip_round()`'s return
   value; persist `alpha_a`/`alpha_b`/`loss_own_*`/`loss_other_*` per op. Also
   restrict `gate_loss()` to `loss_classifier + loss_box_reg` — the RPN is
   frozen, so including its terms only desensitizes the gate.

5. **Report `mean |own_vs_other_gap|`** alongside (or instead of) the signed
   mean in `evaluate_cross_client()`, and report the matrix column means so
   test-set difficulty is visibly separated from transfer.

---

## Addendum 1: RETRACTED — the RPN unfreeze section

**The premise was false and I should not have built on it.** Nobody unfroze
the RPN. `config.py:59` reads `FREEZE_RPN = True`; the 2026-09-13 mtimes on
`config.py`/`model.py` were `RPN_PRE_NMS_TOP_N_TEST` /
`RPN_POST_NMS_TOP_N_TEST` going 1000 → 2000 and being threaded into
`fasterrcnn_resnet50_fpn(...)` — an inference-time proposal budget, chosen as
the cheap probe of the `mAR@100 ≈ 0.30` ceiling before paying for an unfreeze.
Nothing became trainable.

Everything below in the original addendum — gate-loss composition under a
trainable RPN, gossip payload size, anchor retuning, the revised work order —
is void. In particular:

- **`gate_loss()` should still be restricted to `loss_classifier +
  loss_box_reg`.** My original recommendation 4 stands as written; the
  "correction" that voided it does not apply. While the RPN is frozen and
  identical across clients, `loss_objectness` and `loss_rpn_box_reg` are a
  shared constant that inflates the `excess` denominator and biases
  `gate_weight` toward 1.0.
- `_head_state()` selecting by `requires_grad` continues to mean head-only,
  and `gossip.py`'s docstring is accurate as written.

**The one finding that survives, reframed as latent rather than live:**
`LR_PRETRAINED = 0.0` (`config.py:78`) has exactly one grep hit — its own
definition. `build_optimizer()` (`model.py:73`) builds a single param group
over `trainable_parameters(model)` at `lr=LR_HEAD`. Nothing is broken today,
because nothing below `roi_heads` is trainable. But the comment on that line
advertises *"kept so unfreezing is one edit"*, and that is wrong: flipping
`FREEZE_RPN` alone would hand COCO-pretrained RPN conv weights the same
unwarmed `lr=0.01` as the randomly-initialised head. Worth fixing the comment
now and the param groups whenever the flag is actually flipped.

---

## Addendum 2: the first gate measurements

### Partial retraction of "the gate is a no-op"

`alpha_a` spanning 0.013–0.43 is direct evidence the gate discriminates, and
"no-op" was too strong. Retracted as stated.

But the two claims are measuring different phases and both can hold. My claim
was about the **steady state**, and my evidence was the zero row-variance in
the *final* 5×5 matrix — an end-of-run observation. These four exchanges are
from the **first slice** of the run. Below is why that distinction is the
whole story, and what number settles it.

### The four exchanges are explained by round-robin position, not by the proposed mechanism

`run()` iterates `clients.values()` in `CLIENTS` order: clear(0), overcast(1),
rainy(2), snowy(3), partly cloudy(4). Each client trains 20 steps and then
gossips, so within slice 1 a client at position *k* has trained iff *k* ≤ the
initiator's index. Lining that up against the log:

| # | initiator (idx) | partner (idx) | partner trained yet? | alpha_a |
|---|---|---|---|---|
| 1 | clear (0) | snowy (3) | **no — random init** | 0.0135 |
| 2 | overcast (1) | rainy (2) | **no — random init** | 0.0428 |
| 3 | rainy (2) | partly cloudy (4) | **no — random init** | 0.0133 |
| 4 | snowy (3) | clear (0) | **yes — 20 steps** | **0.4319** |

In exchanges 1–3 the partner's head is still at `FastRCNNPredictor`'s random
initialisation. In exchange 4 — the only one whose partner has trained —
`alpha_a` jumps by a factor of 32, to within 14% of lambda.

So `alpha_a`'s range is not the gate resolving quality differences among
comparable peers. It is the gate detecting **trained vs. randomly-initialised**,
which is the largest quality gap that will ever exist in the run, at the one
moment it exists. That is consistent with saturation later, not evidence
against it.

The handoff's proposed mechanism for `alpha_b` — *"the partner has not trained
recently and has been mixed since it last did, so its fit to its own data is
degraded"* — cannot be what produced these four points: in exchanges 1–3 the
partner had been mixed **zero** times and trained **zero** steps. The
mechanism is plausible for the steady state; it is just not what this evidence
shows.

### A simpler and more general cause for `alpha_b = lam`: the gate is clamped from above

```python
excess = max(0.0, (loss_neighbour - loss_own) / loss_own)
return math.exp(-sensitivity * excess)
```

`gate_weight` returns **exactly 1.0 whenever the neighbour's loss is at or
below mine, by any margin at all.** It can only ever penalise; it can never
reward. A neighbour that is dramatically better and one that is better by
1e-6 receive identical alphas.

Combined with `lam = 0.5`, "at least as good as me on this batch" means half
my weights are replaced. And `gate_loss()` runs in `train()` mode, where the
RPN's `fg_bg_sampler` and `roi_heads.select_training_samples` randomly
subsample anchors and proposals — so the two calls being compared draw
*different* random samples. Between two statistically indistinguishable
models, which one "wins" is then a coin flip, and the clamp converts every
win into exactly `lam`.

**This yields a clean discriminating test, already in the data being
collected:**

- If `alpha_b`'s `frac_at_lam` settles near **0.5** once all five clients have
  trained, with the remainder clustered just below lam → it is the clamp plus
  sampling noise, and there is no initiator/partner asymmetry to explain.
- If it stays near **1.0** across ~150 exchanges → the handoff's asymmetry is
  real (p ≈ 2⁻¹⁵⁰ under the clamp hypothesis). Decisive either way.

### The proposed mechanism points the wrong way for finding (a)

The handoff argues that because the scheduler rotates the initiator role,
"every client alternates between protecting itself and being half-replaced,
which would plausibly hold the whole system below `local_only`." Check the
arithmetic:

```
a' = (1 - α_a)·a + α_a·b
b' = (1 - α_b)·b + α_b·a
a' + b' = a·(1 - α_a + α_b) + b·(1 + α_a - α_b)
```

Symmetric mixing (α_a = α_b) preserves the population mean exactly — it
collapses variance and nothing else. With the observed α_a = 0.0135,
α_b = 0.5, the sum becomes `1.4865·a + 0.5135·b`: weight shifts **toward the
initiator**, which is the better-fit model. The asymmetry is mean-*increasing*
when the gate is working, because the gate's whole job is to make the better
model dominate the exchange.

So phase-1 role alternation should push the consensus **up**, not below
`local_only`. It cannot be the cause of finding (a). Something has to break
the symmetry in the other direction.

### Reconciliation with §11: two phases, and only one of them is load-bearing

The two findings are complementary, but they operate in different phases of
the run and carry very unequal weight.

**Phase 1 — all clients active** (seed 42: slices 1 to ~44, 45 of clear's 268
exchanges). Both sides gate and both sides mix. The initiator/partner
asymmetry operates here, and per the arithmetic above it is benign or mildly
beneficial. Its real effect is speed: the population reaches consensus fast.
This is the regime the four seed-7 points sample.

**Phase 2 — only `clear` still active** (slices ~45 to 269, **223 of clear's
268 exchanges, 83%**). Every peer has finished all three tasks. Structurally:

- `clear` is the initiator in **100%** of exchanges — it never occupies the
  partner role again, so the `alpha_b` side is not merely unobserved, it is
  **switched off**: `if not client_b["finished"]` is False, alpha_b stays at
  its initialised 0.0, and no mix into the partner occurs.
- The four peers only ever give. In a pairwise-averaging network, nodes that
  give without taking are **absorbing boundary conditions** — they are fixed
  points, and the mean-preservation property above no longer holds at all.
  The consensus is dragged toward the frozen values in proportion to how often
  they are sampled, which here is every single exchange.

**And the gate cannot stop it — for a reason the handoff's mechanism predicts
backwards.** By that mechanism `clear`, as the perpetual freshly-trained
initiator, should have a small `alpha_a` and be well protected. It was not
protected: it fell from 0.2253 (local_only) to 0.2009 (gossip) on its own test
set.

The reason is that **`gate_weight` measures quality, not staleness or
informativeness.** The frozen peers are not bad models — they completed their
entire curriculum. From the local_only matrix, overcast's finished model
scores 0.2055 on clear's pooled test set against clear's own 0.2253: a gap of
~0.02 mAP, a few percent in loss terms, giving `excess ≈ 0.03` and
`gate_weight ≈ exp(-0.06) ≈ 0.94` → **α_a ≈ 0.47**. A stale peer passes at
near-full strength *precisely because it is a decent model*. Nothing in
`gate_weight(loss_own, loss_neighbour)` can see that this neighbour stopped
learning 4,000 steps ago and has nothing left to contribute.

So the unified statement is:

> The gate discriminates well exactly when the quality gap is large, and
> saturates exactly when it is small. Finding (a) lives entirely in the
> regime where the gap is small and the peer is frozen — where the gate is
> blind by construction.

### What the completed seed-7 run should show

In order of diagnostic value:

1. **`alpha_a` during phase 2** (the single most informative number in the
   log). My anchor mechanism predicts it sits at **0.45–0.50** once peers are
   finished — near-full mixing into stale weights. If it instead stays small,
   the gate *is* protecting `clear` and §11 is wrong.
2. **`alpha_a`'s trajectory** across phase 1: should climb from ~0.01 toward
   lam as clients stop being randomly initialised. Exchange 4's 0.4319 is
   already the leading edge.
3. **`frac_at_lam` on the b-side alone**, restricted to phase 1: ~0.5 → clamp;
   ~1.0 → genuine asymmetry.

### One defect in the new gate summary, worth fixing before the run lands

`_gate_summary()` (`train.py:361`) pools both sides into one list:

```python
alphas = [r[k] for r in gate_log for k in ("alpha_a", "alpha_b")
          if r.get(k, 0.0) > 0.0]
```

The `> 0.0` filter correctly drops the structural zeros from finished
partners — good, that would otherwise have been ~223 fake "the gate rejected
everything" entries skewing the mean. But pooling `alpha_a` with `alpha_b`
merges two distributions that this very measurement just showed are wildly
different (0.013–0.43 vs. a spike at exactly 0.5). A single `alpha_mean` /
`alpha_std` / `frac_at_lam` over the pool is not interpretable.

Worse, the pool is dominated by phase 2, which is 83% of exchanges and is
**entirely a-side** — so the headline `frac_at_lam` will in practice be
"clear's gate against frozen peers" while being labelled a global statistic.

Two small changes make the log answer all three questions above:

- report `alpha_a` and `alpha_b` summaries **separately**;
- record `b_finished` (and the initiator name) per exchange, so phase 1 and
  phase 2 can be split apart afterwards.

