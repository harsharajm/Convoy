# Diagnosis request: seed-42 baseline metrics look off

## Project context

Decentralised gossip-based federated continual-learning object detector on
BDD100K. Faster R-CNN (COCO-pretrained), ResNet+FPN+RPN frozen, only
`roi_heads.box_predictor` (an 11-class head, 10 BDD100K classes +
background) is trainable — see `model.py`. Clients = 5 weather conditions
(`clear`, `overcast`, `rainy`, `snowy`, `partly cloudy`), tasks = 3
time-of-day splits trained sequentially per client (`daytime`, `night`,
`dawn/dusk`). No server, no FedAvg — clients gossip pairwise, gated by how
well a neighbour's weights score on the client's own replay memory
(`gossip.py`). Four run modes exist (`config.py` MODES): `gossip` (the
actual mechanism under test), `local_only` (replay, no gossip — isolated
baseline), `sequential` (no replay, no gossip), `joint` (all data pooled,
no continual learning — the ceiling baseline).

Hyperparameters for this run: `GOSSIP_LAMBDA=0.5`, `GATE_SENSITIVITY=2.0`,
`GOSSIP_EVERY_N_STEPS=20`, `GATE_BATCH_SIZE=32`, `EPOCHS_PER_TASK=3`,
`LR_HEAD=0.01`, `BUFFER_CAPACITY=2000`, `REPLAY_RATIO=1.0`. All from
`config.py`.

Results below are from `results/runs/{mode}_seed42.json`, produced by
`train.py`, metrics computed by `evaluate_detection.py`
(`DetectionEvaluator.summarize()` — COCO-style mAP@[.5:.95], mAP50, mAR@100,
plus `forgetting`/`bwt` from `compute_forgetting`/`compute_bwt` over each
client's per-task accuracy history).

## The numbers

| Mode | mean mAP@[.5:.95] (system) | mean forgetting | mean BWT | global cross-client mAP | own-vs-other gap |
|---|---|---|---|---|---|
| **gossip** | 0.2244 | 0.00061 | 0.00279 | 0.2229 | 0.00063 |
| **local_only** | 0.2311 | 0.00080 | 0.00215 | 0.2275 | 0.00450 |
| **sequential** | 0.2306 | 0.00064 | 0.00251 | — (schema has no cross_client for this mode) | — |
| **joint** (ceiling, different schema — `per_client_pooled_mAP`/`mean_mAP`) | 0.2460 | n/a | n/a | n/a | n/a |

Per-client final mAP, all four modes, ranges roughly 0.21–0.26 across
`clear`/`overcast`/`rainy`/`snowy`/`partly cloudy` — no client is a wild
outlier in any mode.

## What looks wrong

**1. `gossip` scores *lower* than `local_only`, not higher.** The entire
point of the gossip mechanism is that clients sharing gated updates should
do at least as well as clients training in total isolation — gossip is
mean 0.2244 vs local_only's 0.2311, about 2.9% relatively worse. Gossip is
even slightly behind `sequential` (0.2306), which has no replay buffer at
all. If gossip is not beating a baseline with *zero* inter-client
communication, either the gate is admitting harmful neighbour updates, the
mixing weight (`GOSSIP_LAMBDA=0.5`) is too aggressive relative to how
distinct each client's task distribution is, or there's a bug in when
momentum gets reset (`reset_momentum` in `model.py`, called from
`gossip.py` `gossip_round` — see the note in `gossip.py`'s docstring about
"the rule settled in PIPELINE_CONFLICTS.md §4").

One mitigating data point: gossip's `own_vs_other_gap` (0.00063) is much
smaller than local_only's (0.0045) — i.e. gossip clients perform almost
identically on their own data vs. other clients' data, while local_only
clients have a real gap (unsurprising, they never see other
distributions). That's a plausible fingerprint of gossip actually
homogenizing the clients' weights. But it looks like it's homogenizing
everyone down toward a slightly worse average, rather than lifting
everyone toward the better-performing clients' level. Worth checking
whether that's the gate's fault (see `gate_weight()` in `gossip.py`) or
just what averaging does to already-similar clients.

**2. Forgetting and BWT are both tiny — order 1e-3 — in every mode,
gossip included.** Mean forgetting is 0.0006–0.0008, mean BWT is
0.002–0.003, and the per-client min/max ranges in `system.forgetting`
and `system.bwt` are similarly small (e.g. gossip's forgetting ranges
0.0 to 0.002). That's a suspiciously flat signal for a continual-learning
setup that's supposed to have measurable forgetting *and* measurable
gossip-driven backward transfer. Possible explanations worth checking:
only 3 epochs/task (`EPOCHS_PER_TASK=3`) may be too few for the head to
move enough to forget anything; the replay buffer (`REPLAY_RATIO=1.0`,
1:1 current-task:replay) may be strong enough to nearly eliminate
forgetting by construction, which would make it hard to see gossip's BWT
effect on top of it; or there's an issue in how `acc_history` gets
populated (`Client.finish_task()` in `train.py`) that's making the
before/after task accuracy nearly identical regardless of what actually
happened during training.

## Ruled out (not actually a bug)

`joint_seed42.json` has an empty `system`/`cross_client` at first glance —
this looked like missing data but isn't: `joint` mode runs a structurally
different code path (`run_joint()` in `train.py`, pools all data, no task
boundaries, so no forgetting/BWT/cross-client concept applies at all). Its
real schema is `per_client_pooled_mAP` + `mean_mAP` (0.2460), which is
in the same ballpark as the other three modes and plausibly the ceiling,
as expected. Not a lead.

## ADDENDUM (2026-09-13): correction to the premise, and first gate measurements

### The RPN was never unfrozen — correct a claim in circulation

A reviewing session was told "the code was since changed, in the new code we
unfreeze the RPN, those results are from when RPN was frozen." **That is
false and no part of it should be carried forward.** `config.py:59` reads
`FREEZE_RPN = True`, `freeze_pretrained()` is called unconditionally from
`build_model()`, and there is no env var or CLI flag overriding it.

What *did* change on 2026-09-13 (and is why `config.py`/`model.py` have new
mtimes) is the RPN **proposal budget**, not its trainability:
`RPN_PRE_NMS_TOP_N_TEST` and `RPN_POST_NMS_TOP_N_TEST` were raised 1000 →
2000 and threaded into `fasterrcnn_resnet50_fpn(...)`. That is a pure
inference-time knob — no weight changes state, nothing new becomes
trainable. It was chosen deliberately as the cheap probe of whether the
`mAR@100 ≈ 0.30` recall ceiling comes from proposal starvation, *before*
spending anything on unfreezing. Any analysis premised on a trainable RPN
(gate-loss composition, gossip payload size, anchor retuning) is answering
a question nobody asked yet.

### One genuinely valuable finding from that review, which does hold

`LR_PRETRAINED = 0.0` is dead code — `grep -rn LR_PRETRAINED --include=*.py`
returns exactly one hit, its own definition at `config.py:78`. It is never
imported or read. `build_optimizer()` (`model.py`) builds a **single** param
group over `trainable_parameters(model)` at `lr=LR_HEAD`. So whenever
someone does flip `FREEZE_RPN`, COCO-pretrained RPN conv weights would
silently receive the same unwarmed `lr=0.01` as the randomly-initialised
head, which would wreck them rather than fine-tune them. Unfreezing is a
two-part change (flag + a second param group), not one. Worth fixing before
it is ever flipped.

### The gate was unobservable, and now isn't

`gossip_round()` returns a record of the gate losses and the alphas actually
applied. Its docstring states this exists specifically so one can check "the
gate is not silently passing everything at full strength, which is otherwise
unobservable now that Layer F is dropped." **`_gossip()` in `train.py` was
discarding that return value entirely.** So every result in the table above
was produced by a run that had no way to report what its own central
mechanism did.

Fixed: `_gossip()` now returns the record, the call site logs
`alpha_a`/`alpha_b` per exchange, and `_collect()` writes both a `gate_log`
and a `gate` summary (`alpha_mean/std/min/max`, `frac_at_lam`) into the
saved JSON. `frac_at_lam ≈ 1.0` with `alpha_std ≈ 0` is the fingerprint of a
decorative gate.

### First measurements — a systematic initiator/partner asymmetry

From the first four exchanges of the re-launched `gossip` seed-7 run
(`GOSSIP_LAMBDA = 0.5`, so `alpha = 0.5` means the gate passed at **full**
strength):

```
[clear]         <-> [snowy]          alpha_a=0.0135  alpha_b=0.5000
[overcast]      <-> [rainy]          alpha_a=0.0428  alpha_b=0.5000
[rainy]         <-> [partly cloudy]  alpha_a=0.0133  alpha_b=0.5000
[snowy]         <-> [clear]          alpha_a=0.4319  alpha_b=0.5000
```

**This four-sample reading was wrong and is retracted — see the completed-run
numbers below.** It suggested the gate discriminated on the initiator's side;
over the full run it does not, in any meaningful sense.

### CONFIRMED over the completed seed-7 run: the gate is effectively a no-op

406 exchanges, 565 mixes:

```
alpha_mean 0.4882   (lambda = 0.5, so 97.6% of full strength)
alpha_std  0.0359
alpha_min  0.0133   alpha_max 0.5
frac_at_lam 0.331
```

`frac_at_lam` is only 0.33, so the gate is not *literally* constant — but
`alpha_mean = 0.488` against a ceiling of 0.5, with `std = 0.036`, means it
passes essentially everything at near-full strength. The thin tail down to
0.013 is rare enough not to matter. Functionally this run is plain pairwise
averaging. The earlier four-sample snapshot happened to catch the tail.

**The cause is measured, not guessed.** From the same `gate_log`:

```
loss_own   mean 1.3397  std 0.2277
loss_other mean 1.3481  std 0.2246
mean |other - own| = 0.0083  ->  0.62% of loss_own

excess: mean 0.0181  median 0.0061  p90 0.0242
frac excess == 0 (neighbour scores >= me): 0.331
```

A neighbour's loss differs from your own by **0.62% on average**. With
`GATE_SENSITIVITY = 2.0`, `gate_weight = exp(-2 x 0.0061) = 0.988` at the
median — indistinguishable from 1.0. The gate cannot separate clients
because the quantity it measures barely varies.

Why it barely varies: `gate_loss()` sums the **entire** Faster R-CNN loss
dict, including `loss_objectness` and `loss_rpn_box_reg`. Those come from the
frozen, client-identical backbone+RPN, so on a given batch they are the *same
constant* for both models being compared. They inflate the denominator of
`excess = (loss_neighbour - loss_own) / loss_own` while contributing zero
signal. The real signal — the trainable head's `loss_classifier` +
`loss_box_reg` — is diluted into a sub-1% difference.

This was listed as a "second contributing factor" above. It is the primary
one. Two changes are needed together, since either alone is insufficient:

1. Gate on `loss_classifier + loss_box_reg` only, dropping the two frozen-RPN
   terms. (Correct **only while the RPN stays frozen** — if it ever becomes
   trainable those terms carry real signal again.)
2. Raise `GATE_SENSITIVITY` by roughly one to two orders of magnitude, or
   replace the absolute-excess formulation with one scaled to the observed
   spread. Even after (1), excess stays in the low single-digit percents, and
   `exp(-2 * 0.01)` is still ~0.98. Sensitivity 2.0 was calibrated for an
   excess range that this setup never produces.

Candidate mechanism, to be confirmed against the full run's `gate_log`:
the initiator has just completed 20 training steps on its own data
immediately before gossiping, so its `loss_own` is freshly minimised and any
neighbour looks bad against it → small `alpha_a`, i.e. **the initiator
protects itself**. The partner has not trained recently *and* has been
mixed since it last did, so its fit to its own data is degraded → a
freshly-trained peer scores at least as well on the partner's own memory →
`excess = 0` → `gate_weight = 1` → **the partner is overwritten at full
strength**. Because the scheduler round-robins the initiator role, every
client alternates between protecting itself and being half-replaced, which
would plausibly hold the whole system below `local_only` — exactly
finding (a).

A second contributing factor worth checking: `gate_loss()` sums the *entire*
Faster R-CNN loss dict, which includes `loss_objectness` and
`loss_rpn_box_reg`. With the RPN frozen and identical across all clients,
those two terms are a **shared constant** on any given batch. They inflate
both `loss_own` and `loss_neighbour`, shrinking the normalised
`excess = (loss_neighbour - loss_own) / loss_own` and biasing `gate_weight`
toward 1.0. That would push alphas toward lambda exactly as observed on the
partner side. Restricting the gate to the classification/box-regression
terms of the trainable head is a candidate fix **while the RPN stays
frozen** (it would stop being correct if the RPN ever became trainable).

Both of these are hypotheses fitted to four data points. The completed run
writes `gate.frac_at_lam` over all ~150 exchanges, which settles it.

## What to actually figure out

Given the numbers above and the mechanism described in `gossip.py` /
`config.py`, what's the most likely reason (a) `gossip` underperforms
`local_only` instead of beating it, and (b) forgetting/BWT stay in the
1e-3 range across every mode including the ones that should show more of
it? Relevant files to pull: `gossip.py` (gate + mixing), `model.py`
(`reset_momentum`, frozen layers), `train.py` (`Client.finish_task`,
`gossip_round` call site, `GOSSIP_EVERY_N_STEPS` pacing), `config.py`
(all hyperparameters), `evaluate_detection.py`
(`compute_forgetting`/`compute_bwt`/`DetectionEvaluator.summarize`).
