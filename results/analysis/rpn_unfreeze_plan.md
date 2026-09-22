# Plan: halve the anchors, unfreeze the RPN

## Why

`local_only` sits at 0.2398 mAP and `joint` — all data pooled, no CL, no
federation — only reaches 0.2568. Every mode is within 0.017 of that ceiling,
so the limit is the detector, not the algorithm.

The detector is recall-limited, and the recall loss is entirely at small
sizes:

```
macro mAP     0.2456      <- AP cannot exceed recall
macro mAR@100 0.3481
macro mAR_S   0.2403
macro mAR_L   0.5065
```

Per-class recall tracks the small-object fraction almost exactly:

| class | % small | AR@100 | AP |
|---|---|---|---|
| car | 46% | 0.495 | 0.408 |
| person | 44% | 0.390 | 0.287 |
| traffic light | 89% | 0.279 | 0.167 |
| traffic sign | 74% | 0.229 | 0.120 |

The cause is a scale mismatch. Over 1,027,770 training boxes:

```
p10 12.6   p25 17.4   p50 28.3   p75 54.3   p90 112.1   p99 336.9
```

The **median box (28.3px) is smaller than the smallest anchor (32px)**.
55.6% of boxes fall below the smallest anchor; 0.09% exceed the largest.
The evaluation's "small" band is area < 1024px² = 32×32 — precisely the
smallest anchor. Every object counted as small was never proposable.

The resolution probe confirms the direction and rules out the cheap fix
(`results/analysis/res_probe.json`): upscaling 1.5x lifts `mAR_S` by +0.030
but drops `mAR_M`/`mAR_L`, netting only +0.011 mAP, and 2x is net negative.
Resolution moves every object; anchors move only the floor.

## The change

**Anchors: halve the defaults.**

```
(32,), (64,), (128,), (256,), (512,)   ->   (16,), (32,), (64,), (128,), (256,)
```

Justified as "driving scenes contain smaller objects than COCO", with no
fitting to our box statistics. It happens to align with the measured
distribution, but nothing here is tuned to it, and nothing is derived from
the test split.

Keep one size per level and three aspect ratios. `num_anchors` per location
stays 3, so the RPN head's conv shapes are unchanged and the pretrained
weights still load.

**But the weights become wrong.** They were trained to score objectness and
regress offsets against 32px anchors. Given 16px anchors the priors are
mismatched, so the RPN has to be retrained. That is the actual work.

### 1. `model.py` — anchors

Pass an explicit `AnchorGenerator` (or `rpn_anchor_generator=`) to
`fasterrcnn_resnet50_fpn` with the halved sizes and the existing
`(0.5, 1.0, 2.0)` ratios.

### 2. `config.py` — unfreeze

`FREEZE_RPN = False`. Keep `FREEZE_BACKBONE = True` — one variable at a time.

### 3. `model.py` — optimizer (the part that is easy to miss)

`build_optimizer` currently makes a **single** param group at
`LR_HEAD = 0.01`:

```python
return torch.optim.SGD(trainable_parameters(model), lr=lr, ...)
```

`LR_PRETRAINED` exists in `config.py` but is never imported — dead code. As
it stands, unfreezing hands pretrained RPN convs the same unwarmed 0.01 as a
randomly initialised head, which will destroy them in the first few steps.

Needs two param groups:

| group | params | lr |
|---|---|---|
| pretrained | `model.rpn.*` | `LR_PRETRAINED`, ~1e-3 |
| new | `model.roi_heads.*` | `LR_HEAD` = 0.01 |

`reset_momentum` iterates `param_groups`, so it keeps working unchanged.

**Warmup.** `LR_WARMUP = None` today, per `design_choices.pdf`. Mixing a
pretrained group with a random one across a 10x lr gap usually needs a few
hundred warmup steps. Adding it is a deliberate deviation from the spec and
should be recorded as one, not slipped in.

### 4. `gossip.py` — what travels, and what the gate measures

Two consequences that are easy to miss because they are not in `model.py`.

**Payload.** `_head_state` sends every `requires_grad` parameter. Unfreeze
the RPN and its ~594k weights start travelling: 13.95M -> ~14.55M per
exchange, about +4%. More importantly, clients begin averaging each other's
RPNs, which is a design decision, not a side effect. Recommend gossiping
everything trainable — "what you train, you share" is the consistent rule —
and renaming `_head_state`, whose name and docstring both say "head".

**Gate signal.** `GATE_LOSS_KEYS` is currently `("loss_classifier",
"loss_box_reg")`. That was correct only because the RPN was frozen and
identical across clients, making `loss_objectness` and `loss_rpn_box_reg` a
shared constant. Unfrozen, they carry signal again and belong back in the
sum. The comment in `gossip.py` already flags this as valid only while
`FREEZE_RPN` is true.

### 5. `smoke_test.py`

The check named "backbone/FPN/RPN frozen, head trainable" asserts
`trainable < 50% of total`. At 14.55M/41.3M = 35% it still passes, so it will
go green while testing the wrong thing. Rename and re-scope it.

## Order of work

Each step gates the next. Do not run gossip until the detector is settled.

1. Anchors + unfreeze + optimizer groups + smoke test. **Gate:** smoke test
   green, loss finite, trainable count ~14.55M.
2. **`joint` only**, seed 42. ~3.5h. **Gate:** does `mAR_S` actually move?
   This is the whole hypothesis. If `mAR_S` does not improve, stop — the
   anchors were not the problem and nothing below is worth the hours.
3. `local_only`, seed 42. Re-establishes the non-gossip baseline on the new
   detector.
4. Only then `gossip` + `sequential`.

Tag everything distinctly (e.g. `a16rpn`) so the `gp3fo2` results stay intact.

## Risks

- **The pretrained RPN may not recover.** 3 epochs per task is short to
  re-adapt a component that was trained for many epochs at different anchor
  scales. If `joint` gets *worse*, that is the likely cause — try a lower
  pretrained lr or longer warmup before abandoning the idea.
- **Every previous number becomes incomparable.** All four modes need
  rerunning; `gp3fo2` cannot be mixed with post-change results.
- **The gossip experiment now sits on a moving detector.** Settle steps 1-3
  before touching gossip, or two variables move at once.
- **Memory.** Backprop through the RPN adds activation memory. The workload
  already needs ~13GB of a 16GB P100 and OOMs if anything else is on the
  card. The preflight guard in `run_baseline.sh` catches a dirty GPU, but if
  the RPN gradients push us over on a *clean* card, batch size has to drop —
  which changes the experiment and forces the baselines to be rerun at the
  new batch size too.

## Cost

~9h wall clock for all four modes if they run in parallel on four clean P100
nodes, plus ~3.5h for the step-2 `joint` gate first. Step 2 is the cheap
decision point: it answers whether the anchor hypothesis is right for 3.5h
rather than 25.

## Still open, unrelated to this

The gossip gate remains inert — `weight_mean` 0.9825, `excess_median` 0.0058,
zero rounds skipped. The cells are near-identical (total variation ~0.05 on
class mix), so there may be nothing for a gate to discriminate. Worth
settling separately from the detector work.
