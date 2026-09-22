# gp3fo2 — seed 42, all four modes

Config: pull-only gossip, fanout 2, participation 3, lambda 0.5, gate
sensitivity 2.0, gate on head losses only, dose gate active, frozen-slice
snapshots, finished clients excluded as neighbours. RPN proposal budget 2000.

All four jobs COMPLETED, exit 0: gossip 8h22, local_only 8h10,
sequential 4h17, joint 3h44.

## Headline

| mode | gp3fo2 | old seed 42 | delta |
|---|---|---|---|
| gossip | 0.2398 | 0.2244 | +0.0154 |
| local_only | 0.2398 | 0.2311 | +0.0087 |
| sequential | 0.2394 | 0.2306 | +0.0088 |
| joint (ceiling) | 0.2568 | 0.2460 | +0.0108 |

**gossip no longer loses to local_only.** It was −0.0067; it is now −0.0000.

But read the decomposition before crediting the gossip work. `sequential`
and `joint` contain no gossip at all and still gained ~+0.009. That is the
RPN proposal budget (1000 → 2000), which the old seed-42 runs did not have.
So of gossip's +0.0154, roughly +0.009 is the budget change that every mode
got, and roughly +0.006 is gossip-specific.

## Per-client: gossip vs local_only

| client | images | gossip | local_only | delta | was |
|---|---|---|---|---|---|
| clear | 34,151 | 0.2413 | 0.2417 | −0.0004 | **−0.0216** |
| overcast | 8,008 | 0.2655 | 0.2635 | +0.0020 | −0.0031 |
| snowy | 5,054 | 0.2333 | 0.2354 | −0.0021 | −0.0048 |
| rainy | 4,638 | 0.2215 | 0.2227 | −0.0012 | −0.0037 |
| partly cloudy | 4,496 | 0.2374 | 0.2357 | +0.0017 | −0.0003 |

`clear` was carrying ~60% of gossip's whole deficit. That damage is gone.

## Why it is gone — and it is not the gate

The gate is still essentially inert:

| metric | gp3fo2 (seed 42) | previous (seed 7) |
|---|---|---|
| `weight_mean` | 0.9825 | — |
| `alpha_mean` (λ = 0.5) | 0.4916 | 0.4882 |
| `frac_at_lam` | 0.2137 | 0.3310 |
| `norm_spread_mean` | 0.0079 | n/a (fanout 1) |
| `excess_median` | 0.0058 | 0.0061 |
| `excess_p90` | 0.0224 | 0.0242 |
| `n_skipped` | 0 | n/a |

Removing the frozen-RPN terms from `gate_loss` was supposed to amplify the
signal. It did not: `excess_median` moved 0.0061 → 0.0058. Neighbours still
score within ~0.6% of the client's own loss, so the gate passes everyone at
98% strength, the dose gate only trims 1.7% off lambda, and `norm_spread_mean`
of 0.008 means the two neighbours in a round are indistinguishable. No round
was ever skipped.

So the gate did not protect `clear`. Something else did.

## The actual cause — and it is a problem

```
local_steps: clear 5340, overcast 861, snowy 708, rainy 654, partly cloudy 480
gossip_ops : clear   31, overcast  27, snowy  22, rainy  24, partly cloudy  13
slices: 269      total gossip exchanges: 117      "no live neighbours": 231
```

Task-2 completion times: partly cloudy 02:25, rainy 03:00, snowy 03:11,
overcast 03:26, **clear 06:49**.

`clear` has 7.6x the images of the smallest client and trained 11x the steps.
Once the last of its neighbours (overcast) finished at 03:26, `clear` had no
eligible neighbour left — finished clients are excluded as sources — and
simply trained alone for the remaining 3h23.

**`clear` had neighbours for only ~861 of its 5340 steps. It did 84% of its
training with zero gossip.** 231 of 269 slices logged "no live neighbours
drawn". Total exchanges across the whole run: 117, against 565 mixes in the
previous run.

That is why `clear` matches `local_only`: for most of its training it *was*
`local_only`.

This is a direct consequence of excluding finished clients as neighbours —
the TA-faithful choice made this session, overriding `design_choices.pdf`
2.2.1, which kept them eligible as sources. In the reference's setting the
clients are balanced and all finish together, so the rule costs nothing.
Here, with a 7.6x cell-size imbalance, it silently switches gossip off for
the dominant client.

**The headline result is therefore substantially an artifact of that rule,
not evidence that the gate or the dose fix works.**

## Small objects and per-class

Averaged over clients, final evaluation. Small = area < 1024 px².

| class | N | AP | AP50 | AP_S | AR_S | N_S | AR@100 |
|---|---|---|---|---|---|---|---|
| car | 11769 | 0.4086 | 0.7247 | 0.1729 | 0.3008 | 5444 | 0.4966 |
| traffic sign | 3679 | 0.1189 | 0.2443 | 0.0816 | 0.1741 | 2730 | 0.2314 |
| traffic light | 2920 | 0.1680 | 0.4637 | 0.1395 | 0.2548 | 2602 | 0.2799 |
| person | 1119 | 0.2925 | 0.5845 | 0.1238 | 0.2340 | 497 | 0.3966 |
| truck | 446 | 0.2622 | 0.4295 | 0.0813 | 0.1857 | 69 | 0.4814 |
| bus | 205 | 0.3028 | 0.4542 | 0.0152 | 0.0771 | 29 | 0.4638 |
| bike | 81 | 0.1658 | 0.3573 | 0.0041 | 0.0242 | 28 | 0.2602 |
| rider | 60 | 0.0670 | 0.1531 | 0.0190 | 0.0500 | 22 | 0.1387 |
| motor | 43 | 0.1458 | 0.2518 | 0.2235 | 0.2875 | 9 | 0.2425 |
| train | 3 | 0.0000 | 0.0000 | nan | nan | 0 | 0.0000 |

Below `CLASS_FLOOR` (50 instances): `motor`, `train`. Their AP_S/AR_S are not
reportable — `motor`'s 0.2235 rests on 9 boxes.

**`traffic sign` is the worst-performing well-populated class**: AP 0.1189
against `car`'s 0.4086. 74% of its instances (2730/3679) are small, and its
AR_S is 0.1741 — the model finds under a fifth of them.

**Small objects are a recall problem, not a classification problem.** For
every well-populated class, AR_S sits far below AR@100: car 0.30 vs 0.50,
traffic sign 0.17 vs 0.23, traffic light 0.25 vs 0.28, person 0.23 vs 0.40.
The detections are not being produced at all.

### Does this mean the RPN needs training?

It means the proposal stage is the suspect, but **training the RPN's weights
is unlikely to be the fix.** The anchors are COCO's, smallest 32px, and 55.5%
of BDD100K boxes are under 32px — across every weather cell (0.53–0.57). For
half the data there is no anchor that fits, and training weights does not
create anchors that do not exist. The lever is anchor *sizes*, which is an
architecture change that invalidates the pretrained RPN weights.

Note also that raising the proposal budget 1000 → 2000 bought ~+0.009 across
every mode, so proposal supply was genuinely part of the constraint.

## Context that has not changed

- `joint` = 0.2568 is still the ceiling, and every other mode sits within
  0.017 of it. The algorithm cannot reach past it.
- `cross_client` own-vs-other gap is 0.0026. Still essentially no client
  specialisation, consistent with the cell-heterogeneity measurement
  (total variation ~0.05 on class mix, ~0.02 on box size).
- gossip now equals local_only rather than losing to it. It still adds
  nothing, which is what near-identical clients predict.

## What to do next

1. **Re-decide the finished-neighbour rule.** This is the big one. Either
   restore `design_choices.pdf` 2.2.1 (finished clients stay eligible as
   sources) or equalise steps per client. As it stands the dominant client
   barely gossips, and no gossip result here is interpretable.
2. **The gate still does not discriminate.** `excess_median` 0.0058 after
   the loss-term fix. `GATE_SENSITIVITY = 2.0` gives `exp(-2 × 0.006) ≈ 0.99`.
   If the gate is meant to do anything, sensitivity has to rise by one to two
   orders of magnitude — or the gate signal has to change, because the
   clients genuinely are near-identical.
3. **Small objects: try anchor sizes, not RPN unfreezing.** And remember
   unfreezing is a two-part change — `build_optimizer` makes one param group
   at `LR_HEAD = 0.01` and `LR_PRETRAINED` is dead code.
4. Seed 7 under this config, to confirm none of the above is seed noise.
