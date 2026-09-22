# Second opinion: v2 → v3 and the gossip anomaly

Grounded against `gossip.py`, `train.py`, `config.py`, `gossip_saturation_finding.md`,
and `rpn_unfreeze_plan.md` (= `v3.md` §"Why v3") before writing this, not just the
headline numbers. Citations below are file:line.

## 1. Is "anchors were too coarse, halving + training the RPN fixed it" the right read?

The evidence for the *anchor-scale* part is stronger than a post-hoc AR_S/AR_L split —
it was predicted in advance, not fitted after the fact: median training box is 28.3px,
55.6% of boxes fall under the old smallest anchor (32px), and the eval "small" band
(<32×32) is defined to be exactly that anchor (`v3.md` §"Why v3"). That's a real,
pre-registered mechanism, not a just-so story built from the AR_S jump.

But the two changes were bundled, and the plan doc's own reasoning explains why an
"anchors-only" ablation isn't meaningful: halving the anchors without retraining the
RPN makes the pretrained objectness/regression weights actively wrong for the new
scale (`rpn_unfreeze_plan.md:61-63`), so that variant is expected to be *bad*, not a
useful control. Fine — but that doesn't make the bundle un-decomposable. The
ablation that **is** coherent and wasn't run: **unfreeze + train the RPN with the old
anchors (32–512)**, same optimizer groups, same warmup. That isolates "RPN gets to
adapt to BDD100K's domain" (different camera geometry, resolution, appearance vs.
COCO) from "RPN gets priors that match the box-size distribution." Both predict the
same qualitative signature you observed — AR_S up, AR_L flat — because COCO's RPN is
generically undertrained for BDD100K regardless of anchor scale. The size-distribution
argument only wins if AR_S under old-anchors-trained-RPN moves noticeably less than
what you saw (0.2412 → 0.2839). Cheap to check: it's the same joint-only run the plan
doc already uses as its step-2 gate (~3.5h), just with `model.py`'s anchor list
reverted.

Until that ablation runs, "anchors were the fix" is one hypothesis consistent with the
data, not the only one. I'd hold it loosely.

## 2. Is gossip's flat result fully explained by participation, or is the gate doing damage too?

They're not cleanly separable, and I don't think the framing of "which one is the
cause" is quite right — here's the number that matters more than either:

`alpha = lambda * gate` (`gossip.py:221`), and in the saturated window `alpha` averaged
**0.4874** (`gossip_saturation_finding.md:72`). That means each gossip operation moves
`clear` ~49% of the remaining distance toward the mixed-in state. For a
quasi-fixed target (four frozen, mutually-similar neighbours — TV ~0.05 apart), the
distance remaining after *n* operations decays like `(1-0.4874)^n`:

```
n=1   51%      n=4   6.9%      n=10  0.13%
```

Saturation is complete in **~10 operations**, not 224. So the gate's near-uniform
pass-through (`weight_mean` 0.97–0.98) matters less as "how often does it fail to
catch a bad neighbour" and more as "when it does pass a neighbour, it passes them at
almost the full dose, every time, with no memory of how many times this has already
happened." A gate that discriminated harder on *distributional* badness genuinely
might not fire here — the neighbours aren't distributionally bad, the cells are
homogeneous (TV ~0.05), which is a separate, previously-established fact. What the
gate has no concept of at all is **staleness**: a neighbour frozen at the end of its
own training and a neighbour mid-training with an identical current loss look
identical to `gate_weight()` (`gossip.py:125-138`), which only ever compares losses,
never checks whether the neighbour has moved since it was last read.

So: participation explains *exposure* (224 ops instead of ~20), the gate's blindness
to staleness explains why each of those ops did real damage instead of being inert.
Given the saturation math, fixing exposure alone won't help much once you're above
~10-15 ops — which any realistic partial fix still leaves you well over (see Q3). To
separate them experimentally: run current (buggy) participation but with a synthetic
staleness penalty added to `gate_weight` (e.g., decay by rounds-since-last-update of
the neighbour's checkpoint) and see whether `clear`'s deficit recovers even with the
same 224-op exposure. If yes, the gate's missing staleness signal is the dominant
lever, and the participation bug is "just" what created the exposure for it to act on.

## 3. Concerns with the proposed fix (fixed per-client Bernoulli rate)

Two separate concerns, one arithmetic and one structural.

**Arithmetic:** `rate = GOSSIP_PARTICIPATION / len(CLIENTS) = 0.6` cuts operations in
the tail window from ~224 to an expected ~135 (60% of 225 slices). Given the
saturation math above (full convergence in ~10 ops), 135 ops is still ~13x more than
needed to fully wash `clear` toward the frozen neighbour pool. This fix reduces
*exposure* but not below the saturation threshold, so I'd expect it to barely move
`clear`'s deficit — it thins out a mixing process that was already complete after the
first few minutes of the solo-survivor window.

**Structural:** `gossip_saturation_finding.md`'s own candidate #2 — `participation =
min(GOSSIP_PARTICIPATION, len(active) - 1)` — is a better fix than the Bernoulli
version, and not just marginally. At `len(active)=1`, that formula gives `min(3, 0) =
0`: a client alone in the network never gossips, full stop. The fixed-rate version
(`GOSSIP_PARTICIPATION / len(CLIENTS)`) still fires 60% of the time even when `active
= [clear]` only, because it doesn't condition on the active-set size at all — it uses
the *total* client count as a constant denominator. It solves the wrong failure mode:
it would smooth out an *overrepresentation among several active clients*, but the
actual bug is a *lone survivor with literally nobody to gossip with as an equal*. The
`-1` variant hard-stops exactly that case; the Bernoulli(0.6) version doesn't stop it,
it just slows it down by 40%, well short of where saturation stops mattering.

I'd take the `-1` formula (or an explicit `if len(active) <= 1: skip`) over the fixed
Bernoulli rate. If you want stochasticity for the multi-active-client case specifically
(to avoid whatever determinism concern motivated moving off top-k `rng.sample` in the
first place — worth stating explicitly, since top-k sampling isn't obviously broken
when `len(active) > 1`), scale the Bernoulli rate by `len(active)` instead of the fixed
client count, so it also collapses to 0 as `active` shrinks to 1.

Either way, per Q2, I'd pair whichever participation fix you pick with *something* on
the dose side (staleness-aware gate, or a hard cap on consecutive-same-neighbour mixing
regardless of participation), because participation-only fixes are capped in how much
they can recover once you're in double-digit operation counts.

## 4. Where to spend effort next: backbone, more training, or revisit CL/federation

Pushing back on the framing a bit first. `joint` (0.2854) vs `local_only` (0.2550) is
being read as "how much is there to gain by handling heterogeneity/federation
better" — but given TV ~0.05, the clients are close to IID, and IID clients don't need
distribution-aware federation to match a pooled model; they mostly need *more
effective data per client*. `joint` gets ~2.7x `clear`'s dataset and ~7-8x the smallest
clients'. If most of that 0.03 mAP gap is concentrated in the small clients (partly
cloudy, rainy, snowy) rather than spread evenly, that's a sample-size story, not a
heterogeneity-handling story — and no amount of smarter CL or gate design closes a
sample-size gap; only actually sharing more signal does (which, incidentally, is
gossip's whole job, currently untestable because of the Q2/Q3 bug). That's a five-line
groupby you already have the data for (`local_only` vs `joint`, per client, v3
checkpoints) and I'd run it before committing to any of the three options — it tells
you whether the gap is a `clear`-sized problem or a small-client-sized problem, which
points at different fixes.

On the three options as stated:

- **Backbone unfreeze**: I'd rank this above generic "more training." The backbone is
  COCO-pretrained and frozen through v2 and v3; BDD100K's night/dawn-dusk tasks are
  exactly the condition where COCO's daytime-dominated feature statistics are least
  likely to transfer, and RPN unfreezing already bought +0.03 mAP on `joint` mostly
  through small-object recall (§1) — the backbone is the next most obviously
  domain-mismatched frozen component, and a plausible place a comparable jump is
  still sitting. Caveat: your smallest client (partly cloudy, 4,496 images) is at real
  overfitting risk if backbone unfreezes on top of an already-trained RPN + head with
  no augmentation change mentioned — worth checking per-client train/val loss curves
  for divergence before committing, not just running it and reading the aggregate.
- **More training/epochs/LR/augmentation**: worth doing but I wouldn't lead with it
  absent evidence of underfitting (e.g., loss curves still descending at cutoff). You
  don't have that evidence in what's here — check it first, it's free.
- **Revisiting the CL/federation setup because of homogeneity**: I wouldn't do this
  yet, for the reason above — you don't actually know how much federation is worth
  because every gossip run so far (v1, v2 both variants, v3) has hit a degenerate
  participation regime for the largest client (`gossip_saturation_finding.md:130-136`).
  "Is federation worth it given TV~0.05" is a fair question to revisit, but not one
  the current numbers can answer yet.

**Priority order I'd actually run, cheapest-diagnostic-first:**
1. `local_only` vs `joint` per-client breakdown (no new runs, data you have) — tells
   you if the ceiling gap is a `clear` problem or a small-client problem.
2. Fix gossip participation (§3) + re-run gossip vs. `local_only` — this is the one
   result you currently cannot trust at all, and it's cheap relative to a training run.
3. RPN-only-with-old-anchors ablation (§1) if you want the anchor claim on solid
   ground before it goes in anything permanent.
4. Backbone unfreeze, gated on watching small-client overfitting.

## Correction: the "gossip barely moved" headline number mixes two different gossip rules

This resolves the loose thread from the first pass, by reading the actual result
files (`OBJ_DET/results/runs/`, `OBJ_DET/v3/results/runs/`) rather than guessing at
aggregation methodology. It changes the story more than expected.

The v2 headline (`gossip 0.2398`) traces exactly to
`gp3fo2_gossip_seed42.json` (system mAP mean 0.23977) — the variant where **finished
clients are excluded as gossip sources**. But `gossip.py`'s current `sample_neighbors`
(`gossip.py:265-281`) hardcodes the opposite rule — finished clients stay eligible as
sources — which is what v3's `a16rpn_gossip_seed42.json` (0.24058) actually ran. v2
also has a same-rule run sitting right next to it: `gp3fo2fs_gossip_seed42.json`,
mean **0.2345**, not 0.2398.

```
gp3fo2   (v2, finished EXCLUDED as source): 0.2398   <- what the headline table uses
gp3fo2fs (v2, finished INCLUDED as source): 0.2345   <- the actual rule-matched v2 baseline
a16rpn   (v3, finished INCLUDED as source): 0.2406

delta as currently reported (gp3fo2 -> a16rpn):   +0.0008  "flat, uniquely anomalous"
delta on a matched source rule (gp3fo2fs -> a16rpn): +0.0061  "roughly tied with sequential (+0.0064)"
```

So the table in `gossip_saturation_finding.md:9-13` (and `v3.md:45`'s "v2 closed
gossip's deficit... 0.2398 vs local_only 0.2398") is differencing two numbers that
don't share a gossip rule, not just a detector. Once you hold the source-inclusion
rule fixed, gossip's v2→v3 gain (+0.0061) is unremarkable — in the same range as
`sequential`'s — not the outlier that motivated treating gossip as uniquely broken by
the detector change.

**What this does *not* undo:** the saturation mechanism itself (224 deterministic
gossip ops for `clear` once it's the sole active client, `alpha` averaging 0.4874,
gate near-inert) is confirmed independently in both `gp3fo2fs` (v2) and `a16rpn` (v3)
with *identical* op counts and schedule (`gossip_saturation_finding.md:104-110`) —
that's a real, seed-driven artifact of the participation rule, not a consequence of
which baseline you diff against. Sections 2 and 3 above stand as written.

**One thing this correction does surface as newly interesting**: `clear`'s per-client
gossip-vs-local_only deficit nearly doubled in absolute mAP terms between the matched
runs, -0.0227 (v2, `gp3fo2fs`) → -0.0412 (v3, `a16rpn`), despite identical operation
counts and identical `alpha` schedule. That's not explained by anything above — a
fixed relative pull toward the same frozen-neighbour pool shouldn't cost more in
absolute mAP just because the detector got better, unless the v3 detector's
local-only ceiling pulled away from what the frozen neighbours represent faster than
the neighbours themselves improved. Flagging as unresolved, not asserting an
explanation.

**Practical fix**: when you re-run gossip after the participation fix, diff it against
`gp3fo2fs`/`a16rpn`-style numbers (matched source-inclusion rule), not `gp3fo2`. And
worth a one-line note in `v3.md`'s motivation section correcting "v2 closed gossip's
deficit" — it didn't; that comparison was against the variant where `clear` mostly
couldn't gossip at all (84% of slices, no live neighbour), a different failure mode
than the one this whole investigation is about.
