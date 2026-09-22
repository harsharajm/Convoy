# Finding: `clear` is gossip-saturated, and it explains gossip's flat result

Seed 42, `a16rpn_gossip` (v3, tag `a16rpn`). Not yet true for v2 - checked
below.

## The number that doesn't fit

```
mode         v2      v3      delta
joint      0.2568  0.2854   +0.0286
local_only 0.2398  0.2550   +0.0152
sequential 0.2394  0.2458   +0.0064
gossip     0.2398  0.2406   +0.0008
```

The anchor fix helped every mode except gossip. `joint` gained nearly triple
what `gossip` did, on the same detector change. That is backwards: gossip
runs the same training as `local_only` plus mixing, so a better detector
should help it at least as much.

## Per-client, gossip vs local_only (v3)

```
client          images   gossip  local   delta
clear           34,151   0.2260  0.2673  -0.0412
overcast         8,008   0.2751  0.2836  -0.0085
snowy            5,054   0.2396  0.2508  -0.0112
rainy            4,638   0.2265  0.2334  -0.0069
partly cloudy    4,496   0.2357  0.2397  -0.0040
```

`clear` alone accounts for most of gossip's shortfall against `local_only`,
by a wide margin over every other client.

## The mechanism

```
gossip_ops : clear 260, overcast 29, rainy 24, snowy 22, partly cloudy 13
local_steps: clear 5340, overcast 861, rainy 654, snowy 708, partly cloudy 480
```

`clear` gossiped 9-20x more often than any other client - far beyond what its
6.2x larger step count would predict on its own.

The cause is in the participation rule, `train.py`:

```python
active = [c for c in clients.values() if not c.finished]
participants = rng.sample(active, min(GOSSIP_PARTICIPATION, len(active)))
```

`GOSSIP_PARTICIPATION = 3` clients are sampled from whoever is still active.
This is fine while five clients are active - each has roughly a 3/5 chance
per slice. It stops being fine once clients start finishing, because
`min(3, len(active))` shrinks with them:

```
task 2 finish times:  partly cloudy 22:43   rainy 23:26
                       snowy 23:39   overcast 23:59   clear 04:38
```

From 23:59 onward, `clear` is the only active client. `active = [clear]`,
so `min(3, 1) = 1` and `rng.sample` has nothing to choose between - `clear`
is selected **every single slice**, deterministically, for the rest of the
run. And per the v2 -> `gp3fo2fs` fix, finished clients stay eligible as
gossip *sources* (`design_choices.pdf` 2.2.1), so `clear` always has someone
to pull from, even though nobody is left to pull from it.

```
slices after the last other client finished (23:59:23): 225
clear's gossip ops in that window:                       224  (all "gossip done")
alpha in that window: mean 0.4874, min 0.4562, max 0.5000, n=224
```

So for the back half of the run - all of task 1 (~4h) and part of task 2 -
`clear` mixed toward frozen, weaker, permanently-stale weights roughly every
20 steps, at 97% of full lambda dose each time (the gate barely discriminates
here; see below). It never got a chance to specialise on its own 7.6x larger
share of the data, because it was structurally unable to stop gossiping once
it became the last one standing.

This is the same finished-client asymmetry fixed once already, showing up in
the opposite direction. `gp3fo2` (finished excluded as *sources*) silenced
`clear`'s gossip entirely once its neighbours were gone. `gp3fo2fs` /
`a16rpn` (finished included as sources) fixed that - and created this: with
sources always available, `clear` cannot stop gossiping either, and ends up
gossiping *more*, not less, exactly when it has the least to gain from doing
so.

## Why the gate didn't catch it

`weight_mean` in the gate summary is 0.97-0.98 across both v2 and v3 runs -
neighbours pass at near-full strength almost always. `excess_median` sits at
0.006-0.010. A frozen, weaker, finished neighbour is not scoring badly enough
on `clear`'s gate batch to be meaningfully down-weighted. Two established
causes compound here: the cells are near-identical (heterogeneity check, TV
~0.05), and gossip itself erases whatever gap exists between clients before
it can be measured (`gate_noise_probe.py`: two never-gossiped clients differ
by 7.7% on the same metric that only reads 0.6-1.0% inside a running gossip
loop). A finished neighbour frozen at a *worse* point should in principle be
an easy case for the gate - but the signal is too flat for that margin to
show up as a real discount.

## Confirmed: v2 has the same problem

`gp3fo2fs_gossip` (v2, finished-as-sources) shows the identical
`gossip_ops` distribution - `clear 260, overcast 29, rainy 24, snowy 22,
partly cloudy 13` - because the gossip schedule is a function of the seed and
`local_steps`, not of the detector, so v2 and v3 run through the exact same
sequence of participation draws.

```
client          gossip  local   delta
clear           0.2190  0.2417  -0.0227
overcast        0.2644  0.2635  +0.0009
rainy           0.2196  0.2227  -0.0031
snowy           0.2322  0.2354  -0.0032
partly cloudy   0.2374  0.2357  +0.0017
```

`clear` again carries almost the entire deficit, by 7-20x over any other
client. `gp3fo2` (finished excluded as sources) does not have this problem,
but has the opposite one documented in `gp3fo2_results.md` - `clear` barely
gossips at all once its neighbours finish. Between the two v2 variants and
v3, every gossip run tried so far mishandles the same client for one of two
opposite reasons.

## What this means for every gossip number so far

None of `v1`, `v2`, or `v3`'s gossip runs are a fair test of the gossip
*mechanism* as designed. In each version, the largest client - the one with
the most to teach and, on 7.6x the data, the most to lose - has ended up in a
degenerate participation regime: either excluded from gossip almost entirely
(`gp3fo2`) or unable to stop gossiping once everyone else finishes
(`gp3fo2fs`, `a16rpn`). The gate exists to prevent exactly the second failure
and is not currently strong enough to do it.

## Candidate fixes, not yet chosen

1. **Cap gossip operations per client**, independent of participation
   sampling - e.g. a max ops budget proportional to that client's own step
   count, so a lone survivor stops initiating once it has gossiped its share.
2. **Scale participation to the active set**, not a flat 3 - e.g.
   `participation = min(GOSSIP_PARTICIPATION, len(active) - 1)` so a client
   alone in the network is never forced to be a guaranteed participant.
3. **Stop treating finished clients as sources once the active set drops
   below some threshold** - a middle ground between `gp3fo2`'s exclude-always
   and `gp3fo2fs`'s include-always.
4. **Fix the gate's discrimination first** (`GATE_SENSITIVITY`, or the
   mixing-erases-signal loop in `gate_noise_probe.py`) and let a working gate
   handle this on its own - a finished, frozen, weaker neighbour is the
   textbook case a gate should suppress.

These are not mutually exclusive. (1) and (2) fix the exposure; (4) fixes
whether the exposure would have mattered anyway. Needs a decision before the
next gossip run - otherwise the same mechanism repeats.
