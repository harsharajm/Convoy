# PPT Updates

Running list of changes to make to the presentation (originally scoped in `slides.md`).
Each numbered item is one update.

## 1. Add: Classes — % of images each class appears in

Where: the "Class Overlap" / BDD100K-vs-COCO slide (or a new slide right after it) should
also show, for each of the 10 BDD100K object classes, what percentage of images contain
at least one box of that class — not just instance count, but image-level presence.
This shows how skewed/long-tailed the class distribution is (e.g. `car` is in almost
every image, `train` is in almost none), which motivates the rarity-weighted replay
buffer described elsewhere in the deck.

Computed over the full usable pool: 70,426 images (5 weather clients x 3 timeofday
tasks, from BDD100K's train+val label JSONs combined, "foggy"/"undefined" excluded).
An image counts once per class regardless of how many boxes of that class it contains.

| Class | % of images | Images | Instances |
|---|---|---|---|
| car | 98.93% | 69,669 | 722,924 |
| traffic sign | 82.24% | 57,918 | 242,795 |
| traffic light | 56.23% | 39,598 | 188,193 |
| person | 29.06% | 20,469 | 78,121 |
| truck | 26.45% | 18,627 | 29,433 |
| bus | 12.23% | 8,614 | 11,135 |
| bike | 5.33% | 3,757 | 6,186 |
| rider | 4.63% | 3,264 | 4,103 |
| motor | 3.09% | 2,175 | 2,863 |
| train | 0.16% | 113 | 145 |

Suggested framing on the slide: sort descending by %, and call out `car` (near-universal)
and `train` (0.16% — the class the replay buffer's eviction policy is specifically
protecting) as the two extremes.

## 2. Add: raw metadata structure example

Where: the "metadata structure" slide (`slides.md` line 3-4 — showing that BDD100K's
per-image metadata has only 2-3 attributes it can be partitioned by).

Show this snippet as the visual example of one image's metadata, straight from
`bdd100k_labels_images_train.json`:

```json
"name": "0000f77c-6257be58.jpg",
"attributes": {
  "weather": "clear",
  "scene": "city street",
  "timeofday": "daytime"
},
```

Point to make: `attributes` has exactly 3 fields — `weather`, `scene`, `timeofday`. The
deck partitions on `weather` (client axis) and `timeofday` (task axis); `scene` exists
but was deliberately not used (see class-overlap/partition-axis discussion for why).

## 3. Add: gate sensitivity formula (gossip gating)

Where: the "gossip" slide — after `GOSSIP_LAMBDA` / `GATE_SENSITIVITY` are introduced as
hyperparameters, show the actual gate math so it's clear what those two knobs control.

Each client gates a neighbour's weights on its own replay memory before mixing. The
neighbour's own loss on that memory (`loss_own`) is compared against the neighbour's loss
on the same memory (`loss_neighbour`):

```
excess      = max(0, (loss_neighbour - loss_own) / loss_own)
gate_weight = exp(-sensitivity * excess)
alpha       = lambda * gate_weight
```

- `excess` is 0 when the neighbour is at least as good as me on my own data (so
  `gate_weight = 1`, full trust), and grows the worse the neighbour scores, normalised by
  my own loss so the gate behaves the same early in training (large losses) as late
  (small losses).
- `sensitivity` (`GATE_SENSITIVITY = 2.0`) controls how fast trust decays with `excess` —
  higher sensitivity punishes a bad neighbour harder.
- `lambda` (`GOSSIP_LAMBDA = 0.5`) is the ceiling: even a neighbour that passes the gate
  at full strength (`gate_weight = 1`) only moves me by `alpha = lambda`, never fully
  replacing my weights.
- Final mixing step: `model <- (1 - alpha) * model + alpha * neighbour`, applied to the
  trainable head parameters only.

Source: `gate_weight()` and `mix_into()` in `gossip.py`.

## 4. Elaborate: Precision-Recall curve (evaluation slide)

Where: the "Evaluation Pipeline" slide, right after IoU / TP / FP are defined and the
Precision/Recall formulas are given (`slides.md` lines 71-76) — before AP is introduced.

**Constructing the curve.** Fix an IoU threshold (this decides what counts as a TP vs FP
in the first place). Sort every prediction across the eval set by confidence score,
descending. Walk down that sorted list one prediction at a time, accepting predictions in
that order; after each one, recompute cumulative precision and recall. Plotting precision
against recall at every step traces the PR curve. Walking the list this way sweeps the
confidence threshold from 1 down to 0 — at the start (only the single most-confident
prediction accepted) precision is high but recall is near 0; by the end (every prediction
accepted, however low-confidence) recall reaches its maximum but precision has typically
fallen, since low-confidence predictions are disproportionately wrong.

**Why Recall on X, Precision on Y — not the reverse.** Recall = TP/(TP+FN), and FN+TP is
fixed (it's just "total ground-truth boxes") — so as the list is walked and predictions
are only ever added, TP can only stay the same or grow, meaning recall is monotonically
non-decreasing. It behaves like a clean sweep variable, 0 to 1, which is exactly what an
x-axis needs. Precision has no such guarantee — adding one more prediction can raise or
lower it depending on whether that prediction was correct, so it wiggles up and down as
recall increases. That makes precision the *measured/dependent* quantity you're watching
respond to the sweep, which is the natural y-axis role. (Same convention as an ROC curve:
the monotonic-under-threshold quantity goes on the x-axis.) This also matches
precision/recall's original information-retrieval framing — recall as "how much of the
relevant set did we cover" (x, the thing you're pushing), precision as "how clean is the
result at that coverage" (y, the thing that suffers as you push).

**It's a slice of a 3D surface, not a standalone 2D curve.** The PR curve above was built
at *one* fixed IoU threshold. IoU threshold is a third axis: pick a different threshold
(0.5 vs 0.75 vs 0.9) and the TP/FP labeling of the exact same raw predictions changes —
stricter IoU throws out more boxes as FP, so both precision and recall at a given
confidence cutoff tend to drop. So the full picture is a surface, precision as a function
of (recall, IoU threshold); each IoU threshold slices out one 2D PR curve from it. AP is
the area under one such slice (e.g. AP@0.5 = area under the IoU=0.5 slice, using the
interpolated precision envelope). AP@[0.5:0.95] — COCO's headline metric — averages AP
over 10 evenly-spaced slices (IoU = 0.50, 0.55, ..., 0.95), so it scores localization
tightness as well as classification, not just "did the box roughly overlap."

Suggested visual: a 3D axes sketch (Recall x, Precision y, IoU threshold z) with 2-3 PR
curves drawn as stacked slices at different depths, front slice labeled IoU=0.5 and back
slice IoU=0.95, back slice visibly lower/tighter than the front one.

## 5. Elaborate: gossip hyperparameters (topology / fanout, interval, payload)

Where: the "gossip" slide, alongside item 3's gate-formula addition — this covers the
*other* knobs (who talks to whom, how often, what's sent), not the gate math itself.

**Topology / fanout — `sample_partner()` in `gossip.py`.** Classic gossip protocols
expose a fanout parameter: how many peers a node contacts per round. This design fixes
fanout at **1** — every gossip event is strictly pairwise, one client and one uniformly-
random partner drawn from the other 4. There's no explicit `FANOUT` constant in
`config.py` because it's architectural, not tunable: the 5 clients (`clear`, `overcast`,
`rainy`, `snowy`, `partly cloudy`) form a complete graph (any client can pair with any
other, no fixed neighbour list), and each event moves exactly one pair. Worth stating
explicitly on the slide since "fanout" is the term the gossip-averaging literature uses
and its absence here is a deliberate choice, not an oversight.

**`GOSSIP_EVERY_N_STEPS = 20`** — how often a client *initiates* gossip, measured in its
own local training steps, not a shared/global round. There is no global clock in a
decentralised setting (see the note already in `slides.md`), so this is per-client
pacing: a client with a large data cell takes more local steps and therefore gossips more
often in wall-clock terms than a client with a small cell, which is the intermittent,
uneven participation the setting is supposed to have. Too large and a client can finish
an entire task alone before ever gossiping; too small and gate evaluations (two forward
passes per event) start dominating training time.

**`GATE_BATCH_SIZE = 32`** — size of the batch each side scores the other on (`gate_batch()`
in `train.py`). Drawn from the client's own replay buffer if it has samples in it, else
from its current task batch — always the client's *own* data, since the gate question is
"does this neighbour help me here," never "on some shared/global set."

**`GOSSIP_LAMBDA = 0.5` and `GATE_SENSITIVITY = 2.0`** — the mixing ceiling and gate decay
rate; full derivation already covered in item 3 above, listed here just so this slide has
the complete hyperparameter set in one place.

**Payload — not a hyperparameter, but worth a line on the same slide:** only head
parameters travel per gossip event (backbone/FPN/RPN are frozen and identical across
clients, so shipping them would be a ~40x larger no-op). This is why fanout=1 is cheap
enough to do every 20 steps — the message is small.

**`GATE_CHUNK_SIZE = 64`** exists in `config.py` but is currently unused — not referenced
by `gossip.py` or `train.py`. Leave it off the slide (or footnote as reserved/not wired
in yet) rather than presenting it as an active mechanism.

Suggested visual: a small 5-node complete-graph diagram (one node per weather client) with
one edge highlighted to show a single pairwise gossip event, labeled with the message
size (head-only) and the trigger condition (every 20 local steps, per-client clock).
