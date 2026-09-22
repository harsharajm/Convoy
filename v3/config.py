"""
Every constant the pipeline runs on, in one place.

Values come from design_choices.pdf (training) and
READTHESE/evaluation_pipeline/EVAL_PIPELINE_DETECTION.md (evaluation).
The evaluation thresholds live in evaluate_detection.py, not here - the spec
freezes them and nothing outside that file may reach in and change them.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- data ------------------------------------------------------------------

# Already-partitioned manifest (filename, weather, timeofday, split) and the
# per-image box lists. Both were generated once from the raw label JSONs.
MANIFEST_PATH = os.environ.get(
    "BDD100K_MANIFEST", os.path.join(_HERE, "bdd100k_manifest.csv"))
BOXES_INDEX_PATH = os.environ.get(
    "BDD100K_BOXES_INDEX", os.path.join(_HERE, "boxes_index.json"))

# Root holding images/100k/{train,val}/... . Differs between the local box and
# the cluster, which is why the image index stores paths *relative* to this
# root and never absolute ones.
#
# BDD100K_IMAGES_ROOT still wins if set. Otherwise, first of these that
# actually exists on this machine - local box first, then the SLURM cluster.
_KNOWN_IMAGES_ROOTS = [
    "/home/breach/Desktop/DecentralAutonomousDriving/BDDTA/bdd100k_data/images",
    "/u/student/2025/cs25mtech11011/BDDTA/bdd100k_data/images",
]
IMAGES_ROOT = os.environ.get("BDD100K_IMAGES_ROOT") or next(
    (p for p in _KNOWN_IMAGES_ROOTS if os.path.isdir(p)), _KNOWN_IMAGES_ROOTS[0])
IMAGE_INDEX_PATH = os.environ.get(
    "BDD100K_IMAGE_INDEX", os.path.join(_HERE, "image_index.json"))

# 100k/test/ has no label JSON (BDD100K withholds test ground truth for its
# leaderboard), so it is never indexed.
INDEX_SPLIT_DIRS = ["100k/train", "100k/val"]

OBJECT_CLASSES = [
    "car", "traffic sign", "traffic light", "person", "truck",
    "bus", "bike", "rider", "motor", "train",
]
# torchvision detection reserves label 0 for background.
CATEGORY_TO_LABEL = {cat: i + 1 for i, cat in enumerate(OBJECT_CLASSES)}
NUM_CLASSES = len(OBJECT_CLASSES) + 1   # 11, background included

CLIENTS = ["clear", "overcast", "rainy", "snowy", "partly cloudy"]
TASKS = ["daytime", "night", "dawn/dusk"]

# --- model -----------------------------------------------------------------

# COCO's classes cover almost all of BDD100K's under different names, so the
# pretrained features already fit this domain. Only the head has real work to
# do, which is why everything below it is frozen.
FREEZE_BACKBONE = True      # ResNet + FPN
# Partial-freeze override, matching MMDetection's `frozen_stages` convention
# (see the BDD100K reference Faster R-CNN configs: `frozen_stages=1`) - freeze
# only the stem (conv1/bn1) plus `layer1..layer{N}`, leaving later ResNet
# stages and the FPN trainable. None keeps the old all-or-nothing behaviour
# (FREEZE_BACKBONE alone decides whether the whole backbone+FPN is frozen).
# Set to 1 or 2 to test how much of the backbone actually needs to unfreeze
# to close the gap with the reference results - a variable FREEZE_BACKBONE
# alone cannot express.
FREEZE_BACKBONE_STAGES = None    # None | 1 | 2
# v3: the RPN trains. Halved anchors (below) mean its pretrained weights no
# longer match the anchors they were learned against, so leaving it frozen
# would be worse than either v2 or a retrained RPN. The backbone stays frozen
# - one variable at a time.
FREEZE_RPN = False

# v3: halved from torchvision's (32, 64, 128, 256, 512).
#
# The evaluation's "small" band is area < 1024px^2 = 32x32, which is exactly
# the old smallest anchor: every object counted as small sat at or below the
# smallest anchor and was never proposable. Measured over 1,027,770 training
# boxes, the median box is 28.3px - smaller than that anchor - 55.6% fall
# below it, and 0.09% exceed 512px, so the top level was nearly dead weight.
#
# A plain halving rather than a fit to those percentiles: "driving scenes have
# smaller objects than COCO" needs no defending, and nothing here is derived
# from the test split. One size per level and three ratios keeps num_anchors
# at 3, so the pretrained RPN head's conv shapes still load.
ANCHOR_SIZES = (16, 32, 64, 128, 256)
ANCHOR_ASPECT_RATIOS = (0.5, 1.0, 2.0)

# The frozen RPN's default anchors were tuned against COCO's object sizes;
# BDD100K skews much smaller (55.5% of boxes < 32px). Widening how many
# proposals survive to the head is a inference-time knob, not a weight
# change, so it is the first thing to try before unfreezing anything: if
# small-class AP recovers, the RPN's weights were never the bottleneck, only
# how many of its proposals were being kept.
RPN_PRE_NMS_TOP_N_TEST = 2000    # torchvision default: 1000
RPN_POST_NMS_TOP_N_TEST = 2000   # torchvision default: 1000

# --- optimisation ----------------------------------------------------------

EPOCHS_PER_TASK = 3
BATCH_SIZE = 32
NUM_WORKERS = 4
USE_AMP = True

LR_HEAD = 0.01              # new head, learning from scratch
# v3: live for the first time. Through v1 and v2 this was defined and never
# imported - build_optimizer made a single group at LR_HEAD - so unfreezing
# the RPN without also fixing build_optimizer would have handed COCO's RPN
# convs the same unwarmed 0.01 as a random head.
LR_PRETRAINED = 1e-3        # RPN: pretrained, and only re-adapting to anchors
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
# v3: linear warm-up, a deliberate deviation from design_choices.pdf ("none").
#
# The reason is the halved anchors, not the random head. torchvision detaches
# the proposal path (rpn.py:371, `pred_bbox_deltas.detach()`), so head
# gradients never reach the RPN - the two are gradient-isolated and the random
# predictor cannot disturb the pretrained convs. What *can* is the anchor
# change itself: the RPN's own objectness and box losses are computed against
# anchors it was never trained on, so its gradients are large and mispointed
# for the first steps.
#
# Kept short because the budget is short. The smallest client trained 480
# steps in gp3fo2, so 100 is already 21% of its run; joint gets ~5280 steps,
# where it is 2%. Anything like torchvision's usual 1000 would be most of a
# small client's training.
LR_WARMUP = 100             # steps of linear ramp, per optimiser
LR_WARMUP_FACTOR = 0.01     # starting multiple of the target lr
AUGMENTATION = None         # design_choices.pdf: none

# joint only (train.py's run_joint `epochs=` override) - the federated modes
# have a communication-round budget to respect and keep reading
# EPOCHS_PER_TASK above; joint has none, so its epoch count and decay are
# separate knobs. Milestones are fractions of whatever `epochs` a given joint
# run uses, not raw epoch numbers, mirroring the reference recipe's
# step=[8, 11] at 12 epochs (2/3, 11/12 of the schedule) - so an 8-epoch run
# decays near epochs 5 and 7 instead of firing at fixed epochs tuned for a
# 12-epoch run.
LR_DECAY_MILESTONES = (2 / 3, 11 / 12)
LR_DECAY_FACTOR = 0.1

# --- replay buffer ---------------------------------------------------------

# Entries are image *references* (filenames), never decoded tensors - 2000
# full-res tensors would be ~22 GB per client.
BUFFER_CAPACITY = 2000
REPLAY_RATIO = 1.0          # current-task : replay = 1:1

# --- gossip ----------------------------------------------------------------

GOSSIP_LAMBDA = 0.5         # weight on the neighbour aggregate
GATE_SENSITIVITY = 2.0
GATE_BATCH_SIZE = 32
GATE_CHUNK_SIZE = 64
# There is no global round and no shared clock - that vocabulary comes from
# server-based FL and does not survive into a decentralised system. Each
# client counts its *own* local steps and initiates gossip every this many.
#
# So clients with large cells gossip often and clients with tiny cells gossip
# rarely, which is the intermittent participation the setting actually has,
# not an artefact. Too large and a client trains a whole task alone (and
# PROBE_INTERVAL never fires); too small and gate evaluations dominate.
GOSSIP_EVERY_N_STEPS = 20

# Neighbours a gossiping client reads per operation. fanout=1 makes the
# per-neighbour weight normalisation in gossip.py a no-op (w / total_w == 1
# always), which is why the discrimination between neighbours was invisible
# before this was ported.
GOSSIP_FANOUT = 2
# The scheduler has no global round either, only "slices" (one pass over
# every still-training client - see train.py's run loop). This is the
# nearest thing to the reference's per-round `comm_clients` sample, so
# participation is read as "how many clients gossip per slice".
GOSSIP_PARTICIPATION = 3

# --- spatial interference probe (Layer F) - DROPPED for now ----------------
#
# The 5x5 spatial interference matrix is not being collected in this round of
# work. At this run length there were not enough gossip operations to sample
# it meaningfully anyway: 3 epochs over the smaller cells is only a few
# hundred steps, giving 21-37 gossip operations per client, so any probe
# interval coarse enough to be cheap left four of five clients with zero
# measurements.
#
# Consequence to be aware of: Layer F was the only direct test of whether the
# gossip gate actually down-weights unhelpful neighbours. Without it, nothing
# in the pipeline reveals a gate that is silently doing nothing.
#
# PROBE_SIZE = 200
# PROBE_INTERVAL = 100

# --- run modes -------------------------------------------------------------

# The three baselines EVAL_PIPELINE_DETECTION.md 8 requires, plus the real run.
MODES = {
    "gossip":     {"gossip": True,  "replay": True},
    "local_only": {"gossip": False, "replay": True},
    "sequential": {"gossip": False, "replay": False},
    "joint":      {"gossip": False, "replay": False},   # all data, no CL
}

SEED = 42
