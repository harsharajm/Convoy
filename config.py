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
FREEZE_RPN = True

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
LR_PRETRAINED = 0.0         # unused while frozen; kept so unfreezing is one edit
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_WARMUP = None            # design_choices.pdf: none
AUGMENTATION = None         # design_choices.pdf: none

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
