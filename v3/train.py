"""
Decentralised gossip-based federated continual learning on BDD100K detection.

Five clients (weather), three sequential tasks each (timeofday). No server and
no shared clock: every client trains its own task, and periodically gossips
with one randomly chosen peer. Clients whose cells are small finish earlier
and simply keep serving their weights to whoever is still training.

There is no global round here and no shared clock. Each client counts its own
local steps and initiates gossip every GOSSIP_EVERY_N_STEPS of them, so a
client with a large cell gossips far more often than one with a small cell.
The scheduler below interleaves clients only because this runs on one
machine - it is not a synchronisation barrier, and no client waits on another
except for the peer inside a gossip operation itself.

Per client:

    for task in 0..2:
        train on current task + replay          # 1:1
        every GOSSIP_EVERY_N_STEPS local steps:
            gossip: read GOSSIP_FANOUT neighbours, gated, pull-only
        task boundary: evaluate tasks 0..task, append to acc_history

Each scheduler slice (one pass over every still-training client), only
GOSSIP_PARTICIPATION of them actually gossip that slice - the nearest
mapping of the reference's per-round participation onto a system with no
global round. Those gossip operations are all *planned* before any of them
is applied, so they read one frozen set of weights rather than each other's
half-finished updates (see gossip.py).

    once every client is done: cross-client transfer matrix (Layer E)

Layer F (the 5x5 spatial interference matrix) is dropped for now - see
config.py. Nothing here probes around the gossip step.

Evaluation never feeds back into training - no metric computed here selects a
weight, a threshold or a partner (EVAL_PIPELINE_DETECTION.md 0.1).
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from config import (
    BATCH_SIZE, BUFFER_CAPACITY, CLIENTS, EPOCHS_PER_TASK, FREEZE_BACKBONE_STAGES,
    GATE_BATCH_SIZE, GOSSIP_EVERY_N_STEPS, GOSSIP_FANOUT, GOSSIP_LAMBDA,
    GOSSIP_PARTICIPATION, LR_DECAY_FACTOR, LR_DECAY_MILESTONES, MODES,
    NUM_CLASSES, NUM_WORKERS, REPLAY_RATIO, SEED, TASKS, USE_AMP,
)
from data import DataRegistry, detection_collate
from evaluate_detection import (
    client_summary, evaluate_cross_client, evaluate_detection,
    evaluate_tasks_detection, print_cross_client, system_summary,
)
from gossip import apply_gossip, plan_gossip, sample_neighbors
from logger import get_logger
from model import build_model, build_optimizer, build_warmup
from replay import ReplayBuffer


# --- small helpers ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{"boxes": t["boxes"].to(device, non_blocking=True),
                "labels": t["labels"].to(device, non_blocking=True)}
               for t in targets]
    return images, targets


class StepFeeder:
    """
    Serves batches one at a time from a loader, re-opening it each epoch and
    stopping once the epoch budget is spent.

    Training is driven a few steps at a time (the client stops to gossip on
    its own step count), so the loop needs to pull N batches without caring
    where an epoch boundary falls.
    """

    def __init__(self, loader, epochs):
        self.loader = loader
        self.epochs_left = epochs
        self.it = iter(loader)
        self.exhausted = len(loader) == 0

    def next_batch(self):
        if self.exhausted:
            return None
        try:
            return next(self.it)
        except StopIteration:
            self.epochs_left -= 1
            if self.epochs_left <= 0:
                self.exhausted = True
                return None
            self.it = iter(self.loader)
            return next(self.it, None)


# --- client ----------------------------------------------------------------

class Client:
    def __init__(self, name, registry, device, mode, seed):
        self.name = name
        self.registry = registry
        self.device = device
        self.use_replay = MODES[mode]["replay"]

        self.model = build_model().to(device)
        self.optimizer = build_optimizer(self.model)
        # Per-client warm-up on this client's own step count - see model.py.
        self.warmup = build_warmup(self.optimizer)
        self.scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

        self.buffer = ReplayBuffer(registry.boxes_index, BUFFER_CAPACITY, seed,
                                   weights=registry.rarity_weights())
        self.rng = random.Random(seed)

        self.task_id = 0
        self.finished = False
        self.acc_history = {t: [] for t in range(len(TASKS))}
        # summarize() computes per-class AP and the size-stratified AP_S/AP_M/
        # AP_L on every task-boundary eval. Keeping only the macro scalar threw
        # all of it away, so questions like "how is `traffic sign` doing" or
        # "did anything change for small objects" could not be answered without
        # a full retrain. Cheap to keep: ~10 classes x 12 fields per eval.
        self.class_history = {t: [] for t in range(len(TASKS))}
        self.local_steps = 0        # this client's own clock, nobody else's
        self.gossip_count = 0

        self.feeder = None
        self._start_task(0)

    # --- setup ---

    def _start_task(self, task_id):
        self.task_id = task_id
        train_names = self.registry.filenames(self.name, task_id, "train")

        # 1:1 current-to-replay means half a batch of each, so the effective
        # batch stays BATCH_SIZE.
        half = max(1, BATCH_SIZE // 2)
        use_replay = self.use_replay and len(self.buffer) > 0
        cur_bs = half if use_replay else BATCH_SIZE

        loader = self.registry.loader(train_names, cur_bs, True, NUM_WORKERS)
        self.feeder = StepFeeder(loader, EPOCHS_PER_TASK)
        self.replay_half = half if use_replay else 0
        self.current_names = train_names

    # --- training ---

    def _replay_batch(self):
        """Half a batch drawn from memory, decoded on demand from filenames."""
        if not self.replay_half:
            return None
        names = self.buffer.sample(self.replay_half)
        return self.registry.batch(names)

    def train_steps(self, steps=GOSSIP_EVERY_N_STEPS):
        """
        Up to `steps` local optimiser steps - the stretch between two of this
        client's own gossip operations. Returns how many it actually took
        (fewer if the task ran out).
        """
        self.model.train()
        taken = 0
        for _ in range(steps):
            batch = self.feeder.next_batch()
            if batch is None:
                break

            images, targets = batch
            replay = self._replay_batch()
            if replay is not None:
                images = list(images) + list(replay[0])
                targets = list(targets) + list(replay[1])

            images, targets = to_device(images, targets, self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                losses = self.model(images, targets)
                loss = sum(losses.values())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.warmup is not None:
                self.warmup.step()
            taken += 1
            self.local_steps += 1

        return taken

    def task_exhausted(self):
        return self.feeder.exhausted

    # --- gossip support ---

    def gate_batch(self):
        """
        A batch to gate on: the client's own memory if it has one, else a
        random sample of its current task. It has to be the client's own data
        - the gate asks "does this neighbour help me *here*".

        The current-task path is not a rare fallback. The buffer only fills at
        a task boundary, so it is empty for the whole of task 0 and every gate
        decision in that third of the run comes through here. Taking the first
        GATE_BATCH_SIZE filenames scored every neighbour, every time, on one
        fixed set of images that was representative of the client's cell only
        by luck. Sampling makes it an estimate instead of a constant.
        """
        names = self.buffer.sample(GATE_BATCH_SIZE) if len(self.buffer) else []
        if not names:
            k = min(GATE_BATCH_SIZE, len(self.current_names))
            names = self.rng.sample(self.current_names, k)
        return self.registry.batch(names)

    # --- boundaries ---

    def finish_task(self):
        """Evaluate every task seen so far, then advance (spec 9C)."""
        seen = {t: self.registry.loader(
                    self.registry.filenames(self.name, t, "test"),
                    BATCH_SIZE, False, NUM_WORKERS)
                for t in range(self.task_id + 1)}

        summaries = evaluate_tasks_detection(self.model, seen, self.device)
        for tid, s in summaries.items():
            self.acc_history[tid].append(s["mAP@[.5:.95]"])
            self.class_history[tid].append(s["per_class"])

        if self.use_replay:
            self.buffer.add_many(self.current_names)

        if self.task_id + 1 < len(TASKS):
            self._start_task(self.task_id + 1)
        else:
            self.finished = True
        return summaries


# --- the run ---------------------------------------------------------------

def _stem(tag, mode, seed):
    """
    Output basename. `tag` names the *configuration*, not the mode or seed -
    results from different gossip settings otherwise overwrite each other,
    since mode+seed alone cannot tell them apart.
    """
    return f"{tag}_{mode}_seed{seed}" if tag else f"{mode}_seed{seed}"


def run(mode="gossip", seed=SEED, device=None, out_dir="results/runs",
        max_rounds=None, tag=""):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger(mode, seed, tag=tag)
    do_gossip = MODES[mode]["gossip"]
    registry = DataRegistry()
    clients = {name: Client(name, registry, device, mode, seed)
               for name in CLIENTS}

    logger.info(f"started training | mode={mode} seed={seed} "
               f"clients={list(clients)} gossip_every={GOSSIP_EVERY_N_STEPS} steps")

    rng = random.Random(seed)
    started = time.time()
    slices = 0
    gate_log = []

    # Round-robin over whoever is still training. This is just how one machine
    # interleaves five independent clients - not a barrier. A client that
    # finishes drops out of the rotation while the others carry on mid-task.
    while not all(c.finished for c in clients.values()):
        slices += 1
        if max_rounds and slices > max_rounds:
            break

        for client in [c for c in clients.values() if not c.finished]:
            # Train up to this client's own gossip interval. Nobody waits for
            # anybody - gossiping itself happens after everyone in this slice
            # has trained, so a fixed subset of clients reads this slice.
            client.train_steps(GOSSIP_EVERY_N_STEPS)
            logger.info(f"  [{client.name}] trained -> {client.local_steps} local steps")

            if client.task_exhausted():
                logger.info(f"  [{client.name}] task {client.task_id} finished, evaluating")
                client.finish_task()

        if do_gossip:
            active = [c for c in clients.values() if not c.finished]
            participants = rng.sample(active, min(GOSSIP_PARTICIPATION, len(active)))

            # Plan every participant before applying any of them, so all of
            # this slice's gossip reads the same frozen weights. Applying as
            # we went would let a client that already pulled hand its blended
            # weights on, echoing its own parameters back to it - clear
            # pulling rainy just after rainy pulled clear would leave clear at
            # 0.75 of itself instead of the intended 0.5.
            plans = []
            for client in participants:
                # Finished clients stay eligible as sources, so this is empty
                # only in a one-client system - see gossip.py.
                neighbour_names = sample_neighbors(
                    client.name, list(clients), GOSSIP_FANOUT, rng)
                if not neighbour_names:
                    logger.info(f"  [{client.name}] no neighbours available, skipping")
                    continue
                planned = _plan_gossip(
                    client, [clients[n] for n in neighbour_names], device)
                if planned is None:
                    logger.info(f"  [{client.name}] no memory to gate on, skipping")
                    continue
                rec, aggregate = planned
                plans.append((client, neighbour_names, rec, aggregate))

            for client, neighbour_names, rec, aggregate in plans:
                apply_gossip(client.model, client.optimizer, aggregate, rec["alpha"])
                gate_log.append(rec)
                logger.info(
                    f"  [{client.name}] <- {neighbour_names} "
                    f"{'gossip done' if aggregate is not None else 'all neighbours gated out'}"
                    f" | alpha={rec['alpha']:.4f}")

        elapsed = (time.time() - started) / 60
        done = sum(c.finished for c in clients.values())
        steps = {n: c.local_steps for n, c in clients.items()}
        logger.info(f"slice {slices:5d} | finished {done}/{len(clients)} "
                   f"| {elapsed:.1f} min | local steps {steps}")

    _save_heads(clients, out_dir, mode, seed, logger, tag)
    results = _collect(clients, registry, device, mode, seed, slices, gate_log)
    _save(results, out_dir, mode, seed, logger, tag)
    return results


def _plan_gossip(client, neighbours, device):
    """
    The read half of one pull-only gossip operation, adapting Client objects
    to gossip.py's dicts. Mutates nothing, so the caller can plan every
    participant in a slice before any of them writes. No probing - Layer F
    is dropped (config.py).

    Returns (record, aggregate), or None if the client has no memory to gate
    on. `aggregate` is None when every neighbour was gated out.

    The record carries the per-neighbour gate losses and weights. With Layer
    F gone it is the *only* evidence of what the gate did: weights pinned at
    1.0 on every read mean the gate is passing everything at full strength,
    which is indistinguishable from plain averaging in the final metrics.
    Discarding it makes a gossip run unable to explain its own result.
    """
    batch_own = client.gate_batch()
    if batch_own is None:
        return None

    client.gossip_count += 1

    return plan_gossip(
        {"name": client.name, "model": client.model, "optimizer": client.optimizer,
         "gate_batch": batch_own},
        [{"name": n.name, "model": n.model} for n in neighbours],
        device, lam=GOSSIP_LAMBDA,
    )


def _save_heads(clients, out_dir, mode, seed, logger=None, tag=""):
    """
    Persist each client's trained head.

    Nothing in this pipeline used to write weights anywhere, so a finished run
    left no way to compute a metric it had not already thought to record - a
    9-hour retrain to answer a one-line question. Only the trainable head is
    saved: everything below roi_heads is frozen pretrained weights that
    build_model() reproduces exactly, so the head alone restores the model.
    That is ~56k floats per client rather than ~41M.
    """
    path = os.path.join(out_dir, f"{_stem(tag, mode, seed)}_heads.pt")
    torch.save({
        "mode": mode,
        "seed": seed,
        "num_classes": NUM_CLASSES,
        "heads": {name: {k: v.detach().cpu()
                         for k, v in c.model.named_parameters() if v.requires_grad}
                  for name, c in clients.items()},
    }, path)
    (logger.info if logger else print)(f"saved heads -> {path}")


def _collect(clients, registry, device, mode, seed, slices, gate_log=()):
    """Everything the report needs, computed once all clients are done."""
    per_client = {name: client_summary(c.acc_history) for name, c in clients.items()}

    # Cross-client transfer only answers "what did gossip contribute" (spec 8:
    # compare against local-only, the difference is what transferred) - it says
    # nothing about replay, which is what `sequential` isolates instead. Skip
    # the single most expensive evaluation (a full 5x5 inference matrix) on
    # runs that were never going to use the number.
    if mode in ("gossip", "local_only"):
        pooled = {name: registry.pooled_test_loader(name, BATCH_SIZE, NUM_WORKERS)
                  for name in clients}
        models = {name: c.model for name, c in clients.items()}
        cross = evaluate_cross_client(models, pooled, device)
    else:
        cross = None

    return {
        "mode": mode,
        "seed": seed,
        "scheduler_slices": slices,
        "local_steps": {n: c.local_steps for n, c in clients.items()},
        "gossip_ops": {n: c.gossip_count for n, c in clients.items()},
        "per_client": per_client,
        "system": system_summary(per_client),
        "cross_client": cross,
        "acc_history": {n: c.acc_history for n, c in clients.items()},
        "class_history": {n: c.class_history for n, c in clients.items()},
        "buffer_coverage": {n: dict(c.buffer.class_coverage()) for n, c in clients.items()},
        "gate": _gate_summary(gate_log),
        "gate_log": list(gate_log),
    }


def _gate_summary(gate_log):
    """
    What the gate actually did over the run.

    The gate controls two separate things, and both have to be visible or a
    flat result cannot be explained:

      weight       raw exp(-s * excess): the *absolute* judgement, 1.0
                   meaning "as good as me on my own memory". frac_at_full
                   near 1.0 is the thing to worry about - the gate waving
                   everyone through.
      norm_weight  weight normalised across the round's neighbours - the
                   *composition*, which is all the reference's gate decides.
                   At fanout 2 an indifferent round splits 0.5/0.5, so
                   norm_spread_mean near 0 means the gate could not tell its
                   neighbours apart.
      alpha        the *dose*, lam * the norm-weighted mean of the raw
                   weights. This is the one place we diverge from the
                   reference, which always mixes a flat lam. frac_at_lam near
                   1.0 means the dose gate is idle and the run has collapsed
                   back to the reference's behaviour.
      excess       (loss_neighbour - loss_own) / loss_own, kept directly
                   comparable to results/metrics_diagnosis.md.
    """
    obs = [(nb, r["loss_own"]) for r in gate_log for nb in r["neighbours"]]
    mixes = sum(1 for r in gate_log if r.get("alpha", 0.0) > 0.0)
    if not obs:
        return {"n_exchanges": len(gate_log), "n_mixes": mixes,
                "n_neighbour_reads": 0}

    weights = [nb["weight"] for nb, _ in obs]
    excess = [max(0.0, (nb["loss"] - own) / own) if own > 0 else 0.0
              for nb, own in obs]

    # Only rounds that read more than one live neighbour can show a spread;
    # a single-neighbour round normalises to 1.0 and would dilute the mean
    # towards "no discrimination" for a reason that has nothing to do with
    # the gate.
    spreads = []
    for r in gate_log:
        nws = [nb["norm_weight"] for nb in r["neighbours"] if "norm_weight" in nb]
        if len(nws) > 1:
            spreads.append(max(nws) - min(nws))

    alphas = [r["alpha"] for r in gate_log if r.get("alpha", 0.0) > 0.0]
    at_lam = sum(1 for a in alphas if abs(a - GOSSIP_LAMBDA) < 1e-6)

    return {
        "n_exchanges": len(gate_log),
        "n_mixes": mixes,
        "n_skipped": len(gate_log) - mixes,
        "n_neighbour_reads": len(obs),
        "lambda": GOSSIP_LAMBDA,
        "alpha_mean": float(np.mean(alphas)) if alphas else 0.0,
        "alpha_std": float(np.std(alphas)) if alphas else 0.0,
        "alpha_min": float(np.min(alphas)) if alphas else 0.0,
        "alpha_max": float(np.max(alphas)) if alphas else 0.0,
        "frac_at_lam": (at_lam / len(alphas)) if alphas else 0.0,
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "frac_at_full": sum(1 for w in weights if w > 1.0 - 1e-6) / len(weights),
        "n_multi_neighbour_rounds": len(spreads),
        "norm_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
        "excess_median": float(np.median(excess)),
        "excess_p90": float(np.percentile(excess, 90)),
    }


def _save(results, out_dir, mode, seed, logger=None, tag=""):
    log = logger.info if logger else print
    path = os.path.join(out_dir, f"{_stem(tag, mode, seed)}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    log(f"=== {mode} (seed {seed}) ===")
    for name, s in results["per_client"].items():
        log(f"{name:>14}  final mAP {s['final_mAP']:.4f}  "
            f"forgetting {s['forgetting']:+.4f}  BWT {s['bwt']:+.4f}")
    sysm = results["system"]
    log(f"system mAP  mean {sysm['mAP']['mean']:.4f}  "
        f"min {sysm['mAP']['min']:.4f}  max {sysm['mAP']['max']:.4f}  "
        f"std {sysm['mAP']['std']:.4f}")
    gate = results.get("gate", {})
    if gate.get("n_neighbour_reads"):
        log(f"gate: {gate['n_mixes']} mixes, {gate['n_skipped']} skipped, over "
            f"{gate['n_exchanges']} exchanges "
            f"({gate['n_neighbour_reads']} neighbour reads)")
        log(f"gate: alpha (dose) mean {gate['alpha_mean']:.4f} "
            f"[{gate['alpha_min']:.4f}, {gate['alpha_max']:.4f}] "
            f"std {gate['alpha_std']:.4f} | at lambda {gate['lambda']}: "
            f"{gate['frac_at_lam']:.1%}")
        log(f"gate: weight mean {gate['weight_mean']:.4f} "
            f"[{gate['weight_min']:.4f}, {gate['weight_max']:.4f}] "
            f"std {gate['weight_std']:.4f} | at full: {gate['frac_at_full']:.1%}")
        log(f"gate: norm spread {gate['norm_spread_mean']:.4f} over "
            f"{gate['n_multi_neighbour_rounds']} multi-neighbour rounds | "
            f"excess median {gate['excess_median']:.4f} "
            f"p90 {gate['excess_p90']:.4f}")
    if results["cross_client"] is not None:
        log(f"global cross-client mAP: {results['cross_client']['global_cross_mAP']:.4f}  "
            f"own-vs-other gap: {results['cross_client']['global_own_vs_other_gap']:+.4f}")
        print_cross_client(results["cross_client"])
    log(f"saved -> {path}")


def run_joint(seed=SEED, device=None, out_dir="results/runs", max_steps=None, tag="",
              epochs=None, lr_decay_milestones=None, lr_decay_factor=LR_DECAY_FACTOR,
              eval_every_epoch=False, backbone_frozen_stages=None, batch_size=None):
    """
    The centralised upper bound (spec 8): one model, everybody's data at once,
    no continual learning and no federation. Not a variant of the loop above -
    there are no clients, no tasks and no gossip here, which is the point. It
    bounds what the decentralised system is chasing.

    batch_size overrides BATCH_SIZE for this run only, same reasoning as
    epochs and backbone_frozen_stages below - unfreezing more of the backbone
    (FREEZE_BACKBONE_STAGES=2) needs more activation memory per step than a
    16GB P100 has at BATCH_SIZE=32, confirmed by an actual CUDA OOM on a fully
    clean GPU (job 37899). Lower batch size here rather than in config.py so
    every existing baseline stays at its original, comparable batch size -
    this run is the one deliberately deviating, not the default.

    backbone_frozen_stages is passed straight to build_model() rather than
    read from config.py here, so a SLURM batch job can pin it via an
    explicit CLI argument. sbatch returns as soon as a job is queued, not
    when it starts - editing config.py for a second job before the first one
    has actually started running would silently change the first job's
    setting too.

    epochs overrides EPOCHS_PER_TASK for this run only. The federated modes
    have a communication-round budget to respect and keep reading
    EPOCHS_PER_TASK from config; joint has none, so this is a separate knob.
    None (the default) is EPOCHS_PER_TASK, unchanged from every joint run
    before this parameter existed.

    lr_decay_milestones, if given, are fractions of `epochs` (not raw epoch
    numbers) at which every optimiser param group's lr is multiplied by
    lr_decay_factor - mirrors the reference recipe's step=[8, 11] at 12
    epochs (2/3, 11/12 of the schedule), so a longer or shorter `epochs` scales
    the decay points with it. None (the default) keeps training flat after
    warmup, same as before this parameter existed.

    eval_every_epoch, if True, evaluates the same per-client pooled test sets
    at the end of every epoch, not just the final one, and records the curve
    in results["epoch_curve"] - the only way to see whether more epochs is
    still improving or has started overfitting, since this mode otherwise
    only ever evaluates once, at the very end.
    """
    epochs = epochs if epochs is not None else EPOCHS_PER_TASK
    backbone_frozen_stages = (backbone_frozen_stages if backbone_frozen_stages is not None
                              else FREEZE_BACKBONE_STAGES)
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    decay_epochs = ({round(f * epochs) for f in lr_decay_milestones}
                    if lr_decay_milestones else set())

    logger = get_logger("joint", seed, tag=tag)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    registry = DataRegistry()
    names = []
    for client in CLIENTS:
        for tid in range(len(TASKS)):
            names.extend(registry.filenames(client, tid, "train"))
    logger.info(f"joint baseline: {len(names)} training images, no CL, no gossip, "
                f"{epochs} epochs, backbone_frozen_stages={backbone_frozen_stages}, "
                f"batch_size={batch_size}" +
                (f", lr decay x{lr_decay_factor} at epochs {sorted(decay_epochs)}"
                 if decay_epochs else ""))

    model = build_model(backbone_frozen_stages=backbone_frozen_stages).to(device)
    optimizer = build_optimizer(model)
    warmup = build_warmup(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    loader = registry.loader(names, batch_size, True, NUM_WORKERS)

    epoch_curve = []

    model.train()
    step = 0
    for epoch in range(epochs):
        for images, targets in loader:
            images, targets = to_device(images, targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = sum(model(images, targets).values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if warmup is not None:
                warmup.step()

            step += 1
            if step % 100 == 0:
                logger.info(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

        if (epoch + 1) in decay_epochs:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay_factor
            logger.info(f"lr decayed x{lr_decay_factor} at epoch {epoch + 1} -> "
                        f"{[g['lr'] for g in optimizer.param_groups]}")

        if eval_every_epoch:
            epoch_aps, epoch_aps_full = [], []
            for client in CLIENTS:
                pooled = registry.pooled_test_loader(client, batch_size, NUM_WORKERS)
                summary = evaluate_detection(model, pooled, device)
                epoch_aps.append(summary["mAP@[.5:.95]"])
                epoch_aps_full.append(summary["mAP_full"])
            point = {"epoch": epoch + 1, "step": step,
                     "mean_mAP": float(np.nanmean(epoch_aps)),
                     "mean_mAP_full": float(np.nanmean(epoch_aps_full))}
            epoch_curve.append(point)
            logger.info(f"epoch {epoch + 1} curve: mean_mAP {point['mean_mAP']:.4f}  "
                        f"mean_mAP_full {point['mean_mAP_full']:.4f}")
            model.train()

    # Same per-client pooled test sets the gossip run reports on, so the two
    # are directly comparable.
    #
    # Keep the whole summary, not just the scalar. The a16rpn_joint run kept
    # `mAP@[.5:.95]` alone, and since the question v3 exists to answer is
    # "did small-object recall move", that discarded exactly the number the
    # run was commissioned to produce - mAR_S, mAR_L and the per-class AP_S /
    # AR_S / N_S. run() has recorded per_class since v2; this path had not.
    per_client, per_client_full = {}, {}
    for client in CLIENTS:
        pooled = registry.pooled_test_loader(client, batch_size, NUM_WORKERS)
        summary = evaluate_detection(model, pooled, device)
        per_client[client] = summary["mAP@[.5:.95]"]
        per_client_full[client] = summary
        logger.info(f"{client:>14}  mAP@[.5:.95] {summary['mAP@[.5:.95]']:.4f}  "
                    f"mAR@100 {summary['mAR@100']:.4f}  "
                    f"mAR_S {summary['mAR_S']:.4f}  mAR_L {summary['mAR_L']:.4f}")

    results = {"mode": "joint", "seed": seed, "steps": step, "epochs": epochs,
               "backbone_frozen_stages": backbone_frozen_stages,
               "batch_size": batch_size,
               "epoch_curve": epoch_curve,
               "per_client_pooled_mAP": per_client,
               "per_client_summary": per_client_full,
               "mean_mAP": float(np.mean(list(per_client.values()))),
               "mean_mAP_full": float(np.nanmean(
                   [s["mAP_full"] for s in per_client_full.values()])),
               "mean_mAR_S": float(np.nanmean(
                   [s["mAR_S"] for s in per_client_full.values()])),
               "mean_mAR_100": float(np.nanmean(
                   [s["mAR@100"] for s in per_client_full.values()]))}

    # One model here, not five, but the same argument as _save_heads: without
    # weights, any metric this run did not think to record costs a full
    # retrain to recover.
    head_path = os.path.join(out_dir, f"{_stem(tag, 'joint', seed)}_heads.pt")
    torch.save({"mode": "joint", "seed": seed, "num_classes": NUM_CLASSES,
                "heads": {"joint": {k: v.detach().cpu()
                                    for k, v in model.named_parameters()
                                    if v.requires_grad}}}, head_path)
    logger.info(f"saved heads -> {head_path}")

    path = os.path.join(out_dir, f"{_stem(tag, 'joint', seed)}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"mean pooled mAP {results['mean_mAP']:.4f}  "
                f"mean pooled mAP_full {results['mean_mAP_full']:.4f}  "
                f"mean mAR_S {results['mean_mAR_S']:.4f}")
    logger.info(f"saved -> {path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gossip", choices=list(MODES))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default="results/runs")
    ap.add_argument("--max-rounds", type=int, default=None,
                    dest="max_rounds",
                    help="cap scheduler slices - for smoke tests, not real runs")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="joint mode only, same purpose")
    ap.add_argument("--tag", default="",
                    help="prefix for run/heads/log filenames - names the "
                         "configuration, since mode+seed cannot distinguish "
                         "two different gossip settings")
    ap.add_argument("--epochs", type=int, default=None,
                    help="joint mode only - overrides EPOCHS_PER_TASK for "
                         "this run without touching the federated modes' "
                         "budget. Default: EPOCHS_PER_TASK, unchanged.")
    ap.add_argument("--lr-decay", action="store_true", dest="lr_decay",
                    help="joint mode only - step-decay every optimiser "
                         "group's lr by LR_DECAY_FACTOR at LR_DECAY_MILESTONES "
                         "(fractions of --epochs). Off by default, matching "
                         "every joint run before this flag existed.")
    ap.add_argument("--eval-every-epoch", action="store_true",
                    dest="eval_every_epoch",
                    help="joint mode only - evaluate all clients at the end "
                         "of every epoch, not just the final one, and record "
                         "the curve in results['epoch_curve'].")
    ap.add_argument("--freeze-backbone-stages", type=int, default=None,
                    dest="freeze_backbone_stages", choices=[1, 2],
                    help="joint mode only - freeze only the ResNet stem + "
                         "layer1..layerN (MMDetection's frozen_stages "
                         "convention), overriding FREEZE_BACKBONE_STAGES for "
                         "this run via an explicit argument rather than "
                         "config.py, so back-to-back sbatch submissions can't "
                         "race on which config a queued job reads once it "
                         "starts. Default: config.py's FREEZE_BACKBONE_STAGES "
                         "(None), unchanged.")
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                    help="joint mode only - overrides BATCH_SIZE for this run "
                         "only, e.g. to fit a partially-unfrozen backbone's "
                         "larger activation memory on a 16GB P100. Default: "
                         "config.py's BATCH_SIZE, unchanged.")
    args = ap.parse_args()

    if args.mode == "joint":
        run_joint(seed=args.seed, out_dir=args.out_dir, max_steps=args.max_steps,
                  tag=args.tag, epochs=args.epochs,
                  lr_decay_milestones=LR_DECAY_MILESTONES if args.lr_decay else None,
                  eval_every_epoch=args.eval_every_epoch,
                  backbone_frozen_stages=args.freeze_backbone_stages,
                  batch_size=args.batch_size)
    else:
        run(mode=args.mode, seed=args.seed, out_dir=args.out_dir,
            max_rounds=args.max_rounds, tag=args.tag)
