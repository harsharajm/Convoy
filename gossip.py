"""
Memory-gated pull-only gossip.

No server and no FedAvg: a client reads a handful of neighbours, measures how
well each one's weights do on its *own* replay memory, and mixes a weighted
average of them into itself. Neighbours are read, never written - this
matches the reference, whose only write is local_models[client_id]
(gossip_pfedcil.py:677). Standard gossip learning (Hegedus et al., "Gossip
Learning as a Decentralized Alternative to Federated Learning") runs the same
exchange in the opposite direction: a node *pushes* its model out and the
receiver merges it. Pull or push, a single exchange moves exactly one model,
and only many independent exchanges spread updates through the network.

One deliberate divergence from the reference: it sets the mixing dose to a
flat lambda and lets the gate decide only *which* neighbour to favour, since
normalising the gate weights discards their absolute size. Here the gate sets
the dose too - see plan_gossip. Everything else (the per-neighbour gate, the
normalisation, the all-poisonous skip) is the reference's.

Only the head parameters travel. Everything below roi_heads is frozen and
identical across clients, so aggregating it would be a no-op that also
multiplies the message size by ~40x. torchvision's detection backbone uses
FrozenBatchNorm2d, so there are no running statistics drifting apart either.

Reading is split from writing (plan_gossip / apply_gossip) so that every
client gossiping in the same scheduler slice sees the same frozen weights.
Applying as we go would let a client that already pulled hand its blended
weights to the next one, and its own parameters would echo back to it: if
rainy pulls clear and clear then pulls rainy, clear ends at 0.75 of itself
rather than the intended 0.5. The paper this follows has no such problem
because a model is *sent* - a message in flight is a copy by construction,
and the sender keeps training behind it. Shared memory has to do on purpose
what the network did for free.

Momentum handling follows the rule settled in PIPELINE_CONFLICTS.md 4: a
client resets its momentum buffers exactly when its own weights changed.
Pull-only means only the reading client's weights ever change from a gossip
operation, so only it ever resets - a client read from as a neighbour is
untouched and keeps its momentum. A finished client still hands its weights
out as a source (see sample_neighbors) but never reads any in, so it never
resets.
"""

import math

import torch

from config import GATE_SENSITIVITY, GOSSIP_LAMBDA, USE_AMP
from model import reset_momentum


@torch.no_grad()
def _head_state(model):
    """
    A detached *copy* of the parameters that travel: the trainable head.

    The copy is the point. `p.detach()` on its own hands back a view sharing
    storage with the live parameter, so a state captured "before" a mix
    quietly tracks that mix - which is how the old pairwise code's supposedly
    simultaneous exchange was order-dependent all along. The head is ~56k
    floats, so cloning it costs nothing worth counting.
    """
    return {name: p.detach().clone()
            for name, p in model.named_parameters() if p.requires_grad}


# The head's losses only. loss_objectness and loss_rpn_box_reg come from the
# RPN, which is frozen and byte-identical across clients, so on a given batch
# they contribute the *same constant* to every model scored. That constant
# inflates both sides of excess = (loss_n - loss_own) / loss_own while
# carrying no information about the neighbour, which is one of the two causes
# the seed-7 diagnosis gave for a gate that passed everything.
#
# The reference gates on the trainable model's task loss alone
# (gossip_pfedcil.py:643, a plain cross_entropy); for a detector that is the
# box head. ONLY VALID WHILE config.FREEZE_RPN IS TRUE - unfreeze the RPN and
# these terms start carrying signal again and belong back in the sum.
GATE_LOSS_KEYS = ("loss_classifier", "loss_box_reg")


def gate_loss(model, batch, device):
    """
    The box head's training loss on a batch - the gate signal.

    Must run in train() mode: a detector returns a loss dict in train() and
    predictions in eval(), never both. No gradients are taken, so this does
    not touch the model's weights.
    """
    images, targets = batch
    images = [img.to(device) for img in images]
    targets = [{"boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device)} for t in targets]

    was_training = model.training
    model.train()
    with torch.no_grad(), torch.amp.autocast(
            device.type, enabled=USE_AMP and device.type == "cuda"):
        losses = model(images, targets)
        gated = [v for k, v in losses.items() if k in GATE_LOSS_KEYS]
        total = sum(gated if gated else losses.values()).item()
    if not was_training:
        model.eval()
    return total


def gate_weight(loss_own, loss_neighbour, sensitivity=GATE_SENSITIVITY):
    """
    How much of the neighbour to accept, in [0, 1].

    A neighbour that scores at least as well as me on my own memory passes at
    full strength. One that scores worse decays exponentially in how much
    worse, scaled by `sensitivity`. Normalised by the client's own loss so the
    gate means the same thing early in training (large losses) and late
    (small ones).
    """
    if loss_own <= 0:
        return 1.0 if loss_neighbour <= loss_own else 0.0
    excess = max(0.0, (loss_neighbour - loss_own) / loss_own)
    return math.exp(-sensitivity * excess)


@torch.no_grad()
def _weighted_average(states, weights):
    """Per-parameter weighted average of several head state dicts."""
    keys = states[0].keys()
    return {k: sum(s[k] * w for s, w in zip(states, weights)) for k in keys}


@torch.no_grad()
def mix_into(model, source_state, alpha):
    """model <- (1 - alpha) * model + alpha * source, head parameters only."""
    if alpha <= 0:
        return
    for name, p in model.named_parameters():
        if p.requires_grad and name in source_state:
            p.mul_(1.0 - alpha).add_(source_state[name].to(p.device), alpha=alpha)


def plan_gossip(client, neighbours, device, lam=GOSSIP_LAMBDA,
                sensitivity=GATE_SENSITIVITY):
    """
    The read half of a pull-only gossip operation. Mutates nothing.

    `client` is a dict: {"name", "model", "optimizer", "gate_batch"}.
    `neighbours` is a list of dicts: {"name", "model"} - no optimizer or gate
    needed, since they are only ever read from.

    Each neighbour is gated on `client`'s own memory (the question is always
    "does this neighbour help me on my data"), then the per-neighbour weights
    are normalised so they sum to 1 - this is what makes fanout > 1 actually
    discriminate between neighbours, rather than each being judged only
    against an absolute threshold. If every neighbour is poisonous (total
    weight ~0) the round is skipped rather than mixing in noise.

    Returns (record, aggregate). `aggregate` is the single head state to mix
    in, or None if the round is to be skipped. Because this touches no
    weights, every participant in a slice can be planned before any of them
    applies, which is what freezes the slice - see the module docstring.

    The record is what the gate did, and with Layer F dropped it is the only
    evidence there is: a run whose per-neighbour weights all sit at 1.0 is
    plain averaging wearing a gate's name.
    """
    own_batch = client["gate_batch"]
    loss_own = gate_loss(client["model"], own_batch, device)

    record = {"client": client["name"], "loss_own": loss_own,
              "neighbours": [], "alpha": 0.0}

    states, weights = [], []
    for n in neighbours:
        loss_n = gate_loss(n["model"], own_batch, device)
        w = gate_weight(loss_own, loss_n, sensitivity)
        states.append(_head_state(n["model"]))
        weights.append(w)
        record["neighbours"].append(
            {"name": n["name"], "loss": loss_n, "weight": w})

    total_w = sum(weights)
    if total_w < 1e-6:
        return record, None

    norm_weights = [w / total_w for w in weights]
    for nb, nw in zip(record["neighbours"], norm_weights):
        nb["norm_weight"] = nw

    # Composition is the reference's; the dose is not. Normalising throws the
    # absolute gate weights away, so the reference always mixes a full `lam`
    # whenever any neighbour survives - the gate picks *which* neighbour and
    # never *how much*. Measured consequence: `clear`, which trains on 7.6x
    # more data than the smallest client and has no neighbour that can teach
    # it anything (cross-cell total variation ~0.05), still took 50% from
    # strictly worse models every exchange and lost 0.022 mAP doing it - on
    # its own about 60% of gossip's whole deficit against local_only.
    #
    # So scale the dose by the gate weight of the blend actually being taken:
    # the norm-weighted mean of the raw weights. With one neighbour this is
    # exactly lam * gate_weight, which is what this code did before the
    # multi-neighbour port; with equally good neighbours it is unchanged.
    gate = sum(w * nw for w, nw in zip(weights, norm_weights))
    record["gate"] = gate
    record["alpha"] = lam * gate
    return record, _weighted_average(states, norm_weights)


def apply_gossip(model, optimizer, aggregate, alpha):
    """
    The write half: mix a planned aggregate into the client that planned it.

    `alpha` is the gated dose from plan_gossip's record, not the raw lambda -
    lambda is the ceiling, alpha is what the gate let through.

    Takes the model and optimizer directly rather than the client dict
    plan_gossip uses - it needs neither the name nor the gate batch, and the
    caller here is train.py, which holds Client objects rather than dicts.
    Only this client's weights move, so its momentum is the only momentum
    that goes stale. A skipped round (aggregate None) is a no-op.
    """
    if aggregate is None:
        return
    mix_into(model, aggregate, alpha)
    reset_momentum(optimizer)


def memory_gated_gossip(client, neighbours, device, lam=GOSSIP_LAMBDA,
                        sensitivity=GATE_SENSITIVITY):
    """
    One complete pull-only gossip operation, read then write.

    Correct on its own - a client is never its own neighbour, so nothing it
    writes can disturb what it just read. Use plan_gossip/apply_gossip
    directly when several clients gossip in the same slice and must all read
    the same frozen weights.
    """
    record, aggregate = plan_gossip(client, neighbours, device, lam, sensitivity)
    apply_gossip(client["model"], client["optimizer"], aggregate, record["alpha"])
    return record


def sample_neighbors(name, clients, fanout, rng):
    """
    Up to `fanout` clients for `name` to read from this gossip operation.

    Finished clients stay eligible as sources (design_choices.pdf 2.2.1).
    The reference drops them (gossip_pfedcil.py:904) and this code did too,
    until the gp3fo2 run showed what that costs here: `clear` holds 7.6x the
    images of the smallest cell, its last live neighbour finished 3h23 before
    it did, and it spent 84% of its training with no eligible neighbour at
    all. Gossip switched itself off for the one client that dominates the
    system mean, and 231 of 269 slices logged no live neighbours. The
    reference's clients are balanced and finish together, so the rule costs
    it nothing; with a 7.6x cell imbalance it silently disables the mechanism
    under test.

    The known cost of keeping them: a finished client's head is frozen at
    whatever it ended on, so it becomes a static attractor that late-running
    clients keep pulling toward. Defending against that is the gate's job -
    which, at weight_mean 0.98, it is currently not doing.
    """
    candidates = [c for c in clients if c != name]
    k = min(fanout, len(candidates))
    return rng.sample(candidates, k) if k else []
