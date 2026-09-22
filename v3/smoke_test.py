"""
Cheap end-to-end check to run on the cluster before burning a SLURM
allocation on a real run.

It targets the three failure modes that look like a broken model but are
plumbing (EVAL_PIPELINE_DETECTION.md 10): coordinate space, missing NMS, and
the background label offset. Each of those produces a near-zero mAP that
wastes a full run before anyone notices.

    python smoke_test.py
"""

import math
import random

import torch

import gossip
from config import BATCH_SIZE, CLIENTS, NUM_CLASSES, OBJECT_CLASSES, TASKS
from data import DataRegistry
from gossip import gate_loss, gate_weight, memory_gated_gossip, sample_neighbors
from model import build_model, build_optimizer, count_parameters
from replay import ReplayBuffer
from train import _gate_summary

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, condition, detail=""):
    print(f"[{OK if condition else FAIL}] {name}" + (f" - {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # --- data ---
    reg = DataRegistry()
    check("manifest loaded", len(reg.manifest) == 70426, f"{len(reg.manifest)} rows")
    check("every manifest image is indexed",
          set(reg.manifest["filename"]) <= set(reg.image_index),
          f"{len(reg.image_index)} indexed")

    names = reg.filenames("rainy", 1, "train")[:4]
    loader = reg.loader(names, 2, False, 0)
    images, targets = next(iter(loader))

    check("image is [3,H,W] float", images[0].ndim == 3 and images[0].dtype == torch.float32,
          str(tuple(images[0].shape)))
    check("pixels in [0,1] (model normalises internally)",
          0.0 <= images[0].min() and images[0].max() <= 1.0,
          f"[{images[0].min():.2f}, {images[0].max():.2f}]")
    check("boxes are [N,4] float32",
          targets[0]["boxes"].ndim == 2 and targets[0]["boxes"].shape[1] == 4)
    check("labels are int64 in 1..10 (0 is background)",
          targets[0]["labels"].dtype == torch.int64
          and int(targets[0]["labels"].min()) >= 1
          and int(targets[0]["labels"].max()) <= len(OBJECT_CLASSES))

    boxes = targets[0]["boxes"]
    check("boxes are x1<x2, y1<y2",
          bool((boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()))
    h, w = images[0].shape[1], images[0].shape[2]
    check("boxes are in original pixel coordinates, not normalised",
          float(boxes.max()) > 1.5 and float(boxes[:, 2].max()) <= w + 1,
          f"image {w}x{h}, max box coord {float(boxes.max()):.0f}")

    # --- model ---
    model = build_model().to(device)
    counts = count_parameters(model)
    # v3 trains the RPN as well as roi_heads, so a bare "under half the
    # params" bound would pass while testing nothing. Assert the split
    # explicitly: backbone frozen, rpn and roi_heads trainable.
    frozen_mods = {n.split(".")[0] for n, p in model.named_parameters()
                   if not p.requires_grad}
    train_mods = {n.split(".")[0] for n, p in model.named_parameters()
                  if p.requires_grad}
    check("backbone frozen; rpn + roi_heads trainable",
          frozen_mods == {"backbone"} and train_mods == {"rpn", "roi_heads"},
          f"frozen={sorted(frozen_mods)} trainable={sorted(train_mods)} "
          f"({counts['trainable']:,} of {counts['total']:,})")
    check("head emits 11 classes",
          model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES)

    # --- train step ---
    model.train()
    opt = build_optimizer(model)
    imgs = [i.to(device) for i in images]
    tgts = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
            for t in targets]
    losses = model(imgs, tgts)
    check("train() returns a loss dict", isinstance(losses, dict),
          ", ".join(losses) if isinstance(losses, dict) else type(losses).__name__)
    total = sum(losses.values())
    check("loss is finite", bool(torch.isfinite(total)), f"{total.item():.4f}")

    before = {n: p.detach().clone()
              for n, p in model.named_parameters() if p.requires_grad}
    total.backward()

    # v3 unfreezes the RPN. If its gradients were not actually flowing - wrong
    # param group, a detached graph, losses excluded - the run would train
    # exactly like v2 while reporting itself as v3. Check per module, before
    # the step, so a zero-grad RPN cannot hide behind roi_heads moving.
    grads = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            mod = n.split(".")[0]
            g = 0.0 if p.grad is None else float(p.grad.abs().sum())
            grads[mod] = grads.get(mod, 0.0) + g
    check("rpn receives gradients", grads.get("rpn", 0.0) > 0,
          f"rpn grad sum {grads.get('rpn', 0.0):.4g}")
    check("roi_heads receives gradients", grads.get("roi_heads", 0.0) > 0,
          f"roi_heads grad sum {grads.get('roi_heads', 0.0):.4g}")

    opt.step()
    moved = {n.split(".")[0] for n, p in model.named_parameters()
             if p.requires_grad and not torch.equal(before[n], p)}
    check("optimiser step moves both rpn and roi_heads",
          {"rpn", "roi_heads"} <= moved, f"moved: {sorted(moved)}")

    # --- inference contract ---
    model.eval()
    with torch.no_grad():
        preds = model(imgs)
    p = preds[0]
    check("eval() returns boxes/labels/scores",
          all(k in p for k in ("boxes", "labels", "scores")))
    check("scores in [0,1] and descending",
          p["scores"].numel() == 0 or
          (0 <= float(p["scores"].min()) and float(p["scores"].max()) <= 1
           and bool((p["scores"][:-1] >= p["scores"][1:]).all())))
    check("predicted labels within 1..10",
          p["labels"].numel() == 0 or
          (int(p["labels"].min()) >= 1 and int(p["labels"].max()) <= len(OBJECT_CLASSES)))
    check("predictions are in the ORIGINAL frame, not the resized one",
          p["boxes"].numel() == 0 or float(p["boxes"][:, 2].max()) <= w + 1,
          f"image width {w}, max predicted x2 "
          f"{float(p['boxes'][:, 2].max()) if p['boxes'].numel() else 0:.0f}")

    # --- replay ---
    buf = ReplayBuffer(reg.boxes_index, capacity=50, seed=0)
    buf.add_many(reg.filenames("clear", 0, "train")[:2000])
    check("buffer respects capacity", len(buf) == 50, f"{len(buf)} entries")
    check("buffer stores references, not tensors",
          all(isinstance(f, str) for f in buf.filenames()))
    cov = buf.class_coverage()
    check("rare classes survive eviction",
          sum(cov.get(c, 0) for c in ("bike", "rider", "motor", "bus", "train")) > 0,
          dict(cov))

    # --- gossip ---
    other = build_model().to(device)
    batch = next(iter(reg.loader(names, 2, False, 0)))

    l_own = gate_loss(model, batch, device)
    check("gate loss is a finite scalar", torch.isfinite(torch.tensor(l_own)).item(),
          f"{l_own:.4f}")
    check("equal-quality neighbour passes the gate at full strength",
          abs(gate_weight(1.0, 1.0) - 1.0) < 1e-9)
    check("worse neighbour is down-weighted", gate_weight(1.0, 2.0) < 0.5,
          f"{gate_weight(1.0, 2.0):.3f}")

    head_before = {n: p.detach().clone()
                   for n, p in model.named_parameters() if p.requires_grad}
    other_head_before = {n: p.detach().clone()
                         for n, p in other.named_parameters() if p.requires_grad}
    rec = memory_gated_gossip(
        {"name": "a", "model": model, "optimizer": opt, "gate_batch": batch},
        [{"name": "b", "model": other}],   # pull-only: read from, never written
        device)
    moved = any(not torch.equal(head_before[n], p)
                for n, p in model.named_parameters() if p.requires_grad)
    check("active client absorbs a neighbour", moved, f"alpha={rec['alpha']:.3f}")
    neighbour_untouched = all(torch.equal(other_head_before[n], p)
                              for n, p in other.named_parameters() if p.requires_grad)
    check("neighbour's own weights untouched (pull-only)", neighbour_untouched)

    _check_mixing(device)
    _check_neighbour_sampling()
    _check_gate_summary()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


def _tiny(v):
    """
    A model with one trainable scalar.

    The mixing code only ever reaches named_parameters(), so the freeze and
    aggregation logic can be checked as plain arithmetic instead of standing
    up a second detector. gate_loss is stubbed alongside it so the gate
    weights are exact and the expected numbers can be worked out by hand.
    """
    m = torch.nn.Module()
    m.w = torch.nn.Parameter(torch.tensor([float(v)]))
    return m


def _val(m):
    return float(m.w.detach())


def _sgd(m):
    return torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)


def _check_mixing(device):
    """The plan/apply freeze, and the snapshot it depends on."""
    real_gate_loss = gossip.gate_loss
    gossip.gate_loss = lambda model, batch, dev: 1.0   # all equal -> weight 1.0
    try:
        # A snapshot that aliases the live parameter is what made the old
        # pairwise "simultaneous" exchange order-dependent.
        m = _tiny(1.0)
        snap = gossip._trainable_state(m)
        with torch.no_grad():
            m.w.mul_(0.0).add_(9.0)
        check("gossip snapshot is a copy, not a view", float(snap["w"]) == 1.0,
              f"snapshot {float(snap['w'])}, live {_val(m)}")

        def client(name, m):
            return {"name": name, "model": m, "gate_batch": None}

        # Applying as you go: rainy pulls clear, then clear pulls the already
        # blended rainy and gets a quarter of itself back.
        clear, rainy = _tiny(1.0), _tiny(0.0)
        for own, nb in ((rainy, clear), (clear, rainy)):
            memory_gated_gossip({**client("x", own), "optimizer": _sgd(own)},
                                [{"name": "y", "model": nb}], device)
        check("apply-as-you-go echoes to 0.75 (why the freeze exists)",
              abs(_val(clear) - 0.75) < 1e-6, f"clear {_val(clear)}")

        # Planning the whole slice first removes the echo.
        clear, rainy = _tiny(1.0), _tiny(0.0)
        plans = [(rainy, gossip.plan_gossip(client("rainy", rainy),
                                            [{"name": "clear", "model": clear}], device)),
                 (clear, gossip.plan_gossip(client("clear", clear),
                                            [{"name": "rainy", "model": rainy}], device))]
        check("planning gossip mutates nothing",
              (_val(clear), _val(rainy)) == (1.0, 0.0),
              f"clear {_val(clear)}, rainy {_val(rainy)}")
        for mdl, (rec, agg) in plans:
            gossip.apply_gossip(mdl, _sgd(mdl), agg, rec["alpha"])
        check("frozen slice: both land at 0.5, no echo",
              abs(_val(clear) - 0.5) < 1e-6 and abs(_val(rainy) - 0.5) < 1e-6,
              f"clear {_val(clear)}, rainy {_val(rainy)}")

        # ... and makes the slice independent of who gossips first.
        def slice_run(order):
            ms = {"clear": _tiny(1.0), "rainy": _tiny(0.0), "snowy": _tiny(4.0)}
            nbrs = {"clear": ["rainy"], "rainy": ["clear", "snowy"], "snowy": ["clear"]}
            planned = [(ms[n], gossip.plan_gossip(
                client(n, ms[n]), [{"name": x, "model": ms[x]} for x in nbrs[n]], device))
                for n in order]
            for mdl, (r, agg) in planned:
                gossip.apply_gossip(mdl, _sgd(mdl), agg, r["alpha"])
            return {k: round(_val(v), 6) for k, v in ms.items()}

        orders = [slice_run(o) for o in (["clear", "rainy", "snowy"],
                                         ["snowy", "rainy", "clear"],
                                         ["rainy", "snowy", "clear"])]
        check("frozen slice is order-independent",
              orders[0] == orders[1] == orders[2], orders)

        # The dose gate: a worse neighbour must move you less than an equal
        # one. The reference mixes a flat lambda in both cases.
        equal = _tiny(0.0)
        rec_eq, agg_eq = gossip.plan_gossip(
            client("equal", equal), [{"name": "n", "model": _tiny(1.0)}], device)
        gossip.apply_gossip(equal, _sgd(equal), agg_eq, rec_eq["alpha"])

        worse = _tiny(0.0)
        gossip.gate_loss = lambda model, batch, dev: 1.0 if model is worse else 2.0
        rec_bad, agg_bad = gossip.plan_gossip(
            client("worse", worse), [{"name": "n", "model": _tiny(1.0)}], device)
        gossip.apply_gossip(worse, _sgd(worse), agg_bad, rec_bad["alpha"])

        check("equal neighbour takes the full lambda dose",
              abs(rec_eq["alpha"] - 0.5) < 1e-6 and abs(_val(equal) - 0.5) < 1e-6,
              f"alpha {rec_eq['alpha']:.4f}, moved to {_val(equal):.4f}")
        check("worse neighbour is dosed down to lambda * gate_weight",
              abs(rec_bad["alpha"] - 0.5 * math.exp(-2.0)) < 1e-6
              and _val(worse) < _val(equal),
              f"alpha {rec_bad['alpha']:.4f} "
              f"(expected {0.5 * math.exp(-2.0):.4f}), moved to {_val(worse):.4f}")

        # Every neighbour poisonous: skip rather than mix in noise.
        gossip.gate_loss = lambda model, batch, dev: 1.0
        keep = _tiny(1.0)
        gossip.gate_loss = lambda model, batch, dev: 1.0 if model is keep else 1e9
        rec, agg = gossip.plan_gossip(client("keep", keep),
                                      [{"name": "bad", "model": _tiny(7.0)}], device)
        check("all-poisonous round is skipped", agg is None and rec["alpha"] == 0.0)
        gossip.apply_gossip(keep, _sgd(keep), agg, rec["alpha"])
        check("applying a skipped round is a no-op", _val(keep) == 1.0)
    finally:
        gossip.gate_loss = real_gate_loss


def _check_neighbour_sampling():
    """
    Finished clients stay eligible as sources (design_choices.pdf 2.2.1).

    The regression this guards is what the gp3fo2 run hit: excluding them
    left the largest client with no neighbour for 84% of its training, so
    gossip quietly stopped happening for the client that dominates the
    system mean.
    """
    rng = random.Random(0)
    drawn = sample_neighbors("clear", CLIENTS, 2, rng)
    check("neighbours exclude self and respect fanout",
          "clear" not in drawn and len(drawn) == 2, drawn)

    draws = [sample_neighbors("clear", CLIENTS, 2, rng) for _ in range(300)]
    check("fanout never shrinks - every other client stays a source",
          {len(d) for d in draws} == {2}, sorted({len(d) for d in draws}))
    check("every other client can be drawn, finished or not",
          {x for d in draws for x in d} == set(CLIENTS) - {"clear"})
    check("single-client system yields no neighbours",
          sample_neighbors("clear", ["clear"], 2, rng) == [])


def _check_gate_summary():
    """
    The summary has to distinguish a gate that discriminates from one that
    waves everyone through - the alpha it replaced could not, being a
    constant once the neighbour weights are normalised.
    """
    def round_rec(own, neigh):
        ws = [math.exp(-2.0 * max(0.0, (l - own) / own)) for _, l in neigh]
        total = sum(ws)
        nbs = [{"name": n, "loss": l, "weight": w} for (n, l), w in zip(neigh, ws)]
        for d, w in zip(nbs, ws):
            d["norm_weight"] = w / total
        gate = sum(w * (w / total) for w in ws)       # same dose as plan_gossip
        return {"client": "a", "loss_own": own, "neighbours": nbs,
                "gate": gate, "alpha": 0.5 * gate}

    flat = _gate_summary([round_rec(1.0, [("b", 1.0), ("c", 1.0)]) for _ in range(10)])
    check("indifferent gate reads as frac_at_full 1.0, zero spread",
          flat["frac_at_full"] == 1.0 and abs(flat["norm_spread_mean"]) < 1e-12,
          f"full {flat['frac_at_full']}, spread {flat['norm_spread_mean']:.4f}")

    disc = _gate_summary([round_rec(1.0, [("b", 1.0), ("c", 2.0)]) for _ in range(10)])
    check("discriminating gate reads as spread > 0",
          disc["frac_at_full"] == 0.5 and disc["norm_spread_mean"] > 0.3,
          f"full {disc['frac_at_full']}, spread {disc['norm_spread_mean']:.4f}")
    check("excess is recorded for comparison with the diagnosis",
          abs(disc["excess_p90"] - 1.0) < 1e-9, f"p90 {disc['excess_p90']:.4f}")

    # The dose must move when neighbours are worse - this is the divergence
    # from the reference, which would report frac_at_lam 1.0 on both of these.
    check("equal neighbours take the full lambda dose",
          abs(flat["alpha_mean"] - 0.5) < 1e-9 and flat["frac_at_lam"] == 1.0,
          f"alpha {flat['alpha_mean']:.4f}, at lam {flat['frac_at_lam']:.2f}")
    check("a worse neighbour reduces the dose below lambda",
          disc["alpha_mean"] < 0.5 and disc["frac_at_lam"] == 0.0,
          f"alpha {disc['alpha_mean']:.4f}, at lam {disc['frac_at_lam']:.2f}")


if __name__ == "__main__":
    main()
