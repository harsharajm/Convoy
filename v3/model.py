"""
The detector: COCO-pretrained Faster R-CNN with a fresh 11-class head.

Why so little is trainable. BDD100K's 10 classes are nearly all present in
COCO's pretraining data under different names - car, bus, truck, train,
person, bicycle~bike, motorcycle~motor, traffic light. The pretrained
backbone already produces features that fit this domain, so the only part
with real work to do is the head that maps proposals onto BDD100K's label
set (plus `traffic sign` and `rider`, which have no direct COCO analogue).

v3: ResNet backbone and FPN stay frozen; **roi_heads and the RPN train**.

v2 froze the RPN and kept COCO's anchors, and its metrics said that was the
binding constraint: recall, not classification, capped mAP (macro mAR@100
0.348 against mAP 0.246), and the recall loss was entirely at small sizes
(mAR_S 0.240 vs mAR_L 0.507). The cause was a scale mismatch - the evaluation
counts "small" as area < 1024px^2, which is exactly the 32x32 smallest anchor,
so every small object sat at or below the smallest anchor and was never
proposable. Over 1,027,770 training boxes the median is 28.3px, smaller than
that anchor.

So the anchors are halved (config.ANCHOR_SIZES) and the RPN is unfrozen,
because weights learned against 32px anchors do not transfer to 16px ones.
Raising RPN_PRE/POST_NMS_TOP_N_TEST 1000 -> 2000 was tried first, as the
inference-only version of the same idea, and bought ~+0.009 everywhere - real
but not the fix. Higher input resolution was tried too and rejected: it lifts
mAR_S but drops mAR_M/mAR_L, since it rescales every object rather than just
the floor (results/analysis/res_probe.json).

Still at its pretrained default: the internal transform (resize + ImageNet
normalisation). It maps predicted boxes back to the original 1280x720 frame
on the way out, which is what keeps predictions comparable to the untouched
ground truth.

Note the RPN and roi_heads are gradient-isolated: torchvision detaches the
proposal path (rpn.py:371), so the randomly initialised box predictor cannot
disturb the pretrained RPN convs. The RPN trains only from its own objectness
and box-regression losses.
"""

import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import (
    ANCHOR_ASPECT_RATIOS, ANCHOR_SIZES, FREEZE_BACKBONE, FREEZE_BACKBONE_STAGES,
    FREEZE_RPN, LR_HEAD, LR_PRETRAINED, LR_WARMUP, LR_WARMUP_FACTOR, MOMENTUM,
    NUM_CLASSES, RPN_POST_NMS_TOP_N_TEST, RPN_PRE_NMS_TOP_N_TEST, WEIGHT_DECAY,
)


def build_anchor_generator(sizes=ANCHOR_SIZES, ratios=ANCHOR_ASPECT_RATIOS):
    """
    One anchor size per FPN level, three aspect ratios - the same shape as
    torchvision's default, so `num_anchors` per location stays 3 and the
    pretrained RPN head's conv weights still load.

    They load, but they no longer *mean* the same thing: those weights learned
    objectness and box regression against 32px anchors. Hand them 16px anchors
    and the priors are wrong, which is why v3 has to unfreeze the RPN. See
    v3.md.
    """
    return AnchorGenerator(sizes=tuple((s,) for s in sizes),
                           aspect_ratios=tuple(ratios for _ in sizes))


def build_model(num_classes=NUM_CLASSES, pretrained=True,
                backbone_frozen_stages=FREEZE_BACKBONE_STAGES):
    """
    COCO-pretrained Faster R-CNN with the box predictor swapped for ours.

    backbone_frozen_stages is a real parameter (not just read from config
    inside freeze_pretrained) so a caller - e.g. a SLURM batch job - can pin
    its freeze depth via an explicit argument instead of editing config.py.
    Two jobs submitted back to back would otherwise race: sbatch queues the
    job and returns immediately, but the script only reads config.py when it
    actually starts running on a node, which can be well after submission -
    edit config.py for the second job before the first has started and it
    silently picks up the second job's setting.
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        rpn_pre_nms_top_n_test=RPN_PRE_NMS_TOP_N_TEST,
        rpn_post_nms_top_n_test=RPN_POST_NMS_TOP_N_TEST,
        rpn_anchor_generator=build_anchor_generator(),
    )

    # The COCO predictor emits 91 classes; ours emits 11 (10 + background).
    # Randomly initialised - this is the "new classifier head".
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    freeze_pretrained(model, backbone_frozen_stages=backbone_frozen_stages)
    return model


def freeze_pretrained(model, freeze_backbone=FREEZE_BACKBONE, freeze_rpn=FREEZE_RPN,
                      backbone_frozen_stages=FREEZE_BACKBONE_STAGES):
    """
    Freezes the pretrained components that v3 does not train.

    v2 froze everything below roi_heads; v3 leaves the RPN trainable, so with
    the default flags only the backbone (ResNet + FPN) is frozen here.

    backbone_frozen_stages, if set, overrides freeze_backbone with a partial
    freeze matching MMDetection's `frozen_stages`: freeze the stem
    (conv1/bn1) plus `layer1..layer{backbone_frozen_stages}` inside
    `model.backbone.body` (torchvision's IntermediateLayerGetter over the
    ResNet trunk), leaving later stages and `model.backbone.fpn` trainable.
    The BDD100K reference configs use frozen_stages=1 - freeze_backbone=True
    alone freezes the whole backbone+FPN, which is a different, stricter
    setting.
    """
    if backbone_frozen_stages is not None:
        body = model.backbone.body
        for p in body.conv1.parameters():
            p.requires_grad_(False)
        for p in body.bn1.parameters():
            p.requires_grad_(False)
        for stage in range(1, backbone_frozen_stages + 1):
            for p in getattr(body, f"layer{stage}").parameters():
                p.requires_grad_(False)
    elif freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    if freeze_rpn:
        for p in model.rpn.parameters():
            p.requires_grad_(False)
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model, lr=LR_HEAD, lr_pretrained=LR_PRETRAINED,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY):
    """
    SGD over two param groups: pretrained weights and freshly initialised ones.

    v2 had a single group at LR_HEAD, which was correct only while everything
    pretrained was frozen. With the RPN trainable that would apply 0.01 to
    convs COCO spent many epochs learning - too large for weights that only
    need to re-adapt to new anchor scales. (Not because the random head could
    reach them: torchvision detaches the proposal path, so the two are
    gradient-isolated. The problem is the step size alone.) LR_PRETRAINED
    existed in config.py through all of v1 and v2 but was never imported -
    this is the first code that reads it.

    The split is `rpn.*` vs everything else, NOT "pretrained vs random".
    roi_heads.box_head (fc6/fc7) is pretrained COCO too, but v1 and v2 trained
    it at LR_HEAD and their results are the baseline v3 is measured against.
    Dropping it to lr_pretrained here would move two things at once and make
    any change in v3 unattributable. Only the newly unfrozen component gets
    the new learning rate.

    Any `backbone.*` param that requires_grad (FREEZE_BACKBONE_STAGES leaving
    layer2+/fpn trainable) joins the same lr_pretrained group as the RPN, for
    the identical reason LR_PRETRAINED exists at all: those are COCO-trained
    convs re-adapting, not a random head learning from scratch, so they get
    the pretrained group's smaller lr and its warmup, not LR_HEAD's.

    reset_momentum iterates param_groups and keeps working unchanged.
    """
    pretrained, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_pretrained = name.startswith("rpn.") or name.startswith("backbone.")
        (pretrained if is_pretrained else head).append(p)

    groups = []
    if pretrained:
        groups.append({"params": pretrained, "lr": lr_pretrained})
    if head:
        groups.append({"params": head, "lr": lr})
    return torch.optim.SGD(groups, lr=lr, momentum=momentum,
                           weight_decay=weight_decay)


def build_warmup(optimizer, steps=LR_WARMUP, factor=LR_WARMUP_FACTOR):
    """
    Linear lr ramp over the first `steps` optimiser steps, or None if disabled.

    LinearLR scales every param group by the same factor, so the 10x gap
    between the RPN group and the head group is preserved throughout the ramp
    and both arrive at their own target lr together. After `steps` it holds at
    1.0 and never decays - this is warm-up only, not a schedule.

    One per optimiser, so each client warms up on its own step count. Nothing
    here is shared across clients.
    """
    if not steps:
        return None
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=factor, end_factor=1.0, total_iters=steps)


def reset_momentum(optimizer):
    """
    Drops SGD's momentum buffers.

    Called after a client's own weights are replaced by a gossip aggregate.
    The buffer is a running average of gradients taken at the *old* weights;
    once those weights move somewhere else, continuing to apply it pushes the
    next step in a direction that was never measured at the new point.

    Only the client whose weights actually changed calls this - a frozen
    client that merely hands its weights to a neighbour has nothing to reset.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.state[p].pop("momentum_buffer", None)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_parameters(model))
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    model = build_model()
    counts = count_parameters(model)
    print(f"total     : {counts['total']:,}")
    print(f"trainable : {counts['trainable']:,}")
    print(f"frozen    : {counts['frozen']:,}")
    print("\ntrainable modules:")
    seen = set()
    for name, p in model.named_parameters():
        if p.requires_grad:
            top = ".".join(name.split(".")[:2])
            if top not in seen:
                seen.add(top)
                print(" ", top)
