"""
The detector: COCO-pretrained Faster R-CNN with a fresh 11-class head.

Why so little is trainable. BDD100K's 10 classes are nearly all present in
COCO's pretraining data under different names - car, bus, truck, train,
person, bicycle~bike, motorcycle~motor, traffic light. The pretrained
backbone already produces features that fit this domain, so the only part
with real work to do is the head that maps proposals onto BDD100K's label
set (plus `traffic sign` and `rider`, which have no direct COCO analogue).

So: ResNet backbone, FPN and RPN are frozen; roi_heads is trained. That also
removes the question of what learning rate the pretrained components get -
nothing pretrained below roi_heads is updated at all.

Two things deliberately left at their pretrained defaults:
  - anchor sizes. 55.5% of BDD100K boxes are under 32px, so tuned anchors
    would help, but the RPN is frozen and its weights were learned against
    the default anchors - changing them would invalidate those weights.
  - the internal transform (resize + ImageNet normalisation). It maps
    predicted boxes back to the original 1280x720 frame on the way out, which
    is what keeps predictions comparable to the untouched ground truth.

Anchors staying default means the RPN is a suspected bottleneck for small
objects. Before touching any frozen weight, RPN_PRE_NMS_TOP_N_TEST /
RPN_POST_NMS_TOP_N_TEST (config.py) widen how many proposals survive to the
head - a pure inference-time budget change, nothing learned moves. If that
alone recovers small-class AP, the RPN's weights were never the problem.
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import (
    FREEZE_BACKBONE, FREEZE_RPN, LR_HEAD, MOMENTUM, NUM_CLASSES,
    RPN_POST_NMS_TOP_N_TEST, RPN_PRE_NMS_TOP_N_TEST, WEIGHT_DECAY,
)


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """COCO-pretrained Faster R-CNN with the box predictor swapped for ours."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        rpn_pre_nms_top_n_test=RPN_PRE_NMS_TOP_N_TEST,
        rpn_post_nms_top_n_test=RPN_POST_NMS_TOP_N_TEST,
    )

    # The COCO predictor emits 91 classes; ours emits 11 (10 + background).
    # Randomly initialised - this is the "new classifier head".
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    freeze_pretrained(model)
    return model


def freeze_pretrained(model, freeze_backbone=FREEZE_BACKBONE, freeze_rpn=FREEZE_RPN):
    """Freezes everything below roi_heads, per design_choices.pdf 2.2."""
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    if freeze_rpn:
        for p in model.rpn.parameters():
            p.requires_grad_(False)
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model, lr=LR_HEAD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY):
    """
    SGD over the trainable head only. No warm-up and no scheduler
    (design_choices.pdf) - the run is 3 epochs per task, too short for a
    decay schedule to matter.
    """
    return torch.optim.SGD(
        trainable_parameters(model),
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )


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
