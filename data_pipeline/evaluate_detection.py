"""
BDD100K object detection evaluation.

Implements EVAL_PIPELINE_DETECTION.md. Nothing carries over from
BDDTA/code_bdd100k/evaluate.py - that file evaluates multi-label classification,
where predictions are a fixed-length vector. Detection emits a variable-length
set of scored boxes, so there is no denominator until predictions are matched
to ground truth. Matching is the primitive everything else is built on.

Naming follows spec 4.1 - bare `AP` is never used:
  AP50          one class, IoU 0.50
  AP@[.5:.95]   one class, averaged over 10 IoU thresholds
  mAP50         frozen core, IoU 0.50
  mAP@[.5:.95]  frozen core, averaged over IoU  <- the scalar fed to acc_history

The AP computation follows COCO (101-point interpolation, greedy score-ordered
matching, area-range ignore semantics) so numbers are comparable to published
detection results. validate_against_pycocotools() checks that claim.

Evaluation is read-only (spec 0.1): it never touches a weight, a gossip
decision, or a threshold. Every threshold below is frozen by the spec.
"""

import numpy as np
import torch

from partition import OBJECT_CLASSES, USABLE_WEATHER, USABLE_TIMEOFDAY

# --- frozen by the spec, before any training run ---------------------------
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)      # spec 4
REC_THRESHOLDS = np.linspace(0.0, 1.0, 101)      # spec 2, COCO 101-point
MAX_DETS = 100                                   # spec 7.2 - max boxes/image is 91
OP_CONF = 0.50                                   # spec 3, operating point
OP_IOU = 0.50                                    # spec 3
FROZEN_CORE = ["car", "traffic sign", "traffic light", "person"]  # spec 6.2
CLASS_FLOOR = 50                                 # spec 6.2, min GT instances

# COCO size bands, as box area in px^2 (spec 5). 32^2 = 1024, 96^2 = 9216.
AREA_RANGES = {
    "all":    (0.0, float("inf")),
    "small":  (0.0, 1024.0),
    "medium": (1024.0, 9216.0),
    "large":  (9216.0, float("inf")),
}

_EPS = np.spacing(1)


def iou_matrix(dt_boxes, gt_boxes):
    """[n_dt,4] x [n_gt,4] xyxy -> [n_dt,n_gt] IoU. Empty-safe."""
    if len(dt_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(dt_boxes), len(gt_boxes)), dtype=np.float64)

    dt = np.asarray(dt_boxes, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)

    dt_area = (dt[:, 2] - dt[:, 0]).clip(0) * (dt[:, 3] - dt[:, 1]).clip(0)
    gt_area = (gt[:, 2] - gt[:, 0]).clip(0) * (gt[:, 3] - gt[:, 1]).clip(0)

    x1 = np.maximum(dt[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(dt[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(dt[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(dt[:, None, 3], gt[None, :, 3])

    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = dt_area[:, None] + gt_area[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, _EPS), 0.0)


def _match(ious, gt_ignore):
    """
    Greedy score-ordered matching for one image, one class, all IoU thresholds.

    `ious` must arrive with detections already sorted by descending score and
    ground truth already ordered non-ignored-first. Greedy (not Hungarian) is
    deliberate: optimal assignment would let a low-confidence box rescue a GT
    that a high-confidence box already claimed, which rewards spraying boxes
    (spec 1).

    Returns (matched, ignored), each [n_iou, n_dt] bool.
    """
    n_dt, n_gt = ious.shape
    T = len(IOU_THRESHOLDS)
    matched = np.zeros((T, n_dt), dtype=bool)
    ignored = np.zeros((T, n_dt), dtype=bool)

    for t, thr in enumerate(IOU_THRESHOLDS):
        gt_taken = np.zeros(n_gt, dtype=bool)
        for d in range(n_dt):
            best_iou = min(thr, 1 - 1e-10)
            best_g = -1
            for g in range(n_gt):
                if gt_taken[g]:
                    continue  # one claim per ground-truth box (spec 1)
                # GT is ordered non-ignored first, so once we hold a real match
                # there is nothing better left among the ignored tail.
                if best_g > -1 and not gt_ignore[best_g] and gt_ignore[g]:
                    break
                if ious[d, g] < best_iou:
                    continue
                best_iou = ious[d, g]
                best_g = g
            if best_g == -1:
                continue
            matched[t, d] = True
            ignored[t, d] = gt_ignore[best_g]
            gt_taken[best_g] = True

    return matched, ignored


class DetectionEvaluator:
    """
    Accumulates matches over a whole cell, then summarizes.

    Curves are transient (spec E4): a cell evaluation builds 100 precision-recall
    curves, integrates each, and discards them. Nothing is stored or plotted.

    Deliberately decoupled from the model so it can be unit-tested and
    cross-checked against pycocotools without a forward pass.
    """

    def __init__(self, class_names=None, frozen_core=None):
        self.class_names = list(class_names or OBJECT_CLASSES)
        self.frozen_core = list(frozen_core if frozen_core is not None else FROZEN_CORE)
        # per class: list of per-image records, in insertion order
        self._per_class = {c: [] for c in self.class_names}
        self.n_images = 0

    def add_batch(self, preds, targets):
        """
        preds:   list of {"boxes":[M,4] xyxy, "labels":[M], "scores":[M]}
        targets: list of {"boxes":[N,4] xyxy, "labels":[N]}

        Both in ORIGINAL image pixel coordinates - dataset.py does no resize, so
        a model that resizes internally must map predictions back first
        (spec 10). NMS must already have been applied by the model; this does
        not deduplicate.
        """
        for pred, target in zip(preds, targets):
            self._add_image(pred, target)
            self.n_images += 1

    def _add_image(self, pred, target):
        p_boxes = _to_numpy(pred["boxes"]).reshape(-1, 4)
        p_labels = _to_numpy(pred["labels"]).reshape(-1).astype(np.int64)
        p_scores = _to_numpy(pred["scores"]).reshape(-1).astype(np.float64)
        t_boxes = _to_numpy(target["boxes"]).reshape(-1, 4)
        t_labels = _to_numpy(target["labels"]).reshape(-1).astype(np.int64)

        # Top-100 cap (spec 7.2). Bounds low-confidence FP spam in the AP tail;
        # provably lossless here since no BDD100K image holds more than 91 boxes.
        if len(p_scores) > MAX_DETS:
            keep = np.argsort(-p_scores, kind="mergesort")[:MAX_DETS]
            keep.sort()
            p_boxes, p_labels, p_scores = p_boxes[keep], p_labels[keep], p_scores[keep]

        for ci, name in enumerate(self.class_names):
            label = ci + 1  # dataset.py reserves 0 for background
            dm = p_labels == label
            gm = t_labels == label
            if not dm.any() and not gm.any():
                continue

            dt_b, dt_s = p_boxes[dm], p_scores[dm]
            # Stable sort: ties break by original box index, and because images
            # are appended in loader order the global tiebreak is
            # (-score, image_index, box_index) as required by spec 10.
            order = np.argsort(-dt_s, kind="mergesort")

            self._per_class[name].append({
                "dt_boxes": dt_b[order],
                "dt_scores": dt_s[order],
                "gt_boxes": t_boxes[gm],
            })

    # --- per-class evaluation ---------------------------------------------

    def _eval_class(self, name, area_range):
        """
        Returns (ap_per_iou [T], max_recall_per_iou [T], n_gt, op_counts).

        AP is nan when the class has no ground truth in this area range - it is
        undefined, not zero, and must be excluded from means rather than
        dragging them down (the same rule the multi-label pipeline used for
        all-negative classes).
        """
        lo, hi = area_range
        records = self._per_class[name]

        all_scores, all_matched, all_ignored = [], [], []
        n_gt = 0

        for rec in records:
            gt = rec["gt_boxes"]
            dt = rec["dt_boxes"]
            gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
            dt_area = (dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1]) if len(dt) else np.zeros(0)

            # GT outside the band is ignored, not deleted: a detection landing on
            # it is neither credited nor penalised (COCO area-range semantics).
            gt_ignore = (gt_area < lo) | (gt_area > hi)
            n_gt += int((~gt_ignore).sum())

            if len(dt) == 0:
                continue

            gt_order = np.argsort(gt_ignore, kind="mergesort")  # non-ignored first
            ious = iou_matrix(dt, gt[gt_order]) if len(gt) else np.zeros((len(dt), 0))
            matched, ignored = _match(ious, gt_ignore[gt_order])

            # An unmatched detection whose own area falls outside the band is
            # also ignored - it was never competing in this band.
            out_of_band = (dt_area < lo) | (dt_area > hi)
            ignored = ignored | (~matched & out_of_band[None, :])

            all_scores.append(rec["dt_scores"])
            all_matched.append(matched)
            all_ignored.append(ignored)

        T = len(IOU_THRESHOLDS)
        op = {"TP": 0, "FP": 0, "FN": n_gt}

        if n_gt == 0:
            return np.full(T, np.nan), np.full(T, np.nan), 0, op
        if not all_scores:
            return np.zeros(T), np.zeros(T), n_gt, op

        scores = np.concatenate(all_scores)
        matched = np.concatenate(all_matched, axis=1)
        ignored = np.concatenate(all_ignored, axis=1)

        order = np.argsort(-scores, kind="mergesort")
        scores, matched, ignored = scores[order], matched[:, order], ignored[:, order]

        tps = matched & ~ignored
        fps = ~matched & ~ignored
        tp_cum = np.cumsum(tps, axis=1, dtype=np.float64)
        fp_cum = np.cumsum(fps, axis=1, dtype=np.float64)

        ap = np.zeros(T)
        max_rec = np.zeros(T)
        for t in range(T):
            tp, fp = tp_cum[t], fp_cum[t]
            rc = tp / n_gt
            pr = tp / np.maximum(tp + fp, _EPS)
            max_rec[t] = rc[-1] if len(rc) else 0.0

            # Monotone envelope: best precision at this recall or beyond
            # (spec E1) - sweep right to left.
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            # Sample the staircase at 101 evenly spaced recall stops and average
            # the heights. Base width is 1, so mean height == area == AP.
            idx = np.searchsorted(rc, REC_THRESHOLDS, side="left")
            q = np.zeros(len(REC_THRESHOLDS))
            valid = idx < len(pr)
            q[valid] = pr[idx[valid]]
            ap[t] = q.mean()

        # Operating point (spec 3): conf >= 0.5 at IoU 0.5. Dropping the
        # low-score tail cannot change the greedy matches of higher-score
        # detections, so this reads straight off the same arrays.
        t50 = int(np.argmin(np.abs(IOU_THRESHOLDS - OP_IOU)))
        kept = scores >= OP_CONF
        op = {
            "TP": int((tps[t50] & kept).sum()),
            "FP": int((fps[t50] & kept).sum()),
        }
        op["FN"] = n_gt - op["TP"]

        return ap, max_rec, n_gt, op

    # --- summary -----------------------------------------------------------

    def summarize(self):
        """
        Returns the full metric set for one cell. Every per-class entry carries
        its ground-truth instance count N (spec 9 Layer A) - a metric without
        its sample size is not reportable.
        """
        per_class = {}
        for name in self.class_names:
            ap_all, rec_all, n_gt, op = self._eval_class(name, AREA_RANGES["all"])
            ap_s, _, _, _ = self._eval_class(name, AREA_RANGES["small"])
            ap_m, _, _, _ = self._eval_class(name, AREA_RANGES["medium"])
            ap_l, _, _, _ = self._eval_class(name, AREA_RANGES["large"])

            prec = op["TP"] / (op["TP"] + op["FP"]) if (op["TP"] + op["FP"]) else 0.0
            rec = op["TP"] / (op["TP"] + op["FN"]) if (op["TP"] + op["FN"]) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

            i50 = int(np.argmin(np.abs(IOU_THRESHOLDS - 0.50)))
            i75 = int(np.argmin(np.abs(IOU_THRESHOLDS - 0.75)))

            per_class[name] = {
                "AP@[.5:.95]": float(np.mean(ap_all)),
                "AP50": float(ap_all[i50]),
                "AP75": float(ap_all[i75]),
                "AP_S": float(np.mean(ap_s)),
                "AP_M": float(np.mean(ap_m)),
                "AP_L": float(np.mean(ap_l)),
                "AR@100": float(np.mean(rec_all)),
                "P@0.5": prec,
                "R@0.5": rec,
                "F1@0.5": f1,
                "N": n_gt,
                "below_floor": n_gt < CLASS_FLOOR,
            }

        # Macro over the frozen core only. Cross-cell comparability requires a
        # fixed class set: per-cell evaluable classes range from 1/10 to 9/10,
        # so a per-cell "macro over whatever was present" would average
        # different quantities in different cells (spec 6.2).
        core_present = [c for c in self.frozen_core
                        if per_class[c]["N"] > 0 and not np.isnan(per_class[c]["AP@[.5:.95]"])]
        missing = [c for c in self.frozen_core if c not in core_present]

        def macro(key):
            vals = [per_class[c][key] for c in core_present]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "mAP@[.5:.95]": macro("AP@[.5:.95]"),   # <- the scalar
            "mAP50": macro("AP50"),
            "mAP75": macro("AP75"),
            "mAR@100": macro("AR@100"),
            "macroP@0.5": macro("P@0.5"),
            "macroR@0.5": macro("R@0.5"),
            "macroF1@0.5": macro("F1@0.5"),
            "per_class": per_class,
            "core_classes": core_present,
            "core_missing": missing,
            "core_complete": not missing,
            "n_images": self.n_images,
            "n_instances": sum(v["N"] for v in per_class.values()),
            "below_floor_classes": [c for c, v in per_class.items()
                                    if 0 < v["N"] < CLASS_FLOOR],
        }


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --- model-facing entry points ---------------------------------------------

@torch.no_grad()
def evaluate_detection(model, dataloader, device, class_names=None, frozen_core=None):
    """
    Run a detector over one cell's test loader and summarize.

    Read-only by construction - no backward pass, no optimizer step. The output
    is allowed to change what we conclude and nothing else (spec 0.1).
    """
    model = model.to(device)
    model.eval()

    ev = DetectionEvaluator(class_names, frozen_core)
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        preds = model(images)
        ev.add_batch(preds, targets)
    return ev.summarize()


@torch.no_grad()
def evaluate_tasks_detection(model, task_loaders, device, class_names=None, frozen_core=None):
    """
    task_loaders: {task_id: DataLoader} or list. Returns {task_id: summary}.

    Feed summary['mAP@[.5:.95]'] into acc_history[cid][task_id] - the direct
    replacement for the multi-label pipeline's 'mAcc'.
    """
    if isinstance(task_loaders, list):
        task_loaders = dict(enumerate(task_loaders))
    return {
        tid: evaluate_detection(model, loader, device, class_names, frozen_core)
        for tid, loader in task_loaders.items()
    }


# --- continual-learning aggregation ----------------------------------------

def compute_forgetting(history):
    """
    history[task_id] = [mAP after task 0, after task 1, ...] for one client.

    Forgetting = mean over past tasks of (best-ever - final). Kept for
    comparability with the CL literature, but note it is >= 0 by construction
    since max() includes the final value - it structurally cannot report that
    gossip IMPROVED a past task. compute_bwt() is what tests that (spec 9C).
    """
    task_ids = sorted(history)
    past = [t for t in task_ids[:-1] if len(history[t]) >= 2]
    if not past:
        return 0.0
    return float(np.mean([max(history[t]) - history[t][-1] for t in past]))


def compute_bwt(history):
    """
    Backward transfer = mean over past tasks of (final - first).

    Signed: positive means later training helped an earlier task, which is the
    project's actual hypothesis and the thing compute_forgetting() cannot express.
    """
    task_ids = sorted(history)
    past = [t for t in task_ids[:-1] if len(history[t]) >= 2]
    if not past:
        return 0.0
    return float(np.mean([history[t][-1] - history[t][0] for t in past]))


def client_summary(history):
    """Final mAP / Forgetting / BWT for one client, from its accuracy history."""
    task_ids = sorted(history)
    finals = [history[t][-1] for t in task_ids if history[t]]
    return {
        "final_mAP": float(np.mean(finals)) if finals else 0.0,
        "forgetting": compute_forgetting(history),
        "bwt": compute_bwt(history),
        "n_tasks": len(finals),
    }


def system_summary(client_summaries):
    """
    Mean across clients, plus the spread. A mean alone hides the failure that
    matters: if one client collapses to 0.18 while another reaches 0.52 the
    average still looks respectable (spec 9D).
    """
    def spread(key):
        vals = [c[key] for c in client_summaries.values()]
        return {
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
        }
    return {
        "mAP": spread("final_mAP"),
        "forgetting": spread("forgetting"),
        "bwt": spread("bwt"),
    }


# --- cross-client transfer: the headline (spec 9E) --------------------------

@torch.no_grad()
def evaluate_cross_client(client_models, pooled_loaders, device,
                          class_names=None, frozen_core=None):
    """
    Each client's FINAL model against every OTHER client's pooled test data.

    This is the headline result, not an optional extra: it is where the gossip
    claim is won or lost. Per-client own-weather mAP would be the wrong headline
    - a local-only client is never pulled off its own distribution, so
    personalization is the framing where this method has its weakest case
    (spec 0).

    pooled_loaders[j] pools client j's 3 task test splits into one loader, which
    makes this the most statistically solid part of the report - the opposite of
    the thin per-cell counts that forced the class floor in spec 6.2.

    Computed once, after every client has finished. Never feeds back.
    """
    ids = list(client_models.keys())
    matrix, own = {}, {}

    for i in ids:
        model = client_models[i].to(device)
        model.eval()
        row = {}
        for j in ids:
            m = evaluate_detection(model, pooled_loaders[j], device, class_names, frozen_core)
            entry = {"mAP@[.5:.95]": m["mAP@[.5:.95]"], "mAP50": m["mAP50"],
                     "mAR@100": m["mAR@100"]}
            if j == i:
                own[i] = entry          # diagonal: own-vs-other gap needs it
            else:
                row[j] = entry
        matrix[i] = row

    per_client = {}
    for i in ids:
        vals = [r["mAP@[.5:.95]"] for r in matrix[i].values()]
        cross = float(np.mean(vals)) if vals else 0.0
        per_client[i] = {
            "cross_mAP": cross,
            "own_mAP": own[i]["mAP@[.5:.95]"],
            # Shrinking across rounds = transfer happening. Stable = five models
            # trained in parallel that never learned from each other.
            "own_vs_other_gap": own[i]["mAP@[.5:.95]"] - cross,
        }

    return {
        "matrix": matrix,
        "diagonal": own,
        "per_client": per_client,
        # The headline number. Compare against the local-only baseline: the
        # difference is what gossip transferred. If it is <= 0, say so.
        "global_cross_mAP": float(np.mean([c["cross_mAP"] for c in per_client.values()])),
        "global_own_vs_other_gap": float(np.mean([c["own_vs_other_gap"]
                                                  for c in per_client.values()])),
    }


# --- reporting --------------------------------------------------------------

def print_cell(summary, label=""):
    """Per-class table for one cell. N beside every value, always."""
    print(f"\n--- {label} --- {summary['n_images']} images, "
          f"{summary['n_instances']} instances")
    if not summary["core_complete"]:
        print(f"  !! frozen core incomplete, missing: {summary['core_missing']} "
              f"- mAP is NOT comparable to other cells")

    print(f"  {'class':16}{'N':>8}{'AP@[.5:.95]':>13}{'AP50':>8}{'AP75':>8}"
          f"{'AP_S':>8}{'AR@100':>9}{'F1@0.5':>9}")
    for name, m in summary["per_class"].items():
        core = "*" if name in summary["core_classes"] else " "
        floor = " (below floor)" if m["below_floor"] else ""
        print(f"{core} {name:16}{m['N']:>8}{m['AP@[.5:.95]']:>13.4f}{m['AP50']:>8.4f}"
              f"{m['AP75']:>8.4f}{m['AP_S']:>8.4f}{m['AR@100']:>9.4f}"
              f"{m['F1@0.5']:>9.4f}{floor}")

    print(f"  {'':16}{'':8}{'-' * 13}")
    print(f"  mAP@[.5:.95] (core {len(summary['core_classes'])}): "
          f"{summary['mAP@[.5:.95]']:.4f}   mAP50: {summary['mAP50']:.4f}   "
          f"mAR@100: {summary['mAR@100']:.4f}   macroF1@0.5: {summary['macroF1@0.5']:.4f}")


def print_cross_client(cross, client_names=None):
    names = client_names or {}
    print("\n" + "=" * 60)
    print("CROSS-CLIENT TRANSFER (final models) -- HEADLINE")
    print("=" * 60)
    for i, row in cross["matrix"].items():
        print(f"Client {i} ({names.get(i, i)}) tested on:")
        for j, m in row.items():
            print(f"  {str(names.get(j, j)):16} -> mAP@[.5:.95]: {m['mAP@[.5:.95]']:.4f}  "
                  f"mAP50: {m['mAP50']:.4f}")
        pc = cross["per_client"][i]
        print(f"  {'CROSS AVG':16} -> {pc['cross_mAP']:.4f}   "
              f"(own: {pc['own_mAP']:.4f}, gap: {pc['own_vs_other_gap']:+.4f})")
    print("-" * 60)
    print(f"Global Cross-Client mAP@[.5:.95]: {cross['global_cross_mAP']:.4f}")
    print(f"Global own-vs-other gap:          {cross['global_own_vs_other_gap']:+.4f}")
    print("Compare against the local-only baseline: the difference is what")
    print("gossip transferred. If it is <= 0, that is the result.")


# --- verification -----------------------------------------------------------

def validate_against_pycocotools(evaluator, tol=1e-6):
    """
    Cross-check our AP against pycocotools on the accumulated data.

    Rolling our own AP is only defensible if it is verified - an unverified
    in-house metric is the easiest place for a self-serving bug to hide
    (spec 11). Run once on a fixture; not part of the training loop.

    Returns {class_name: (ours, theirs, abs_diff)} or raises if the mismatch
    exceeds tol.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    ann_id = 1
    n_img = max((len(recs) for recs in evaluator._per_class.values()), default=0)
    for k in range(n_img):
        images.append({"id": k + 1, "width": 1280, "height": 720})

    for ci, name in enumerate(evaluator.class_names):
        for k, rec in enumerate(evaluator._per_class[name]):
            for b in rec["gt_boxes"]:
                w, h = float(b[2] - b[0]), float(b[3] - b[1])
                anns.append({"id": ann_id, "image_id": k + 1, "category_id": ci + 1,
                             "bbox": [float(b[0]), float(b[1]), w, h],
                             "area": w * h, "iscrowd": 0})
                ann_id += 1
            for b, s in zip(rec["dt_boxes"], rec["dt_scores"]):
                dets.append({"image_id": k + 1, "category_id": ci + 1,
                             "bbox": [float(b[0]), float(b[1]),
                                      float(b[2] - b[0]), float(b[3] - b[1])],
                             "score": float(s)})

    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns,
                    "categories": [{"id": i + 1, "name": n}
                                   for i, n in enumerate(evaluator.class_names)]}
    coco.createIndex()

    ev = COCOeval(coco, coco.loadRes(dets), "bbox")
    ev.params.maxDets = [1, 10, MAX_DETS]
    ev.evaluate(); ev.accumulate()

    ours = evaluator.summarize()["per_class"]
    out = {}
    for ci, name in enumerate(evaluator.class_names):
        p = ev.eval["precision"][:, :, ci, 0, 2]
        theirs = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
        mine = ours[name]["AP@[.5:.95]"]
        if np.isnan(mine) and np.isnan(theirs):
            continue
        diff = abs(mine - theirs)
        out[name] = (mine, theirs, diff)
        if diff > tol:
            raise AssertionError(
                f"AP mismatch for {name}: ours={mine:.8f} pycocotools={theirs:.8f} "
                f"diff={diff:.2e} > tol={tol:.0e}")
    return out


if __name__ == "__main__":
    # Synthetic smoke test - no model, no data, runs anywhere.
    rng = np.random.default_rng(0)
    ev = DetectionEvaluator()

    for _ in range(40):
        n_gt = rng.integers(2, 12)
        labels = rng.integers(1, len(OBJECT_CLASSES) + 1, size=n_gt)
        x1 = rng.uniform(0, 1100, n_gt); y1 = rng.uniform(0, 600, n_gt)
        w = rng.uniform(10, 160, n_gt); h = rng.uniform(10, 110, n_gt)
        gt = np.stack([x1, y1, x1 + w, y1 + h], axis=1)

        # Most GT recovered with jitter, plus a few pure false positives.
        keep = rng.random(n_gt) > 0.25
        jitter = rng.normal(0, 4, size=(int(keep.sum()), 4))
        p_boxes = gt[keep] + jitter
        p_labels = labels[keep]
        p_scores = rng.uniform(0.35, 0.99, int(keep.sum()))

        n_fp = rng.integers(0, 4)
        if n_fp:
            fx = rng.uniform(0, 1100, n_fp); fy = rng.uniform(0, 600, n_fp)
            fw = rng.uniform(10, 160, n_fp); fh = rng.uniform(10, 110, n_fp)
            p_boxes = np.concatenate([p_boxes, np.stack([fx, fy, fx + fw, fy + fh], 1)])
            p_labels = np.concatenate([p_labels, rng.integers(1, 11, n_fp)])
            p_scores = np.concatenate([p_scores, rng.uniform(0.05, 0.5, n_fp)])

        ev.add_batch(
            [{"boxes": p_boxes, "labels": p_labels, "scores": p_scores}],
            [{"boxes": gt, "labels": labels}],
        )

    print_cell(ev.summarize(), label="synthetic smoke test")

    # Continual-learning aggregation on a hand-made history.
    history = {0: [0.41, 0.36, 0.33], 1: [0.38, 0.35], 2: [0.44]}
    print("\nforgetting:", round(compute_forgetting(history), 4),
          " bwt:", round(compute_bwt(history), 4),
          " summary:", {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in client_summary(history).items()})

    try:
        print("\npycocotools cross-check:")
        for name, (a, b, d) in validate_against_pycocotools(ev).items():
            print(f"  {name:16} ours={a:.6f}  theirs={b:.6f}  diff={d:.2e}")
    except ImportError:
        print("\npycocotools not installed - cross-check skipped (run on the cluster)")
