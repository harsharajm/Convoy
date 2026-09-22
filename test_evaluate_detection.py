"""
Unit tests for evaluate_detection.py, against hand-computed values.

An in-house AP implementation is only defensible if it is verified - this is the
cheap half of that (runs anywhere, no torch, no dataset). The expensive half is
validate_against_pycocotools(), which needs the cluster.

Run:  python3 test_evaluate_detection.py
"""

import numpy as np

from evaluate_detection import (
    DetectionEvaluator, iou_matrix, compute_forgetting, compute_bwt,
    IOU_THRESHOLDS, REC_THRESHOLDS,
)

CAR, SIGN = 1, 2  # dataset.py labels: 0 is background, so car == index 0 + 1
_passed, _failed = 0, 0


def check(name, got, want, tol=1e-9):
    global _passed, _failed
    ok = (np.isnan(got) and np.isnan(want)) if isinstance(want, float) and np.isnan(want) \
        else abs(got - want) <= tol
    if ok:
        _passed += 1
        print(f"  PASS  {name}: {got}")
    else:
        _failed += 1
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")


def box(x, y, w=40, h=40):
    return [x, y, x + w, y + h]


def one_image(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    ev = DetectionEvaluator()
    ev.add_batch(
        [{"boxes": np.array(pred_boxes, dtype=float).reshape(-1, 4),
          "labels": np.array(pred_labels, dtype=np.int64),
          "scores": np.array(pred_scores, dtype=float)}],
        [{"boxes": np.array(gt_boxes, dtype=float).reshape(-1, 4),
          "labels": np.array(gt_labels, dtype=np.int64)}],
    )
    return ev.summarize()


# --- IoU --------------------------------------------------------------------

print("\n[IoU]")
# Two identical 40x40 boxes.
check("identical boxes", float(iou_matrix([box(0, 0)], [box(0, 0)])[0, 0]), 1.0)
# Disjoint.
check("disjoint boxes", float(iou_matrix([box(0, 0)], [box(500, 500)])[0, 0]), 0.0)
# Half overlap: 40x40 vs 40x40 offset by 20 in x.
#   intersection 20*40=800, union 1600+1600-800=2400 -> 1/3
check("half overlap", float(iou_matrix([box(0, 0)], [box(20, 0)])[0, 0]), 1 / 3)


# --- the worked example from eval_qna.md D3 ---------------------------------
# 3 real cars. P1 .95 hits car1, P2 .80 hits car2, P3 .60 hits nothing,
# P4 .40 hits car3, P5 .20 hits nothing.
#
# At IoU 0.5, precision after each detection: 1/1, 2/2, 2/3, 3/4, 3/5
#                          cumulative recall: 1/3, 2/3, 2/3, 3/3, 3/3
# Monotone envelope -> [1, 1, .75, .75, .6]
# 101-point sampling: 67 stops land on precision 1.0, 34 land on 0.75
#   AP50 = (67*1.0 + 34*0.75) / 101 = 92.5/101
print("\n[worked example - eval_qna.md D3]")
gt = [box(0, 0), box(200, 0), box(400, 0)]
pred = [box(0, 0), box(200, 0), box(800, 400), box(400, 0), box(900, 500)]
s = one_image(pred, [CAR] * 5, [0.95, 0.80, 0.60, 0.40, 0.20], gt, [CAR] * 3)
check("AP50", s["per_class"]["car"]["AP50"], 92.5 / 101)
check("N", s["per_class"]["car"]["N"], 3, tol=0)
# Operating point conf>=0.5: keeps P1,P2,P3 -> TP=2, FP=1, FN=1
check("P@0.5", s["per_class"]["car"]["P@0.5"], 2 / 3)
check("R@0.5", s["per_class"]["car"]["R@0.5"], 2 / 3)
check("F1@0.5", s["per_class"]["car"]["F1@0.5"], 2 / 3)
# AR@100 is max recall averaged over IoU; all boxes are exact so recall is 1.0
check("AR@100", s["per_class"]["car"]["AR@100"], 1.0)


# --- perfect and degenerate cases -------------------------------------------

print("\n[perfect / degenerate]")
s = one_image([box(0, 0), box(200, 0)], [CAR, CAR], [0.9, 0.8],
              [box(0, 0), box(200, 0)], [CAR, CAR])
check("perfect -> AP@[.5:.95] == 1", s["per_class"]["car"]["AP@[.5:.95]"], 1.0)
check("perfect -> F1 == 1", s["per_class"]["car"]["F1@0.5"], 1.0)

s = one_image([], [], [], [box(0, 0)], [CAR])
check("no predictions -> AP == 0", s["per_class"]["car"]["AP@[.5:.95]"], 0.0)
check("no predictions -> FN counted", s["per_class"]["car"]["R@0.5"], 0.0)

s = one_image([box(0, 0)], [CAR], [0.9], [box(0, 0)], [SIGN])
check("no gt for class -> AP is nan, not 0",
      s["per_class"]["car"]["AP@[.5:.95]"], float("nan"))


# --- duplicate boxes must cost (eval_qna.md C2, C3) -------------------------

print("\n[duplicates cost]")
# One real car, one perfect prediction.
s1 = one_image([box(0, 0)], [CAR], [0.9], [box(0, 0)], [CAR])
check("1 box on 1 car -> AP == 1", s1["per_class"]["car"]["AP50"], 1.0)

# Same, plus a duplicate on the SAME car. Only one may claim it, so the other
# is an FP - and the operating point sees it immediately.
s2 = one_image([box(0, 0), box(1, 1)], [CAR, CAR], [0.9, 0.85], [box(0, 0)], [CAR])
check("duplicate -> P@0.5 == 0.5", s2["per_class"]["car"]["P@0.5"], 0.5)
check("duplicate -> R@0.5 == 1.0", s2["per_class"]["car"]["R@0.5"], 1.0)

# ...but AP does NOT see it, and that is correct behaviour, not a bug.
# AP measures RANKING: an FP scored below every TP arrives after recall has
# already maxed out, so it never lowers the interpolated curve. This is a real
# blind spot of AP and precisely why spec 3 also reports P/R/F1 at a fixed
# operating point - AP alone would call this model perfect.
check("AP is blind to FPs ranked below all TPs", s2["per_class"]["car"]["AP50"], 1.0)

# Rank the junk ABOVE the true box and AP collapses, as it should.
s3 = one_image([box(800, 400), box(0, 0)], [CAR, CAR], [0.95, 0.90],
               [box(0, 0)], [CAR])
check("FP ranked above the TP halves AP", s3["per_class"]["car"]["AP50"], 0.5, tol=1e-9)

# The sprayer: 50 confident junk boxes ahead of one correct low-scoring box.
# Precision can never exceed 1/51 at any recall, so AP is ~0.0196.
rng = np.random.default_rng(7)
spam = [box(float(x), float(y)) for x, y in
        zip(rng.uniform(600, 1200, 50), rng.uniform(300, 650, 50))]
s4 = one_image(spam + [box(0, 0)], [CAR] * 51, [0.99] * 50 + [0.50],
               [box(0, 0)], [CAR])
check("50-box sprayer scores near zero", s4["per_class"]["car"]["AP50"], 1 / 51, tol=1e-9)
check("sprayer P@0.5 == 1/51", s4["per_class"]["car"]["P@0.5"], 1 / 51, tol=1e-9)


# --- matching is class-scoped (eval_qna.md C7) ------------------------------

print("\n[class-scoped matching]")
# A 'car' box placed exactly on a real SIGN: FP for car AND FN for sign.
s = one_image([box(0, 0)], [CAR], [0.9], [box(0, 0)], [SIGN])
check("wrong class -> FP for predicted class",
      s["per_class"]["car"]["P@0.5"], 0.0)
check("wrong class -> FN for true class",
      s["per_class"]["traffic sign"]["R@0.5"], 0.0)
check("wrong class -> true class AP == 0",
      s["per_class"]["traffic sign"]["AP@[.5:.95]"], 0.0)


# --- IoU threshold changes the verdict (eval_qna.md G1) ---------------------

print("\n[IoU threshold moves the curve]")
# Box overlapping a car at IoU ~0.60: TP at 0.5, FP at 0.75.
# 40x40 at (0,0) vs 40x40 at (d,0): inter=(40-d)*40, union=3200-(40-d)*40
# IoU 0.6 -> (40-d)*40 = 0.6*(3200-(40-d)*40) -> let A=(40-d)*40
# A = 1920 - 0.6A -> 1.6A = 1920 -> A = 1200 -> 40-d = 30 -> d = 10
s = one_image([box(10, 0)], [CAR], [0.9], [box(0, 0)], [CAR])
check("IoU 0.60 box -> AP50 == 1 (matched)", s["per_class"]["car"]["AP50"], 1.0)
check("IoU 0.60 box -> AP75 == 0 (rejected)", s["per_class"]["car"]["AP75"], 0.0)


# --- top-100 cap (spec 7.2) -------------------------------------------------

print("\n[detection cap]")
# 150 predictions; the 100 highest-scoring are kept. Put the only correct box
# at the lowest score so it is dropped by the cap.
low = [box(0, 0)]
high = [box(float(x), 700.0) for x in range(149)]
s = one_image(high + low, [CAR] * 150, [0.9] * 149 + [0.01], [box(0, 0)], [CAR])
check("cap drops the sub-100 correct box", s["per_class"]["car"]["AP50"], 0.0)


# --- frozen core / floor flags ----------------------------------------------

print("\n[frozen core and floor]")
s = one_image([box(0, 0)], [CAR], [0.9], [box(0, 0)], [CAR])
check("core incomplete when 3 of 4 core classes absent",
      float(s["core_complete"]), 0.0)
check("core_missing lists the 3 absent core classes",
      len(s["core_missing"]), 3, tol=0)
check("car flagged below floor at N=1", float(s["per_class"]["car"]["below_floor"]), 1.0)


# --- continual-learning aggregation -----------------------------------------

print("\n[forgetting / BWT]")
# Task 0: 0.41 -> 0.36 -> 0.33 (best 0.41, final 0.33, forget 0.08)
# Task 1: 0.38 -> 0.35            (best 0.38, final 0.35, forget 0.03)
# Task 2: no re-test after it, excluded.
hist = {0: [0.41, 0.36, 0.33], 1: [0.38, 0.35], 2: [0.44]}
check("forgetting == mean(0.08, 0.03)", compute_forgetting(hist), 0.055, tol=1e-9)
# BWT: (0.33-0.41) = -0.08, (0.35-0.38) = -0.03 -> mean -0.055
check("bwt == mean(-0.08, -0.03)", compute_bwt(hist), -0.055, tol=1e-9)
# Backward transfer: a task that IMPROVED after later training.
# forgetting cannot express this (max includes final, so it clamps to 0).
up = {0: [0.30, 0.35, 0.40], 1: [0.44]}
check("forgetting hides improvement (== 0)", compute_forgetting(up), 0.0)
check("bwt reports improvement (> 0)", compute_bwt(up), 0.10, tol=1e-9)


# --- constants match the spec -----------------------------------------------

print("\n[spec constants]")
check("10 IoU thresholds", len(IOU_THRESHOLDS), 10, tol=0)
check("IoU range 0.50 to 0.95", float(IOU_THRESHOLDS[0] + IOU_THRESHOLDS[-1]), 1.45, tol=1e-9)
check("101 recall stops", len(REC_THRESHOLDS), 101, tol=0)


print(f"\n{'=' * 50}\n{_passed} passed, {_failed} failed\n{'=' * 50}")
raise SystemExit(1 if _failed else 0)
