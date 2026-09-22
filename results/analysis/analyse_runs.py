#!/usr/bin/env python3
"""
Analysis of the first batch of successful seed-42 runs.

Reads  results/runs/{gossip,local_only,sequential,joint}_seed42.json
       bdd100k_manifest.csv                     (cell sizes, for the caveats)
Writes results/analysis/figures/*.png
       results/analysis/summary_tables.md       (every number, paste-ready)

Run:
    python3 analyse_runs.py

Needs numpy + matplotlib. If the system python is externally managed:
    python3 -m venv .venv && .venv/bin/pip install matplotlib && .venv/bin/python analyse_runs.py

----------------------------------------------------------------------------
THE TWO METRICS - the single thing to get right before reading any chart
----------------------------------------------------------------------------
The four runs do NOT all report the same number. There are two, and they are
not interchangeable:

M1  "task-averaged final mAP"   json: per_client.final_mAP / system.mAP.mean
    After a client finishes all 3 tasks, score its final model on each of its
    3 task test cells and take the unweighted mean of the 3 cell scores.
    A 10-image cell counts as much as a 2,843-image cell.
    Present for: gossip, local_only, sequential.   Absent for joint - joint
    has no task boundaries, so there are no per-task cells to average.

M2  "pooled own-weather mAP"    json: cross_client.diagonal[c] / joint.per_client_pooled_mAP
    Score the client's final model on its 3 test cells POOLED into one loader.
    Image-weighted, so it is the statistically solid one.
    Present for: gossip, local_only (evaluate_cross_client only runs for these
    two, see train.py::_collect) and joint.   Absent for sequential.

Consequence, and the reason the figures are split into two panels:
  * joint 0.2460 vs sequential 0.2306 is NOT a valid comparison. Different
    metrics.
  * The only 4-way-comparable statement does not exist in this batch. The two
    valid statements are M1 over {gossip, local_only, sequential} and M2 over
    {gossip, local_only, joint}.
"""

import csv
import json
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from vizstyle import (
    apply_style, tidy, caption, MODE_COLOR, MODE_LABEL, MODE_SHORT,
    SEQ_CMAP, CLIENTS, TASKS, INK, INK_2, INK_MUTED, GRID, SURFACE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
OBJ_DET = os.path.dirname(RESULTS)
RUNS = os.path.join(RESULTS, "runs")
FIGS = os.path.join(HERE, "figures")
MANIFEST = os.path.join(OBJ_DET, "bdd100k_manifest.csv")

SEED = 42
CL_MODES = ["gossip", "local_only", "sequential"]        # have M1
POOLED_MODES = ["gossip", "local_only", "joint"]         # have M2
ALL_MODES = ["gossip", "local_only", "sequential", "joint"]

MAP_KEY = "mAP@[.5:.95]"


# ---------------------------------------------------------------- loading --

def load_runs():
    out = {}
    for m in ALL_MODES:
        with open(os.path.join(RUNS, f"{m}_seed{SEED}.json")) as f:
            out[m] = json.load(f)
    return out


def partition_counts():
    """{split: {(client, task): n_images}} straight from the manifest."""
    cnt = Counter()
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            cnt[(r["split"], r["weather"], r["timeofday"])] += 1
    return {
        s: np.array([[cnt[(s, c, t)] for t in TASKS] for c in CLIENTS], dtype=float)
        for s in ("train", "test")
    }


def m1(runs, mode):
    """Task-averaged final mAP, per client + system mean."""
    pc = runs[mode]["per_client"]
    v = np.array([pc[c]["final_mAP"] for c in CLIENTS])
    return v, float(v.mean())


def m2(runs, mode):
    """Pooled own-weather mAP, per client + mean."""
    if mode == "joint":
        v = np.array([runs["joint"]["per_client_pooled_mAP"][c] for c in CLIENTS])
    else:
        diag = runs[mode]["cross_client"]["diagonal"]
        v = np.array([diag[c][MAP_KEY] for c in CLIENTS])
    return v, float(v.mean())


def full_matrix(runs, mode, key=MAP_KEY):
    """
    5x5 [model i, test set j]. The JSON stores off-diagonals in
    cross_client.matrix and the diagonal separately; stitch them back together.
    """
    cc = runs[mode]["cross_client"]
    M = np.zeros((5, 5))
    for i, mi in enumerate(CLIENTS):
        for j, tj in enumerate(CLIENTS):
            M[i, j] = cc["diagonal"][mi][key] if i == j else cc["matrix"][mi][tj][key]
    return M


def diag_metric(runs, mode, key):
    d = runs[mode]["cross_client"]["diagonal"]
    return np.array([d[c][key] for c in CLIENTS])


# ------------------------------------------------- helpers for bar charts --

def grouped_bars(ax, groups, series, values, colors, labels,
                 fmt="{:.3f}", label_rot=0, gap_frac=0.14, value_fs=7.5):
    """
    values[s][g]. A 2px-equivalent surface gap between adjacent bars, a direct
    value label on every bar (the contrast-relief rule for the aqua slot), and
    no per-bar outline. Labels sit outside the bar on both signs, and rotate to
    90 degrees when the groups are too tight to take them flat.
    """
    n = len(series)
    slot = 1.0 / (n + 0.9)
    w = slot * (1 - gap_frac)
    x = np.arange(len(groups), dtype=float)
    for s_i, s in enumerate(series):
        off = (s_i - (n - 1) / 2) * slot
        ax.bar(x + off, values[s_i], width=w, color=colors[s_i],
               label=labels[s_i], linewidth=0, zorder=3)
        for g_i, v in enumerate(values[s_i]):
            up = v >= 0
            ax.annotate(fmt.format(v), (x[g_i] + off, v),
                        ha="center" if label_rot else "center",
                        va="bottom" if up else "top",
                        fontsize=value_fs, color=INK_2, rotation=label_rot,
                        xytext=(0, 3 if up else -3), textcoords="offset points",
                        zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    return x


def headroom(ax, values, frac=0.30):
    """Leave room above the tallest bar for rotated labels and the legend."""
    top = max(np.max(v) for v in values)
    ax.set_ylim(0, top * (1 + frac))


def savefig(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, OBJ_DET)}")


# ================================================================ SLIDE 1 ==

def fig01_run_design(runs):
    """What each of the four runs actually switched on. Table, not a chart."""
    rows = [
        # mode,        FL/cross-client,  gossip mixing,  replay,   task-sequential
        ("gossip",     "Yes",  "Yes",  "Yes",  "Yes"),
        ("local_only", "No",   "No",   "Yes",  "Yes"),
        ("sequential", "No",   "No",   "No",   "Yes"),
        ("joint",      "No",   "No",   "n/a",  "No"),
    ]
    isolates = [
        "the full method",
        "replay alone - the baseline gossip must beat",
        "naive CL - what forgetting looks like unprotected",
        "ceiling: all 17k images pooled & shuffled, one model",
    ]
    headers = ["Run", "FL\n(cross-client)", "Gossip\nmixing", "Replay\nbuffer",
               "Continual\n(3 tasks)", "What it isolates"]

    fig, ax = plt.subplots(figsize=(13.2, 2.9))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.02)
    ax.set_axis_off()
    colx = [0.012, 0.175, 0.285, 0.385, 0.485, 0.590]
    colw = [0.16, 0.10, 0.09, 0.09, 0.10, 0.40]

    nrow = len(rows)
    rowh = 1.0 / (nrow + 1.6)
    y0 = 1.0 - 1.05 * rowh
    for k, h in enumerate(headers):
        ax.text(colx[k] + (colw[k] / 2 if 0 < k < 5 else 0), y0 + 0.55 * rowh, h,
                ha="center" if 0 < k < 5 else "left", va="bottom",
                fontsize=9.5, color=INK_2, weight="bold", linespacing=1.35)
    ax.plot([0, 1], [y0 + 0.42 * rowh] * 2, color=INK_2, lw=1.0)

    for r, (mode, fl, gos, rep, seq) in enumerate(rows):
        y = y0 - (r + 0.5) * rowh
        ax.add_patch(Rectangle((0, y - 0.46 * rowh), 1, 0.92 * rowh,
                               facecolor="#f7f7f5" if r % 2 else SURFACE,
                               edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((0.0, y - 0.30 * rowh), 0.007, 0.60 * rowh,
                               facecolor=MODE_COLOR[mode], edgecolor="none", zorder=2))
        ax.text(colx[0], y, MODE_SHORT[mode], ha="left", va="center",
                fontsize=11, color=INK, weight="bold")
        for k, val in enumerate([fl, gos, rep, seq], start=1):
            yes = val == "Yes"
            ax.text(colx[k] + colw[k] / 2, y, "✓" if yes else "–",
                    ha="center", va="center",
                    fontsize=13 if yes else 12, color=INK if yes else INK_MUTED,
                    weight="bold" if yes else "normal")
            if val == "n/a":
                ax.text(colx[k] + colw[k] / 2, y - 0.30 * rowh, "n/a", ha="center",
                        va="center", fontsize=7.5, color=INK_MUTED)
        ax.text(colx[5], y, isolates[r], ha="left", va="center",
                fontsize=9.5, color=INK_2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    caption(fig,
            "Source: config.py MODES. joint's replay is — not ✗: run_joint() pools all 15 (weather x time-of-day) cells into one\n"
            "shuffled dataset with no task boundaries, so there is no past task to replay - the shuffle already does replay's job.\n"
            "FL here IS gossip: there is no server and no FedAvg, so the two columns move together by construction.")
    savefig(fig, "fig01_run_design.png")


def fig02_partition(counts):
    """Backup slide: why several numbers later on are fragile."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.8))
    for ax, split, cmap_hi in zip(axes, ("train", "test"), (None, None)):
        M = counts[split]
        im = ax.imshow(np.log10(M + 1), cmap=SEQ_CMAP, aspect="auto",
                       vmin=0, vmax=np.log10(counts["train"].max() + 1))
        for i in range(5):
            for j in range(3):
                v = int(M[i, j])
                dark = np.log10(v + 1) > 0.62 * np.log10(counts["train"].max() + 1)
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if dark else INK,
                        weight="bold" if v < 100 else "normal")
                if v < 100:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#e34948", lw=2.0, zorder=5))
        ax.set_xticks(range(3)); ax.set_xticklabels(TASKS)
        ax.set_yticks(range(5)); ax.set_yticklabels(CLIENTS)
        ax.set_title(f"{split} images per cell   (total {int(M.sum()):,})", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    caption(fig,
            "Rows = clients (weather), columns = tasks (time of day). Red outline = cell under 100 images.\n"
            "clear holds 61% of all training images; partly cloudy/night is 44 train + 10 test. Those two night cells are\n"
            "9 and 15 optimiser steps of 'task', and their test mAP is computed on 10 and 18 images - see the caveats on slide 2.")
    savefig(fig, "fig02_partition.png")


def fig02b_budget(runs):
    """
    The runs are NOT step-matched, and it is not a small difference.

    Without a replay buffer `use_replay` is false, so `cur_bs` stays at
    BATCH_SIZE=32 instead of dropping to 16 for tasks 1-2 (train.py:146). Same
    images, half as many optimiser steps. `sequential` therefore trains on 34%
    fewer steps than `local_only`, which is exactly the pair you would want to
    read as "what did replay buy".
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    groups = [c.replace(" ", "\n") for c in CLIENTS]

    ax = axes[0]
    vals = [np.array([runs[m]["local_steps"][c] for c in CLIENTS], dtype=float)
            for m in CL_MODES]
    grouped_bars(ax, groups, CL_MODES, vals, [MODE_COLOR[m] for m in CL_MODES],
                 [MODE_SHORT[m] for m in CL_MODES], fmt="{:,.0f}", value_fs=7.5,
                 label_rot=90)
    tidy(ax)
    headroom(ax, vals, frac=0.34)
    ax.set_ylabel("optimiser steps")
    ax.set_title("Optimiser steps per client", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    totals = [float(sum(runs[m]["local_steps"].values())) for m in CL_MODES] + \
             [float(runs["joint"]["steps"])]
    modes4 = CL_MODES + ["joint"]
    for i, (m, t) in enumerate(zip(modes4, totals)):
        ax.barh(len(modes4) - 1 - i, t, height=0.34, color=MODE_COLOR[m],
                linewidth=0, zorder=3)
        ax.annotate(f"{t:,.0f}", (t, len(modes4) - 1 - i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK_2)
    tidy(ax, axis="x")
    ax.set_yticks(range(len(modes4)))
    ax.set_yticklabels([MODE_SHORT[m] for m in reversed(modes4)])
    ax.set_xlim(0, max(totals) * 1.22)
    ax.set_xlabel("total optimiser steps in the run")
    ax.set_title("Total training budget", loc="left")

    lo, sq = totals[1], totals[2]
    caption(fig,
            f"local-only and gossip are step-matched ({lo:,.0f} each). `sequential` is not - it runs {sq:,.0f}, "
            f"{100*(1-sq/lo):.0f}% fewer.\nWith no buffer, `use_replay` is false, so cur_bs stays at BATCH_SIZE=32 instead of "
            "dropping to 16 for tasks 1-2 (train.py:146):\nsame images, half the optimiser steps. So "
            f"'sequential {m1(runs, 'sequential')[1]:.4f} ≈ local-only {m1(runs, 'local_only')[1]:.4f}' is NOT a controlled "
            "read of what replay bought -\nsequential matched it on a third fewer steps. Only the gossip-vs-local-only "
            "comparison is step-matched in this batch.\n"
            "(joint's 5,283 is one model over the pooled set, not a sum over five clients - it is on this axis for scale only.)")
    savefig(fig, "fig02b_training_budget.png")


# ================================================================ SLIDE 2 ==

def fig03_headline(runs):
    """The headline comparison, split by metric because they are not the same."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3))
    groups = [c.replace(" ", "\n") for c in CLIENTS] + ["MEAN"]

    for ax, modes, getter, title, sub in (
        (axes[0], CL_MODES, m1,
         "M1  task-averaged final mAP@[.5:.95]",
         "mean of the 3 per-task cell scores  ·  joint has no tasks, so it cannot appear here"),
        (axes[1], POOLED_MODES, m2,
         "M2  pooled own-weather mAP@[.5:.95]",
         "the 3 cells pooled into one loader, image-weighted  ·  sequential was never cross-evaluated"),
    ):
        vals, means = [], []
        for mode in modes:
            v, mu = getter(runs, mode)
            vals.append(np.append(v, mu))
            means.append(mu)
        grouped_bars(ax, groups, modes, vals,
                     [MODE_COLOR[m] for m in modes],
                     [f"{MODE_SHORT[m]}  ({mu:.4f})" for m, mu in zip(modes, means)],
                     fmt="{:.3f}", value_fs=7.2, label_rot=90)
        tidy(ax)
        ax.set_ylim(0, 0.335)
        ax.set_ylabel("mAP@[.5:.95]")
        ax.set_title(title, loc="left", pad=24)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.6, color=INK_MUTED)
        ax.axvline(4.5, color=GRID, lw=1.2, zorder=1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)

    a1, a2, a3 = (m1(runs, m)[1] for m in CL_MODES)
    b1, b2, b3 = (m2(runs, m)[1] for m in POOLED_MODES)
    caption(fig,
            "Left and right are DIFFERENT numbers for the same runs - M1 macro-averages tasks (a 10-image cell counts as much as a\n"
            f"2,843-image one), M2 micro-averages images. Never quote joint's {b3:.4f} against sequential's {a3:.4f}.\n"
            f"Valid readings:  M1  local-only {a2:.4f} ≈ sequential {a3:.4f} > gossip {a1:.4f}   |   "
            f"M2  joint {b3:.4f} > local-only {b2:.4f} > gossip {b1:.4f}.")
    savefig(fig, "fig03_headline_map.png")


def fig04_detection_metrics(runs):
    """mAP is one operating point; mAP50 and mAR@100 say where the ceiling is."""
    keys = [(MAP_KEY, "mAP@[.5:.95]"), ("mAP50", "mAP@0.50"), ("mAR@100", "mAR@100")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    modes = ["gossip", "local_only"]
    groups = [c.replace(" ", "\n") for c in CLIENTS] + ["MEAN"]

    for ax, (key, title) in zip(axes, keys):
        vals = []
        for mode in modes:
            v = diag_metric(runs, mode, key)
            vals.append(np.append(v, v.mean()))
        grouped_bars(ax, groups, modes, vals,
                     [MODE_COLOR[m] for m in modes],
                     [MODE_SHORT[m] for m in modes], fmt="{:.3f}", value_fs=7.0,
                     label_rot=90)
        tidy(ax)
        ax.set_ylim(0, 0.66)
        ax.set_title(title, loc="left")
        ax.axvline(4.5, color=GRID, lw=1.2, zorder=1)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="upper left")
    axes[2].axhline(1 / 3, color="#e34948", lw=1.2, zorder=2)
    axes[2].text(-0.45, 0.47, "red line = 1/3\nthe frozen RPN's recall ceiling",
                 ha="left", va="bottom", fontsize=8.5, color="#e34948",
                 linespacing=1.4)
    caption(fig,
            "Metric M2 (pooled own-weather), the only split where mAP50 and mAR@100 were recorded. mAP50 ≈ 0.44-0.51 against\n"
            "mAR@100 ≈ 0.30-0.33 says the head localises what it is shown but the frozen COCO RPN proposes only about a third of\n"
            "BDD100K's boxes (55.5% of them are under 32px). No training scheme can cross that - it is the same ceiling in all four runs.")
    savefig(fig, "fig04_detection_metrics.png")


def fig05_per_task(runs):
    """Where the mAP actually goes: night is the hard task, in every mode."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    vals = []
    for mode in CL_MODES:
        h = runs[mode]["acc_history"]
        # final score on each task = last entry of that task's history
        per_task = np.array([[h[c][str(t)][-1] for t in range(3)] for c in CLIENTS])
        vals.append(per_task.mean(axis=0))
    grouped_bars(ax, [f"task {i+1}\n{t}" for i, t in enumerate(TASKS)],
                 CL_MODES, vals, [MODE_COLOR[m] for m in CL_MODES],
                 [MODE_SHORT[m] for m in CL_MODES], fmt="{:.3f}", value_fs=8)
    tidy(ax)
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("final mAP@[.5:.95], mean over 5 clients")
    ax.set_title("Final mAP per task - the night gap, not the method, sets M1", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)

    ax = axes[1]
    h = runs["local_only"]["acc_history"]
    night = np.array([h[c]["1"][-1] for c in CLIENTS])
    counts = partition_counts()
    n_test = counts["test"][:, 1]
    n_train = counts["train"][:, 1]
    order = np.argsort(night)                       # ascending -> best on top
    ys = np.arange(5)
    for y, i in zip(ys, order):
        thin = n_test[i] < 50
        ax.barh(y, night[i], height=0.46, linewidth=0,
                color="#e34948" if thin else MODE_COLOR["local_only"], zorder=3)
        ax.annotate(f"{night[i]:.3f}", (night[i], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_2, zorder=4)
        tag = (f"⚠ {int(n_train[i])} train / {int(n_test[i])} test images"
               if thin else f"{int(n_train[i]):,} train / {int(n_test[i]):,} test images")
        ax.annotate(tag, (0.004, y), va="center", fontsize=8.2, zorder=5,
                    color="#ffffff", weight="bold" if thin else "normal")
    tidy(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels([CLIENTS[i] for i in order])
    ax.set_xlim(0, 0.335)
    ax.set_xlabel("night (task 2) mAP@[.5:.95], local-only run")
    ax.set_title("...and the top two night scores sit on 18 and 10 test images",
                 loc="left")

    night_means = [v[1] for v in vals]
    day_means = [v[0] for v in vals]
    caption(fig,
            f"Left: task 2 (night) averages {min(night_means):.3f}-{max(night_means):.3f} against "
            f"{min(day_means):.3f}-{max(day_means):.3f} daytime in every mode - the modes are separated by far less\n"
            "than the tasks are, so M1 mostly reports how much night is in the mix. Right: overcast and partly cloudy hold the two\n"
            "best night scores and are measured on 18 and 10 test images (red). evaluate_detection.py's CLASS_FLOOR=50 exists to catch\n"
            "exactly this but only filters classes with N==0, so both cells pass through into M1 and into BWT. Noise, not results.")
    savefig(fig, "fig05_per_task.png")


def fig06_retention(runs):
    """The continual-learning plot: task accuracy re-measured after each later task."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), sharey=True)
    for t, ax in enumerate(axes):
        npts = 3 - t
        xs = np.arange(t, 3)
        ends = []
        for mode in CL_MODES:
            h = runs[mode]["acc_history"]
            series = np.array([h[c][str(t)] for c in CLIENTS])   # 5 x npts
            for row in series:
                ax.plot(xs, row, color=MODE_COLOR[mode], lw=1.0, alpha=0.22, zorder=2)
            mu = series.mean(axis=0)
            ax.plot(xs, mu, color=MODE_COLOR[mode], lw=2.0, marker="o", ms=6,
                    mec=SURFACE, mew=1.5, label=MODE_SHORT[mode], zorder=4)
            ends.append([mu[-1], MODE_COLOR[mode]])
        # de-overlap the endpoint labels: nothing closer than 0.006 mAP apart
        ends.sort()
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.0065)
        for (y_lab, col), mode in zip(ends, sorted(
                CL_MODES, key=lambda m: np.mean([runs[m]["acc_history"][c][str(t)][-1]
                                                 for c in CLIENTS]))):
            y_true = np.mean([runs[mode]["acc_history"][c][str(t)][-1] for c in CLIENTS])
            ax.annotate(f"{y_true:.3f}", (xs[-1], y_lab), fontsize=8.5, color=col,
                        xytext=(8, 0), textcoords="offset points", va="center")
        tidy(ax)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"after\ntask {i+1}" for i in range(3)])
        ax.set_xlim(-0.35, 2.75)
        ax.set_title(f"task {t+1} · {TASKS[t]}", loc="left")
        if npts == 1:
            ax.text(0.5, 0.5, "trained last -\nnothing after it", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=INK_MUTED)
    axes[0].set_ylabel("mAP@[.5:.95] on that task's test cell")
    axes[0].set_ylim(0.13, 0.30)
    axes[0].legend(loc="lower left")
    caption(fig,
            "Thin lines = the 5 clients, thick = their mean. A forgetting curve should slope DOWN to the right. None of these do -\n"
            "task 1 ends at or above where it started in every mode, including sequential, which has no replay buffer at all.\n"
            "Only roi_heads.box_predictor is trainable and all 3 tasks share the same 10 classes over the same frozen features, so\n"
            "'ROI feature -> class + box' is the same function for day, night and dusk. There is no task-specific mapping to overwrite.")
    savefig(fig, "fig06_retention.png")


def fig07_forgetting_bwt(runs):
    """Forgetting and BWT are 1e-3. Show that against the scale of the score."""
    fig = plt.figure(figsize=(13.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.72], wspace=0.30)
    ax_f, ax_b, ax_s = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                        fig.add_subplot(gs[2]))

    ys = np.arange(5)[::-1]
    for ax, key, title in ((ax_f, "forgetting", "Forgetting  (higher = worse)"),
                           (ax_b, "bwt", "BWT  (higher = better)")):
        for m_i, mode in enumerate(CL_MODES):
            pc = runs[mode]["per_client"]
            v = np.array([pc[c][key] for c in CLIENTS])
            off = (m_i - 1) * 0.20
            ax.scatter(v, ys + off, s=55, color=MODE_COLOR[mode], zorder=4,
                       linewidth=1.4, edgecolor=SURFACE, label=MODE_SHORT[mode])
        ax.axvline(0, color=INK_2, lw=1.0, zorder=3)
        for y in ys:
            ax.axhline(y, color=GRID, lw=0.8, zorder=1)
        tidy(ax, axis="x", grid=False)
        ax.set_yticks(ys)
        ax.set_yticklabels(CLIENTS)
        ax.set_xlim(-0.0075, 0.0075)
        ax.set_xticks([-0.005, 0, 0.005])
        ax.set_title(title, loc="left")
        ax.set_xlabel("mAP@[.5:.95]")
    ax_f.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02))

    # scale reference: the same axis, but zoomed out to the size of the score
    means = [m1(runs, m)[1] for m in CL_MODES]
    mu = float(np.mean(means))
    allvals = np.abs([runs[m]["per_client"][c][k]
                      for m in CL_MODES for c in CLIENTS for k in ("forgetting", "bwt")])
    ax_s.barh([1.0], [mu], height=0.20, color="#c9c7c0", linewidth=0, zorder=3)
    ax_s.barh([0.62], [allvals.max()], height=0.20, color="#e34948", linewidth=0, zorder=3)
    ax_s.annotate(f"mean final mAP   {mu:.4f}", (mu, 1.0), xytext=(8, 0),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=9, color=INK)
    ax_s.annotate(f"largest |forgetting| or |BWT|\nanywhere in any run   {allvals.max():.4f}\n"
                  f"= {100*allvals.max()/mu:.1f}% of the score",
                  (allvals.max(), 0.62), xytext=(10, 0), textcoords="offset points",
                  ha="left", va="center", fontsize=9, color="#e34948", linespacing=1.45)
    ax_s.set_xlim(0, 0.30)
    ax_s.set_ylim(0.30, 1.35)
    ax_s.set_yticks([])
    tidy(ax_s, axis="x", grid=False)
    ax_s.set_title("The same axis, to scale", loc="left")

    seq_bwt = runs["sequential"]["system"]["bwt"]["mean"]
    gos_bwt = runs["gossip"]["system"]["bwt"]["mean"]
    caption(fig,
            f"Every forgetting and BWT value in every mode is under {allvals.max():.4f} mAP - at most "
            f"{100*allvals.max()/mu:.1f}% of the score itself, and within the run-to-run\n"
            f"noise of a 10-to-18-image test cell. Sequential (no replay, no gossip) reports BWT {seq_bwt:+.4f} and gossip "
            f"reports {gos_bwt:+.4f}: the two are\nindistinguishable, so this number is not measuring backward transfer. It is measuring "
            "'a few thousand more SGD steps on a shared\nhead nudge everything up slightly'. Do not claim replay or gossip prevented forgetting - nothing here forgot.")
    savefig(fig, "fig07_forgetting_bwt.png")


# ================================================================ SLIDE 3 ==

def fig08_cross_matrix(runs):
    """5x5 model-vs-test-set, gossip and local_only on one shared scale."""
    Mg, Ml = full_matrix(runs, "gossip"), full_matrix(runs, "local_only")
    vmin, vmax = min(Mg.min(), Ml.min()), max(Mg.max(), Ml.max())

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    for ax, M, mode in ((axes[0], Mg, "gossip"), (axes[1], Ml, "local_only")):
        im = ax.imshow(M, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(5):
            for j in range(5):
                frac = (M[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=9.5,
                        color="#ffffff" if frac > 0.58 else INK,
                        weight="bold" if i == j else "normal")
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=MODE_COLOR[mode], lw=2.2, zorder=5))
        ax.set_xticks(range(5)); ax.set_xticklabels(CLIENTS, rotation=18, ha="right")
        ax.set_yticks(range(5)); ax.set_yticklabels(CLIENTS)
        ax.set_xlabel("evaluated on client's pooled test set")
        if mode == "gossip":
            ax.set_ylabel("client's final model")
        ax.set_title(f"{MODE_SHORT[mode]}   ·   global cross-client mAP "
                     f"{runs[mode]['cross_client']['global_cross_mAP']:.4f}", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02, label="mAP@[.5:.95]")
    caption(fig,
            "Outlined cell = the model on its own weather. Read DOWN a column: in gossip every model scores the same on a given test\n"
            "set to 3 decimals - the five clients have become one model. In local-only the columns spread, and clear's model (11x the\n"
            "data) is the best model on all five test sets including the other four clients' own data.\n"
            "Gossip's consensus lands at 0.2009 on clear's test set - the level of local-only's WORST client (0.1974), not its best (0.2253).")
    savefig(fig, "fig08_cross_matrix.png")


def fig09_matrix_structure(runs):
    """Separate the two things the 5x5 is made of: which model, and which test set."""
    Mg, Ml = full_matrix(runs, "gossip"), full_matrix(runs, "local_only")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))

    ax = axes[0]
    vals = [Mg.std(axis=0), Ml.std(axis=0)]
    grouped_bars(ax, [c.replace(" ", "\n") for c in CLIENTS],
                 ["gossip", "local_only"], vals,
                 [MODE_COLOR["gossip"], MODE_COLOR["local_only"]],
                 [f"gossip  (mean {vals[0].mean():.5f})",
                  f"local-only  (mean {vals[1].mean():.5f})"],
                 fmt="{:.4f}", value_fs=7.5)
    tidy(ax)
    headroom(ax, vals, frac=0.42)
    ax.set_ylabel("std across the 5 models")
    ax.set_xlabel("on this test set")
    ax.set_title("Disagreement between clients  —  gossip has none left", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    means = [Mg.mean(axis=0), Ml.mean(axis=0)]
    grouped_bars(ax, [c.replace(" ", "\n") for c in CLIENTS],
                 ["gossip", "local_only"], means,
                 [MODE_COLOR["gossip"], MODE_COLOR["local_only"]],
                 ["gossip", "local-only"], fmt="{:.3f}", value_fs=7.5)
    tidy(ax)
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("mean over the 5 models")
    ax.set_xlabel("on this test set")
    ax.set_title("Test-set difficulty  —  same shape in both runs", loc="left")
    ax.legend(loc="upper center", ncol=2)

    sg, sl = vals[0].mean(), vals[1].mean()
    caption(fig,
            f"Left is the evidence gossip did what it claims: the spread between the 5 clients' models collapses from "
            f"{sl:.4f} to {sg:.4f},\na {sl/sg:.1f}x reduction - by the end of the run the five clients are effectively one model.\n"
            f"Right is what dominates the raw matrix and is NOT a property of any method: clear's pooled test set scores "
            f"~{means[0][0]:.2f} and\novercast's ~{means[0][1]:.2f} whichever model is looking at it. Any per-client 'own vs other' "
            "number is mostly reading this.")
    savefig(fig, "fig09_matrix_structure.png")


def fig10_own_vs_other(runs):
    """Why the headline own-vs-other gap is the wrong summary."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    vals = []
    for mode in ("gossip", "local_only"):
        pc = runs[mode]["cross_client"]["per_client"]
        vals.append(np.array([pc[c]["own_vs_other_gap"] for c in CLIENTS]))
    grouped_bars(ax, [c.replace(" ", "\n") for c in CLIENTS],
                 ["gossip", "local_only"], vals,
                 [MODE_COLOR["gossip"], MODE_COLOR["local_only"]],
                 ["gossip", "local-only"], fmt="{:+.3f}", value_fs=7.5)
    tidy(ax)
    ax.axhline(0, color=INK_2, lw=1.0, zorder=3)
    ax.set_ylim(-0.042, 0.042)
    ax.set_ylabel("own mAP − mean(other) mAP")
    ax.set_title("Per-client own-vs-other gap  —  same ±2-3% in both runs", loc="left")
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(-0.02, 0.0))

    ax = axes[1]
    sg = [v.mean() for v in vals]
    ab = [np.abs(v).mean() for v in vals]
    grouped_bars(ax, ["signed mean\n(what the JSON reports)", "mean absolute\n(what it should report)"],
                 ["gossip", "local_only"], [[sg[0], ab[0]], [sg[1], ab[1]]],
                 [MODE_COLOR["gossip"], MODE_COLOR["local_only"]],
                 ["gossip", "local-only"], fmt="{:.4f}", value_fs=8.5)
    tidy(ax)
    ax.set_ylim(0, 0.028)
    ax.set_title("Summarising those five numbers", loc="left")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(-0.02, 1.0))

    caption(fig,
            f"global_own_vs_other_gap says gossip {sg[0]:.4f} vs local-only {sg[1]:.4f} and invites 'gossip homogenised the clients'. "
            "It is a\nsigned-mean cancellation artefact: the per-client gaps are ±2-3% in BOTH runs, and the mean ABSOLUTE gap is "
            f"{ab[0]:.4f} vs\n{ab[1]:.4f} - indistinguishable. What the sign actually tracks is whether your own pooled test set is harder "
            "than average\n(clear and rainy negative, overcast and partly cloudy positive), which is a property of the partition, not of gossip.\n"
            "Report mean |gap| alongside the signed mean, and read homogenisation off the column spread in the previous figure instead.")
    savefig(fig, "fig10_own_vs_other.png")


# ------------------------------------------------------------ the numbers --

def write_tables(runs, counts):
    L = []
    a = L.append
    a("# Summary tables - seed 42, first successful batch\n")
    a("Generated by `analyse_runs.py`. Every number is read straight from "
      "`results/runs/*_seed42.json`.\n")

    a("\n## 1. What each run switched on\n")
    a("| Run | FL (cross-client) | Gossip mixing | Replay buffer | Continual (3 tasks) | What it isolates |")
    a("|---|---|---|---|---|---|")
    a("| **gossip** | yes | yes | yes | yes | the full method |")
    a("| **local_only** | no | no | yes | yes | replay alone - the baseline gossip must beat |")
    a("| **sequential** | no | no | no | yes | naive CL - what forgetting looks like unprotected |")
    a("| **joint** | no | no | n/a | no | ceiling: all data pooled and shuffled, one model |")
    a("\n`joint` does **not** use replay (`config.py` `MODES['joint'] = {'gossip': False, 'replay': False}`). "
      "`run_joint()` pools every cell into one shuffled dataset with no task boundaries, so there is no past task "
      "to replay - the shuffle already gives i.i.d. exposure to all of it.\n")
    a("\nIn this system FL *is* gossip - there is no server and no FedAvg - so those two columns are never independent.\n")

    a("\n## 2. Training volume actually spent\n")
    a("| Run | optimiser steps | gossip ops | notes |")
    a("|---|---|---|---|")
    for m in CL_MODES:
        st = runs[m]["local_steps"]
        go = runs[m]["gossip_ops"]
        a(f"| {m} | {sum(st.values()):,} over 5 clients | {sum(go.values())} | "
          f"{runs[m]['scheduler_slices']} scheduler slices |")
    a(f"| joint | {runs['joint']['steps']:,} (single model) | 0 | no clients, no tasks |")
    a("\n**The runs are not all step-matched.** gossip and local_only are (8,043 each); "
      "`sequential` runs 5,304, 34% fewer. Without a buffer `use_replay` is false, so `cur_bs` "
      "stays at `BATCH_SIZE=32` instead of dropping to 16 for tasks 1-2 (`train.py:146`) - same "
      "images, half the optimiser steps. Treat any sequential-vs-local_only difference as "
      "uncontrolled.\n")
    a("| client | train imgs | gossip steps | local_only steps | sequential steps | gossip ops |")
    a("|---|---|---|---|---|---|")
    for i, c in enumerate(CLIENTS):
        a(f"| {c} | {int(counts['train'][i].sum()):,} | "
          + " | ".join(f"{runs[m]['local_steps'][c]:,}" for m in CL_MODES)
          + f" | {runs['gossip']['gossip_ops'][c]} |")
    a("| **total** | **56,347** | "
      + " | ".join(f"**{sum(runs[m]['local_steps'].values()):,}**" for m in CL_MODES)
      + f" | **{sum(runs['gossip']['gossip_ops'].values())}** |")

    a("\n## 3. The two metrics\n")
    a("| | M1 task-averaged final mAP | M2 pooled own-weather mAP |")
    a("|---|---|---|")
    a("| how | mean of the 3 per-task cell scores | the 3 cells pooled into one loader |")
    a("| weighting | per task (a 10-image cell = a 2,843-image cell) | per image |")
    a("| json | `per_client.final_mAP` / `system.mAP.mean` | `cross_client.diagonal[c]` / `per_client_pooled_mAP` |")
    a("| available | gossip, local_only, sequential | gossip, local_only, joint |")
    a("\n### M1 - task-averaged final mAP@[.5:.95]\n")
    a("| client | " + " | ".join(CL_MODES) + " |")
    a("|---" * (len(CL_MODES) + 1) + "|")
    for i, c in enumerate(CLIENTS):
        a(f"| {c} | " + " | ".join(f"{m1(runs, m)[0][i]:.4f}" for m in CL_MODES) + " |")
    a("| **mean** | " + " | ".join(f"**{m1(runs, m)[1]:.4f}**" for m in CL_MODES) + " |")
    a("| std | " + " | ".join(f"{m1(runs, m)[0].std():.4f}" for m in CL_MODES) + " |")

    a("\n### M2 - pooled own-weather mAP@[.5:.95]\n")
    a("| client | " + " | ".join(POOLED_MODES) + " |")
    a("|---" * (len(POOLED_MODES) + 1) + "|")
    for i, c in enumerate(CLIENTS):
        a(f"| {c} | " + " | ".join(f"{m2(runs, m)[0][i]:.4f}" for m in POOLED_MODES) + " |")
    a("| **mean** | " + " | ".join(f"**{m2(runs, m)[1]:.4f}**" for m in POOLED_MODES) + " |")
    gm, lm, jm = (m2(runs, m)[1] for m in POOLED_MODES)
    a(f"\n`joint` beats `local_only` by {jm - lm:+.4f} mAP ({100*(jm-lm)/lm:+.1f}%). "
      f"`gossip` trails `local_only` by {gm - lm:+.4f} ({100*(gm-lm)/lm:+.1f}%).\n")

    a("\n### mAP50 and mAR@100 (M2 only - recorded for gossip and local_only)\n")
    a("| metric | gossip | local_only |")
    a("|---|---|---|")
    for key in (MAP_KEY, "mAP50", "mAR@100"):
        a(f"| {key} | {diag_metric(runs, 'gossip', key).mean():.4f} "
          f"| {diag_metric(runs, 'local_only', key).mean():.4f} |")

    a("\n## 4. Forgetting and BWT\n")
    a("| client | " + " | ".join(f"{m} F | {m} BWT" for m in CL_MODES) + " |")
    a("|---" * (2 * len(CL_MODES) + 1) + "|")
    for c in CLIENTS:
        cells = []
        for m in CL_MODES:
            pc = runs[m]["per_client"][c]
            cells += [f"{pc['forgetting']:.5f}", f"{pc['bwt']:+.5f}"]
        a(f"| {c} | " + " | ".join(cells) + " |")
    cells = []
    for m in CL_MODES:
        s = runs[m]["system"]
        cells += [f"**{s['forgetting']['mean']:.5f}**", f"**{s['bwt']['mean']:+.5f}**"]
    a("| **mean** | " + " | ".join(cells) + " |")

    a("\n## 5. Cross-client structure\n")
    for mode in ("gossip", "local_only"):
        M = full_matrix(runs, mode)
        a(f"\n### {mode} - 5x5 mAP@[.5:.95] (row = model, col = test set)\n")
        a("| model \\ test | " + " | ".join(CLIENTS) + " |")
        a("|---" * 6 + "|")
        for i, c in enumerate(CLIENTS):
            a(f"| **{c}** | " + " | ".join(f"{M[i, j]:.4f}" for j in range(5)) + " |")
        a(f"| *spread across models* | " + " | ".join(f"*{M.std(axis=0)[j]:.4f}*" for j in range(5)) + " |")
        a(f"| *test-set difficulty* | " + " | ".join(f"*{M.mean(axis=0)[j]:.4f}*" for j in range(5)) + " |")

    a("\n### Own-vs-other gap\n")
    a("| | " + " | ".join(CLIENTS) + " | signed mean | **mean abs** |")
    a("|---" * 8 + "|")
    for mode in ("gossip", "local_only"):
        pc = runs[mode]["cross_client"]["per_client"]
        v = np.array([pc[c]["own_vs_other_gap"] for c in CLIENTS])
        a(f"| {mode} | " + " | ".join(f"{x:+.4f}" for x in v) +
          f" | {v.mean():+.5f} | **{np.abs(v).mean():.4f}** |")

    a("\n## 6. Caveats that must appear on the slides\n")
    a("1. **Two of the reported cells are noise.** `overcast`/night is 72 train + 18 test images; "
      "`partly cloudy`/night is 44 train + 10 test. Those are 15 and 9 optimiser steps of 'task'. "
      "`CLASS_FLOOR = 50` in `evaluate_detection.py` exists to catch this but only drops classes with "
      "`N == 0`, so both cells feed straight into M1 and into BWT.")
    a("2. **M1 and M2 are not interchangeable.** joint 0.2460 vs sequential 0.2306 is not a comparison.")
    a("2b. **Only gossip vs local_only is step-matched.** `sequential` trains 5,304 steps against "
      "their 8,043 (34% fewer) because no buffer means `cur_bs` stays at 32 rather than 16. "
      "`joint` is a different code path entirely. So the one clean A/B in this batch is "
      "gossip vs local_only - which gossip loses.")
    a("3. **Nothing here shows forgetting.** Sequential has no replay buffer and still reports "
      "forgetting 0.0006 and BWT +0.0025, within noise of gossip's 0.0006 / +0.0028. Only "
      "`roi_heads.box_predictor` is trainable and the 3 tasks share the same 10 classes over the same "
      "frozen features, so there is no task-specific function to overwrite.")
    a("4. **mAR@100 caps at ~1/3 in every run** while mAP50 is 0.44-0.51: the frozen COCO RPN does not "
      "propose BDD100K's small objects. That ceiling is the same in all four runs and caps how far apart "
      "any two of them can land.")
    a("5. **The gate was not recorded in this batch.** `_gossip()` discarded `gossip_round()`'s return "
      "value, so these four runs contain zero evidence of what the gate did. Instrumentation landed after "
      "them; the seed-7 re-run measures alpha_mean 0.4882 against lambda 0.5, i.e. near-plain averaging.")

    path = os.path.join(HERE, "summary_tables.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {os.path.relpath(path, OBJ_DET)}")


def main():
    apply_style()
    os.makedirs(FIGS, exist_ok=True)
    runs = load_runs()
    counts = partition_counts()

    print("slide 1 - run design")
    fig01_run_design(runs)
    fig02_partition(counts)
    fig02b_budget(runs)

    print("slide 2 - metrics")
    fig03_headline(runs)
    fig04_detection_metrics(runs)
    fig05_per_task(runs)
    fig06_retention(runs)
    fig07_forgetting_bwt(runs)

    print("slide 3 - cross-client")
    fig08_cross_matrix(runs)
    fig09_matrix_structure(runs)
    fig10_own_vs_other(runs)

    print("tables")
    write_tables(runs, counts)


if __name__ == "__main__":
    main()
