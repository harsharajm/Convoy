"""
Shared loading + plotting for the v2/v3 results report. Used by
v2_plots/generate_v2_plots.py and v3_plots/generate_v3_plots.py.

Aggregation rule for CL modes (local_only, sequential, gossip): a client's
"final" per-class metrics are the unweighted mean, across its 3 tasks, of
that task's LAST evaluation snapshot (class_history[client][task][-1]) - the
snapshot taken after all training finished. This exactly reproduces the
pipeline's own `final_mAP` (verified: mean over tasks of acc_history equals
per_client['final_mAP'] to float precision), so the same rule is applied here
to every other per-class field for consistency. Joint has no task axis - its
per-class numbers are already pooled and used directly.

Core classes (macro-averaging set) match evaluate_detection.py's FROZEN_CORE:
car, traffic sign, traffic light, person - the fixed 4-class set every cell
can be macro-averaged over, since per-cell class coverage otherwise ranges
1/10 to 9/10.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FROZEN_CORE = ["car", "traffic sign", "traffic light", "person"]
CLIENTS = ["clear", "overcast", "rainy", "snowy", "partly cloudy"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})

CMAP = "viridis"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_joint(run):
    return "per_client_pooled_mAP" in run


def all_classes_seen(run):
    if is_joint(run):
        pcs = run.get("per_client_summary")
        if not pcs:
            return []
        any_client = next(iter(pcs.values()))
        return list(any_client["per_class"].keys())
    any_client = next(iter(run["class_history"].values()))
    return list(any_client["0"][-1].keys())


def final_per_class(run, client):
    """dict[class_name] -> metrics dict, for one client's final model."""
    if is_joint(run):
        pcs = run.get("per_client_summary")
        if not pcs:
            return None
        return pcs[client]["per_class"]

    ch = run["class_history"][client]
    task_ids = sorted(ch.keys(), key=int)
    per_class = {}
    for cls in all_classes_seen(run):
        snaps = [ch[t][-1][cls] for t in task_ids if ch[t][-1][cls]["N"] > 0]
        if not snaps:
            per_class[cls] = {"N": 0}
            continue
        keys = [k for k in snaps[0] if isinstance(snaps[0][k], (int, float))]
        # nanmean: a task with zero instances in a size band (e.g. N_L==0)
        # reports that band's AP/AR as NaN for that task; skip it rather than
        # letting one empty band poison the other two tasks' real values.
        per_class[cls] = {k: float(np.nanmean([s[k] for s in snaps])) for k in keys}
        per_class[cls]["N"] = int(np.sum([s["N"] for s in snaps]))
    return per_class


def macro_over_core(per_class, key, core=FROZEN_CORE):
    vals = [per_class[c][key] for c in core if per_class.get(c, {}).get("N", 0) > 0]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def client_scalar_mAP(run, client):
    """The single final mAP number for one client, whatever the mode."""
    if is_joint(run):
        return run["per_client_pooled_mAP"][client]
    return run["per_client"][client]["final_mAP"]


def system_mean_std(run):
    if is_joint(run):
        vals = list(run["per_client_pooled_mAP"].values())
        return float(np.mean(vals)), float(np.std(vals))
    s = run["system"]["mAP"]
    return s["mean"], s["std"]


# --- plots -------------------------------------------------------------


def fig_per_cell_per_class(modes, metric_key, outfile, title, clients=CLIENTS):
    """modes: list of (label, run). Heatmap grid, one panel per mode."""
    usable = [(lbl, r) for lbl, r in modes if not (is_joint(r) and not r.get("per_client_summary"))]
    classes = all_classes_seen(usable[0][1])
    n = len(usable)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 4.2), squeeze=False)
    axes = axes[0]
    mat_all = []
    for lbl, run in usable:
        mat = np.array([[final_per_class(run, c)[cls][metric_key]
                          if final_per_class(run, c)[cls]["N"] > 0 else np.nan
                          for cls in classes] for c in clients])
        mat_all.append(mat)
    vmax = np.nanmax([np.nanmax(m) for m in mat_all])
    im = None
    for ax, (lbl, run), mat in zip(axes, usable, mat_all):
        im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(clients)))
        ax.set_yticklabels(clients if ax is axes[0] else [], fontsize=8)
        ax.set_title(lbl)
        for c in FROZEN_CORE:
            if c in classes:
                ax.get_xticklabels()[classes.index(c)].set_fontweight("bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.5, color="white" if mat[i, j] < vmax * 0.6 else "black")
    fig.colorbar(im, ax=axes, shrink=0.85, label=metric_key)
    fig.suptitle(title, y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(outfile)
    plt.close(fig)


def fig_per_cell_avg(modes, outfile, title, clients=CLIENTS):
    """Grouped bar: client x mode, final mAP (core-class macro)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_modes = len(modes)
    width = 0.8 / n_modes
    x = np.arange(len(clients))
    for i, (lbl, run) in enumerate(modes):
        vals = [client_scalar_mAP(run, c) for c in clients]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.set_ylabel("final mAP@[.5:.95] (core-class macro)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(outfile)
    plt.close(fig)


def fig_system_avg(modes, outfile, title):
    """Bar: mode x mean +/- std across clients."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = [lbl for lbl, _ in modes]
    means = [system_mean_std(run)[0] for _, run in modes]
    stds = [system_mean_std(run)[1] for _, run in modes]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=plt.get_cmap(CMAP)(np.linspace(0.25, 0.85, len(labels))))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mAP@[.5:.95], mean +/- std across clients")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.003, f"{m:.4f}", ha="center", fontsize=8)
    fig.savefig(outfile)
    plt.close(fig)


def fig_small_large_per_mode(run, label, outfile, clients=CLIENTS):
    """2x2 heatmap grid (AP_S, AP_L, AR_S, AR_L) x (clients, classes) for one mode."""
    if is_joint(run) and not run.get("per_client_summary"):
        return False
    classes = all_classes_seen(run)
    metrics = ["AP_S", "AP_L", "AR_S", "AR_L"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, metric in zip(axes.flat, metrics):
        mat = np.array([[final_per_class(run, c)[cls][metric]
                          if final_per_class(run, c)[cls]["N"] > 0 else np.nan
                          for cls in classes] for c in clients])
        im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(clients)))
        ax.set_yticklabels(clients, fontsize=8)
        ax.set_title(metric)
        for c in FROZEN_CORE:
            if c in classes:
                ax.get_xticklabels()[classes.index(c)].set_fontweight("bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.5, color="white" if mat[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Small vs large object AP/AR per cell per class - {label}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile)
    plt.close(fig)
    return True


def fig_system_small_large(modes, outfile, title, clients=CLIENTS):
    """Bar: mode x [mAP_S, mAP_L, mAR_S, mAR_L], mean +/- std across clients (core macro)."""
    metrics = ["AP_S", "AP_L", "AR_S", "AR_L"]
    usable = [(lbl, r) for lbl, r in modes if not (is_joint(r) and not r.get("per_client_summary"))]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    x = np.arange(len(usable))
    for i, metric in enumerate(metrics):
        means, stds = [], []
        for lbl, run in usable:
            vals = [macro_over_core(final_per_class(run, c), metric) for c in clients]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in usable], rotation=20, ha="right")
    ax.set_ylabel("core-class macro, mean +/- std across clients")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(outfile)
    plt.close(fig)


def fig_forgetting_bwt(modes, outfile, title, clients=CLIENTS):
    """modes: CL-mode runs only (skip joint - no task boundaries)."""
    usable = [(lbl, r) for lbl, r in modes if not is_joint(r)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, name in zip(axes, ["forgetting", "bwt"], ["Forgetting", "Backward transfer (BWT)"]):
        n_modes = len(usable)
        width = 0.8 / n_modes
        x = np.arange(len(clients))
        for i, (lbl, run) in enumerate(usable):
            vals = [run["per_client"][c][key] for c in clients]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(clients, rotation=20, ha="right")
        ax.set_title(name)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(outfile)
    plt.close(fig)


def fig_gossip_ops(gossip_runs, outfile, title, clients=CLIENTS):
    """gossip_runs: list of (label, run) for gossip-mode runs only."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n = len(gossip_runs)
    width = 0.8 / n
    x = np.arange(len(clients))
    for i, (lbl, run) in enumerate(gossip_runs):
        vals = [run["gossip_ops"][c] for c in clients]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=lbl)
        ax.bar_label(bars, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=20, ha="right")
    ax.set_ylabel("gossip operations (as initiator)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(outfile)
    plt.close(fig)


def fig_gate_distribution(gossip_runs, outfile, title):
    """Histograms of per-exchange alpha and neighbour gate weight."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for lbl, run in gossip_runs:
        alphas = [rec["alpha"] for rec in run["gate_log"]]
        weights = [nb["weight"] for rec in run["gate_log"] for nb in rec["neighbours"]]
        axes[0].hist(alphas, bins=30, alpha=0.55, label=lbl)
        axes[1].hist(weights, bins=30, alpha=0.55, label=lbl)
    axes[0].set_title("Applied mixing dose (alpha = lambda x gate)")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=0.7, label="lambda ceiling")
    axes[0].set_xlabel("alpha")
    axes[1].set_title("Per-neighbour gate weight")
    axes[1].set_xlabel("weight (1.0 = full trust)")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(outfile)
    plt.close(fig)


def fig_cross_client_matrices(modes, outfile, title, clients=CLIENTS):
    usable = [(lbl, r) for lbl, r in modes if r.get("cross_client")]
    n = len(usable)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1, 4), squeeze=False)
    axes = axes[0]
    im = None
    for ax, (lbl, run) in zip(axes, usable):
        cc = run["cross_client"]
        mat = np.full((len(clients), len(clients)), np.nan)
        for i, ci in enumerate(clients):
            for j, cj in enumerate(clients):
                if ci == cj:
                    mat[i, j] = cc["diagonal"][ci]["mAP@[.5:.95]"]
                elif cj in cc["matrix"].get(ci, {}):
                    mat[i, j] = cc["matrix"][ci][cj]["mAP@[.5:.95]"]
        im = ax.imshow(mat, cmap=CMAP, vmin=0, aspect="auto")
        ax.set_xticks(range(len(clients)))
        ax.set_xticklabels(clients, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(clients)))
        ax.set_yticklabels(clients if ax is axes[0] else [], fontsize=8)
        ax.set_xlabel("evaluated on cell's data")
        if ax is axes[0]:
            ax.set_ylabel("client's own final model")
        ax.set_title(lbl)
        for i in range(len(clients)):
            for j in range(len(clients)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if mat[i, j] < np.nanmax(mat) * 0.6 else "black",
                            fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=axes, shrink=0.85, label="mAP@[.5:.95]")
    fig.suptitle(title, y=1.03, fontweight="bold")
    fig.savefig(outfile)
    plt.close(fig)


def fig_own_vs_other_gap(modes, outfile, title, clients=CLIENTS):
    usable = [(lbl, r) for lbl, r in modes if r.get("cross_client")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n = len(usable)
    width = 0.8 / n
    x = np.arange(len(clients))
    for i, (lbl, run) in enumerate(usable):
        vals = [run["cross_client"]["per_client"][c]["own_vs_other_gap"] for c in clients]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=lbl)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=20, ha="right")
    ax.set_ylabel("own mAP - mean cross mAP\n(positive = personalized, negative = generalizes better than own)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(outfile)
    plt.close(fig)
