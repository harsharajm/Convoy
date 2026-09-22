"""
Shared plotting style for the results figures.

Palette and mark rules follow the project's data-viz conventions:

  * Categorical hues are assigned to *entities* (run modes) in a fixed order and
    never cycled or reassigned by rank, so a mode keeps its colour across every
    figure in the deck.
  * The four-colour set below was validated all-pairs on a light surface
    (worst CVD dE 9.2, worst normal-vision dE 16.3). Aqua sits at 2.74:1
    contrast, under the 3:1 bar, which obliges visible direct labels on any
    chart that uses it - every bar chart here carries its value.
  * Sequential encoding (the 5x5 matrices) uses ONE hue light->dark. Never a
    rainbow.
  * Grid and axes are solid hairlines one shade off the surface; spines on the
    value axis only.

Figures are rendered for a light slide background at 16:9-ish aspect ratios.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- surfaces and ink ------------------------------------------------------

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e3"

# --- categorical slots, bound to run modes (never reassigned) --------------

MODE_COLOR = {
    "gossip":     "#2a78d6",   # slot 1, blue   - the mechanism under test
    "local_only": "#eb6834",   # slot 2, orange - the baseline it must beat
    "sequential": "#1baf7a",   # slot 3, aqua   - needs direct labels (2.74:1)
    "joint":      "#4a3aa7",   # slot 7, violet - centralised ceiling
}

MODE_LABEL = {
    "gossip":     "Gossip\n(FL + replay + gate)",
    "local_only": "Local-only\n(replay, no FL)",
    "sequential": "Sequential\n(no replay, no FL)",
    "joint":      "Joint\n(centralised ceiling)",
}

MODE_SHORT = {
    "gossip": "gossip",
    "local_only": "local-only",
    "sequential": "sequential",
    "joint": "joint",
}

# --- sequential ramp for the cross-client matrices -------------------------

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("bdd_blue", BLUE_RAMP, N=256)

CLIENTS = ["clear", "overcast", "rainy", "snowy", "partly cloudy"]
TASKS = ["daytime", "night", "dawn/dusk"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",          # never dashed
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def tidy(ax, axis="y", grid=True):
    """Hairline grid on the value axis only; drop the box."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis=axis, color=GRID, linewidth=0.8)
        ax.grid(False, axis="x" if axis == "y" else "y")


def caption(fig, text):
    fig.text(0.005, -0.02, text, ha="left", va="top",
             fontsize=8, color=INK_MUTED, wrap=True)
