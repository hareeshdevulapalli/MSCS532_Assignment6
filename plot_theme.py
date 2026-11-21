"""Look and feel for the result charts (same style as my Assignment 5).

Each algorithm / structure keeps the same colour and marker everywhere:
    deterministic (median of medians), array-based  -> navy, diamonds
    randomized quickselect, linked                  -> brick red, circles
    Python built-in reference                       -> grey, squares
    fixed-pivot / naive baseline                    -> amber, triangles
"""

PALETTE = {
    "Deterministic": "#1f3b73",
    "Randomized": "#c8553d",
    "sorted()": "#8d99ae",
    "Fixed pivot": "#e09f3e",
    "array": "#1f3b73",
    "linked": "#c8553d",
    "builtin": "#8d99ae",
    "naive": "#e09f3e",
}
SHAPES = {
    "Deterministic": "D",
    "Randomized": "o",
    "sorted()": "s",
    "Fixed pivot": "^",
    "array": "D",
    "linked": "o",
    "builtin": "s",
    "naive": "^",
}
NICE_NAMES = {
    "Deterministic": "Median of medians (deterministic_select)",
    "Randomized": "Quickselect (randomized_select)",
    "sorted()": "sorted(data)[k-1]  (C Timsort, reference)",
    "Fixed pivot": "Last-item pivot (fixed_pivot_select)",
}
DIST_NAMES = {
    "random": "Random",
    "sorted": "Sorted",
    "reverse": "Reverse-sorted",
    "few_unique": "Few unique (10 values)",
}
DIST_LINES = {
    "random": "-",
    "sorted": "--",
    "reverse": ":",
    "few_unique": "-.",
}

INK = "#222222"
MUTED = "#6c757d"
FAINT = "#d9d9d9"
PAPER = "#ffffff"
SHADE = "#f4f1ea"


def apply(plt):
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": FAINT,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": FAINT,
        "legend.fancybox": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })
