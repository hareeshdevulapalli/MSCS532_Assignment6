# chart_style.py - colours and matplotlib settings for my charts
#
# I kept one colour per algorithm/structure type across every figure:
#   green  = median of medians / array based
#   orange = quickselect / linked
#   grey   = Python built-in (just a reference)
#   purple = the "bad" baseline (last pivot, list.pop(0))

COLOR = {
    "MoM": "#2a7f62",
    "Quickselect": "#e4572e",
    "sorted()": "#8c8c8c",
    "Last pivot": "#7b4ea3",
    "array": "#2a7f62",
    "linked": "#e4572e",
    "builtin": "#8c8c8c",
    "bad": "#7b4ea3",
}

MARKER = {
    "MoM": "s",
    "Quickselect": "^",
    "sorted()": "P",
    "Last pivot": "v",
    "array": "s",
    "linked": "^",
    "builtin": "P",
    "bad": "v",
}

LABEL = {
    "MoM": "median of medians",
    "Quickselect": "quickselect (random pivot)",
    "sorted()": "sorted()[k-1]  (built-in, C)",
    "Last pivot": "quickselect, last item as pivot",
}

INPUT_TITLE = {
    "random": "random input",
    "sorted": "sorted input",
    "reversed": "reverse sorted input",
    "few_values": "only 10 distinct values",
}

INPUT_DASH = {
    "random": "-",
    "sorted": "--",
    "reversed": ":",
    "few_values": "-.",
}

BG = "#f7f7f7"
TEXT = "#333333"
GRID = "#ffffff"


def use(plt):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": BG,
        "axes.edgecolor": "#bbbbbb",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": TEXT,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
        "axes.titlepad": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.1,
        "grid.linestyle": "-",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
    })
