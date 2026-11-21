"""
make_charts.py - turns the CSVs from benchmark.py into PNG charts.

    python3 make_charts.py                 uses results/
    python3 make_charts.py results/small   uses another folder

fig1_runtime.png    time vs n, one panel per input type
fig2_per_item.png   comparisons per element and time per element
fig3_variation.png  spread of the work over many runs
fig4_by_k.png       work depending on which k you ask for
fig5_pivot.png      last-item pivot vs random pivot on sorted input
fig6_structures.png data structure timings + memory
"""

import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

import chart_style as cs  # noqa: E402

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(fname):
    with open(os.path.join(FOLDER, fname), newline="") as f:
        return list(csv.DictReader(f))


def write_png(fig, fname):
    out = os.path.join(FOLDER, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def plot_algo(ax, xs, ys, key, **kw):
    style = {"color": cs.COLOR[key], "marker": cs.MARKER[key],
             "label": cs.LABEL.get(key, key)}
    style.update(kw)
    ax.plot(xs, ys, **style)


def guide(ax, xs, y_start, power, text):
    ax.plot(xs, [y_start * (x / xs[0]) ** power for x in xs], color="#555555",
            linewidth=1, linestyle=(0, (4, 3)), label=text)


def plain_numbers(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ---- part 1 ----------------------------------------------------------------

def fig1():
    rows = load("a1_time_vs_n.csv")
    pts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pts[r["input"]][r["algo"]].append((int(r["n"]), float(r["median_secs"])))
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, kind in zip(axs, cs.INPUT_TITLE):
        for algo in ("MoM", "Quickselect", "sorted()"):
            p = sorted(pts[kind][algo])
            plot_algo(ax, [a for a, _ in p], [b * 1000 for _, b in p], algo)
        p = sorted(pts[kind]["MoM"])
        guide(ax, [a for a, _ in p], p[0][1] * 1000 * 1.8, 1, "linear growth")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(cs.INPUT_TITLE[kind])
        ax.set_xlabel("n")
        plain_numbers(ax)
    axs[0].set_ylabel("time (ms, median of runs)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Fig 1. Time to find the median")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    write_png(fig, "fig1_runtime.png")


def fig2():
    rows = load("a1_time_vs_n.csv")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["input"], r["algo"])].append(r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for (kind, algo), rs in groups.items():
        rs.sort(key=lambda r: int(r["n"]))
        ns = [int(r["n"]) for r in rs]
        style = {"linestyle": cs.INPUT_DASH[kind],
                 "label": "%s / %s" % (algo, kind.replace("_", " "))}
        if algo != "sorted()":
            plot_algo(ax1, ns, [float(r["avg_compares"]) / int(r["n"])
                                for r in rs], algo, **style)
        plot_algo(ax2, ns, [float(r["median_secs"]) / int(r["n"]) * 1e9
                            for r in rs], algo, **style)
    ax1.set_title("comparisons per element")
    ax1.set_ylabel("comparisons / n")
    ax1.set_ylim(0, 12)
    ax2.set_title("time per element")
    ax2.set_ylabel("ns / element")
    ax2.set_yscale("log")
    plain_numbers(ax2)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("n")
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig 2. Work per element stays flat (= linear)")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    write_png(fig, "fig2_per_item.png")


def fig3():
    rows = load("a2_variation.csv")
    n = int(rows[0]["n"])
    cmps = defaultdict(list)
    for r in rows:
        cmps[r["algo"]].append(int(r["compares"]) / n)
    qs_touched = [int(r["touched"]) / n for r in rows if r["algo"] == "Quickselect"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for algo in ("MoM", "Quickselect"):
        ax1.hist(cmps[algo], bins=45, color=cs.COLOR[algo], alpha=0.8,
                 label=cs.LABEL[algo])
    ax1.set_title("comparisons / n, %d runs, n = %d" % (len(cmps["MoM"]), n))
    ax1.set_xlabel("comparisons / n")
    ax1.set_ylabel("number of runs")
    ax1.legend()

    ax2.hist(qs_touched, bins=45, color=cs.COLOR["Quickselect"], alpha=0.8)
    theory = 2 + 2 * math.log(2)
    avg = sum(qs_touched) / len(qs_touched)
    ax2.axvline(theory, color="black", linestyle="--", linewidth=1.2,
                label="expected 2 + 2ln2 = %.2f" % theory)
    ax2.axvline(avg, color=cs.COLOR["MoM"], linewidth=1.6,
                label="measured average = %.2f" % avg)
    ax2.set_title("quickselect: elements touched / n")
    ax2.set_xlabel("touched / n")
    ax2.legend()
    fig.suptitle("Fig 3. How much the work changes from run to run")
    fig.tight_layout()
    write_png(fig, "fig3_variation.png")


def fig4():
    rows = load("a3_by_k.csv")
    x = [float(r["k_frac"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(x, [float(r["qs_min"]) for r in rows],
                    [float(r["qs_max"]) for r in rows],
                    color=cs.COLOR["Quickselect"], alpha=0.18, linewidth=0,
                    label="quickselect range (min-max)")
    plot_algo(ax, x, [float(r["qs_avg"]) for r in rows], "Quickselect",
              label="quickselect average")
    ax.plot(x, [float(r["theory"]) for r in rows], color="black",
            linestyle="--", linewidth=1.2, label="Knuth's formula")
    plot_algo(ax, x, [float(r["mom_touched"]) for r in rows], "MoM",
              label="median of medians")
    ax.set_xlabel("k / n    (0 = min, 0.5 = median, 1 = max)")
    ax.set_ylabel("elements touched / n")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Fig 4. Work for different k")
    fig.tight_layout()
    write_png(fig, "fig4_by_k.png")


def fig5():
    rows = load("a4_pivot_choice.csv")
    pts = defaultdict(list)
    for r in rows:
        pts[r["algo"]].append((int(r["n"]), float(r["median_secs"]),
                               float(r["avg_touched"])))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for algo in ("Last pivot", "MoM", "Quickselect"):
        p = sorted(pts[algo])
        ns = [a[0] for a in p]
        plot_algo(ax1, ns, [a[1] * 1000 for a in p], algo)
        plot_algo(ax2, ns, [a[2] / a[0] for a in p], algo)
    p = sorted(pts["Last pivot"])
    guide(ax1, [a[0] for a in p], p[0][1] * 1000 * 0.4, 2, "quadratic growth")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        plain_numbers(ax)
    ax1.set_title("time")
    ax1.set_ylabel("ms")
    ax2.set_title("elements touched / n")
    ax1.legend()
    fig.suptitle("Fig 5. Median of an already sorted list")
    fig.tight_layout()
    write_png(fig, "fig5_pivot.png")


# ---- part 2 ----------------------------------------------------------------

KIND = {
    "ArrStack": "array", "NodeStack": "linked", "list (built-in)": "builtin",
    "RingQueue": "array", "NodeQueue": "linked", "list.pop(0)": "bad",
    "deque (built-in)": "builtin", "ResizableArray": "array",
    "LinkedList": "linked", "list.insert(0)": "builtin",
}
PANEL = {
    "B1": "stack: push n, pop n",
    "B2": "queue: enqueue n, dequeue n",
    "B3": "insert at the front",
    "B4": "lookup at random index",
    "B5": "append at the end",
}


def fig6():
    rows = load("b_times.csv")
    mem = load("b6_memory.csv")
    pts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pts[r["exp"]][r["structure"]].append((int(r["n"]), float(r["secs"])))
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 8))
    for ax, exp in zip(axs.flat, PANEL):
        for name, p in pts[exp].items():
            p.sort()
            kind = KIND[name]
            ax.plot([a for a, _ in p], [b / a * 1e9 for a, b in p],
                    color=cs.COLOR[kind], marker=cs.MARKER[kind], label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        if hi / lo < 30:
            ax.set_ylim(5, 500)
        plain_numbers(ax)
        ax.set_title("%s  %s" % (exp, PANEL[exp]))
        ax.set_xlabel("n")
        ax.set_ylabel("ns per operation")
        ax.legend(fontsize=7.5)

    ax = axs[1, 2]
    names = [m["structure"] for m in mem]
    vals = [float(m["bytes_per_item"]) for m in mem]

    def kind_of(name):
        if name in ("ResizableArray", "ArrStack", "RingQueue"):
            return "array"
        if name in ("LinkedList", "NodeStack", "NodeQueue"):
            return "linked"
        return "builtin"
    ax.barh(names, vals, color=[cs.COLOR[kind_of(x)] for x in names],
            height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.7, i, "%.1f B" % v, va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("bytes per item")
    ax.set_title("B6  memory, n = %s" % format(int(mem[0]["n"]), ","))
    fig.suptitle("Fig 6. Data structures: cost of one operation "
                 "(flat line = O(1), going up = O(n))")
    fig.tight_layout()
    write_png(fig, "fig6_structures.png")


def main(folder=None):
    global FOLDER
    if folder:
        FOLDER = folder
    cs.use(plt)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
