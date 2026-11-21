"""
Draw the charts in results/ from the CSV files run_benchmarks.py saved.

    python3 plot_results.py                  charts for results/
    python3 plot_results.py results/quick    charts for a --quick run

Charts:
    selection_times.png     time vs n, one panel per input shape
    selection_work.png      comparisons per item and time per item vs n
    selection_spread.png    run-to-run spread of the work at one n
    selection_vs_k.png      work vs which order statistic is asked for
    fixed_pivot.png         fixed vs random pivot on sorted input
    ds_operations.png       the data-structure timings and memory
"""

import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

import plot_theme as th  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def read(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


def line(ax, xs, ys, key, label=None, **kw):
    opts = dict(color=th.PALETTE[key], marker=th.SHAPES[key],
                label=label if label is not None else th.NICE_NAMES.get(key, key))
    opts.update(kw)
    ax.plot(xs, ys, **opts)


def reference_slope(ax, xs, y0, power, text):
    """Dotted guide line through (xs[0], y0) growing like n^power."""
    ys = [y0 * (x / xs[0]) ** power for x in xs]
    ax.plot(xs, ys, color=th.MUTED, linestyle=(0, (1, 2)), linewidth=1,
            marker=None, label=text)


# ----------------------------------------------------------------------
# Part 1
# ----------------------------------------------------------------------

def selection_times():
    rows = read("selection_times.csv")
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["distribution"]][r["algorithm"]].append(
            (int(r["n"]), float(r["median_seconds"])))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, sharey=True)
    for ax, dist in zip(axes.flat, th.DIST_NAMES):
        for algo in ("Deterministic", "Randomized", "sorted()"):
            pts = sorted(data[dist][algo])
            line(ax, [p[0] for p in pts], [p[1] * 1e3 for p in pts], algo)
        pts = sorted(data[dist]["Deterministic"])
        reference_slope(ax, [p[0] for p in pts], pts[0][1] * 1e3 * 1.6, 1,
                        "slope of O(n)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(th.DIST_NAMES[dist])
    for ax in axes[1]:
        ax.set_xlabel("n (items)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median time (ms)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Selecting the median: running time vs input size", x=0.06,
                 ha="left")
    fig.tight_layout()
    save(fig, "selection_times.png")


def selection_work():
    rows = read("selection_times.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    by = defaultdict(list)
    for r in rows:
        by[(r["distribution"], r["algorithm"])].append(r)
    for (dist, algo), rs in by.items():
        rs.sort(key=lambda r: int(r["n"]))
        ns = [int(r["n"]) for r in rs]
        label = f"{algo}, {th.DIST_NAMES[dist].lower()}"
        style = dict(linestyle=th.DIST_LINES[dist], label=label)
        if algo != "sorted()":
            line(left, ns, [float(r["mean_comparisons"]) / int(r["n"])
                            for r in rs], algo, **style)
        line(right, ns, [float(r["median_seconds"]) / int(r["n"]) * 1e9
                         for r in rs], algo, **style)
    left.set_title("Comparisons per item (flat line = linear work)")
    left.set_ylabel("comparisons / n")
    right.set_title("Time per item (flat line = linear time)")
    right.set_ylabel("nanoseconds / item")
    right.set_yscale("log")
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("n (items)")
    left.set_ylim(0, 12)
    handles, labels = right.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles, labels, ncol=4, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "selection_work.png")


def selection_spread():
    rows = read("selection_spread.csv")
    vals = defaultdict(list)
    for r in rows:
        vals[r["algorithm"]].append(int(r["comparisons"]))
    examined = [int(r["examined"]) for r in rows if r["algorithm"] == "Randomized"]
    n = int(rows[0]["n"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    for algo in ("Deterministic", "Randomized"):
        xs = [v / n for v in vals[algo]]
        left.hist(xs, bins=40, color=th.PALETTE[algo], alpha=0.75,
                  label=th.NICE_NAMES[algo], edgecolor=th.PAPER, linewidth=0.3)
    left.set_title(f"Comparisons per item over {len(vals['Randomized'])} trials"
                   f" (n = {n:,})")
    left.set_xlabel("comparisons / n")
    left.set_ylabel("trials")
    left.legend()
    xs = [e / n for e in examined]
    right.hist(xs, bins=40, color=th.PALETTE["Randomized"], alpha=0.75,
               edgecolor=th.PAPER, linewidth=0.3, label="measured")
    expect = 2 + 2 * math.log(2)
    right.axvline(expect, color=th.INK, linestyle="--", linewidth=1,
                  label=f"theory: 2 + 2 ln 2 = {expect:.2f}")
    mean = sum(xs) / len(xs)
    right.axvline(mean, color=th.PALETTE["Randomized"], linewidth=1.2,
                  label=f"measured mean = {mean:.2f}")
    right.set_title("Quickselect: items examined per item")
    right.set_xlabel("items examined / n")
    right.legend()
    fig.tight_layout()
    save(fig, "selection_spread.png")


def selection_vs_k():
    rows = read("selection_vs_k.csv")
    kf = [float(r["k_over_n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.fill_between(kf, [float(r["rand_examined_min"]) for r in rows],
                    [float(r["rand_examined_max"]) for r in rows],
                    color=th.PALETTE["Randomized"], alpha=0.15, linewidth=0,
                    label="quickselect, min to max over trials")
    line(ax, kf, [float(r["rand_examined_mean"]) for r in rows], "Randomized",
         label="quickselect, mean")
    ax.plot(kf, [float(r["theory_examined"]) for r in rows], color=th.INK,
            linestyle="--", linewidth=1, label="theory (Knuth): "
            r"$2n + 2k\ln\frac{n}{k} + 2(n-k)\ln\frac{n}{n-k}$")
    line(ax, kf, [float(r["det_examined"]) for r in rows], "Deterministic",
         label="median of medians (partition passes only)")
    ax.set_xlabel("k / n   (0 = minimum, 0.5 = median, 1 = maximum)")
    ax.set_ylabel("items examined / n")
    ax.set_title("Work depends on which order statistic you ask for")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "selection_vs_k.png")


def fixed_pivot():
    rows = read("fixed_pivot.csv")
    by = defaultdict(list)
    for r in rows:
        by[r["algorithm"]].append((int(r["n"]), float(r["median_seconds"]),
                                   float(r["mean_examined"])))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for algo in ("Fixed pivot", "Deterministic", "Randomized"):
        pts = sorted(by[algo])
        ns = [p[0] for p in pts]
        line(left, ns, [p[1] * 1e3 for p in pts], algo)
        line(right, ns, [p[2] / p[0] for p in pts], algo)
    pts = sorted(by["Fixed pivot"])
    reference_slope(left, [p[0] for p in pts], pts[0][1] * 1e3 * 0.5, 2,
                    "slope of O(n²)")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_title("Median of a sorted list: time")
    left.set_ylabel("median time (ms)")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_title("Items examined per item")
    right.set_ylabel("items examined / n")
    for ax in (left, right):
        ax.set_xlabel("n (items)")
    left.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fixed_pivot.png")


# ----------------------------------------------------------------------
# Part 2
# ----------------------------------------------------------------------

DS_STYLE = {
    "ArrayStack": "array", "LinkedStack": "linked", "Python list": "builtin",
    "ArrayQueue (ring)": "array", "LinkedQueue": "linked",
    "list + pop(0)": "naive", "collections.deque": "builtin",
    "DynamicArray.insert(0)": "array", "SinglyLinkedList": "linked",
    "Python list.insert(0)": "builtin",
    "DynamicArray[i]": "array", "SinglyLinkedList.get(i)": "linked",
    "Python list[i]": "builtin",
    "DynamicArray.append": "array", "Python list.append": "builtin",
}
DS_TITLES = {
    "stack": "D1. Stack: n push + n pop",
    "queue": "D2. Queue: n enqueue + n dequeue",
    "front_insert": "D3. n inserts at the front",
    "random_access": "D4. n reads at random positions",
    "append": "D5. n appends at the back",
}


def ds_operations():
    rows = read("ds_times.csv")
    mem = read("ds_memory.csv")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["experiment"]][r["structure"]].append(
            (int(r["n"]), float(r["seconds"])))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    for ax, exp in zip(axes.flat, DS_TITLES):
        for name, pts in by[exp].items():
            pts.sort()
            key = DS_STYLE[name]
            ax.plot([p[0] for p in pts], [p[1] / p[0] * 1e9 for p in pts],
                    color=th.PALETTE[key], marker=th.SHAPES[key], label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        if hi / lo < 30:                 # flat panels: show a fixed range
            ax.set_ylim(5, 500)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(DS_TITLES[exp])
        ax.set_xlabel("n")
        ax.set_ylabel("ns per operation")
        ax.legend(fontsize=7)
    ax = axes[1, 2]
    names = [m["structure"] for m in mem]
    per = [float(m["bytes_per_item"]) for m in mem]

    def mem_key(name):
        if name.startswith(("Dynamic", "Array")):
            return "array"
        if "Linked" in name:
            return "linked"
        return "builtin"
    ax.barh(names, per, color=[th.PALETTE[mem_key(n)] for n in names])
    for y, v in enumerate(per):
        ax.text(v + 0.8, y, f"{v:.1f}", va="center", fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("bytes per item (payload excluded)")
    ax.set_title(f"D6. Memory, n = {int(mem[0]['n']):,}")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(per) * 1.18)
    fig.suptitle("Elementary data structures: cost per operation "
                 "(flat = O(1) each, rising = O(n) each)", x=0.04, ha="left")
    fig.tight_layout()
    save(fig, "ds_operations.png")


def main(results_dir=None):
    global RESULTS
    if results_dir is not None:
        RESULTS = results_dir
    th.apply(plt)
    selection_times()
    selection_work()
    selection_spread()
    selection_vs_k()
    fixed_pivot()
    ds_operations()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
