"""
Benchmarks for both parts of the assignment.

Part 1 (selection)
    S1  running time and comparisons vs n on random, sorted, reverse-sorted
        and few-unique inputs, for deterministic_select, randomized_select
        and Python's sorted() as a reference point
    S2  how much the work varies from run to run (many trials, one n)
    S3  how the work depends on k (which order statistic you ask for)
    S4  fixed-pivot quickselect vs the other two on sorted input, to show
        what randomizing the pivot buys

Part 2 (data structures)
    D1  stack: n pushes then n pops
    D2  queue: n enqueues then n dequeues
    D3  n inserts at the front
    D4  n reads at random positions
    D5  n appends at the back
    D6  memory per stored item

Usage:
    python3 run_benchmarks.py           full run (a few minutes)
    python3 run_benchmarks.py --quick   small sizes, a few seconds

Everything goes into results/ (results/quick/ for --quick): one CSV per
experiment, the printed tables in benchmark_output.txt, and the charts
(drawn by plot_results.py).
"""

import argparse
import collections
import csv
import gc
import math
import os
import random
import statistics
import sys
import time
import tracemalloc

from data_structures import (ArrayQueue, ArrayStack, DynamicArray, LinkedQueue,
                             LinkedStack, SinglyLinkedList)
from selection import (SelectTracker, deterministic_select, fixed_pivot_select,
                       randomized_select)

SEED = 532
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

FULL = dict(
    sizes=[1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000],
    repeats=5,
    spread_n=10_000, spread_trials=500,
    k_n=50_000, k_trials=100,
    fixed_sizes=[250, 500, 1_000, 2_000, 4_000],
    ds_sizes=[1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000],
    ds_slow_sizes=[250, 500, 1_000, 2_000, 4_000, 8_000],
    ds_repeats=3,
    mem_n=100_000,
)
QUICK = dict(
    sizes=[1_000, 2_000, 5_000, 10_000, 20_000],
    repeats=3,
    spread_n=5_000, spread_trials=100,
    k_n=10_000, k_trials=10,
    fixed_sizes=[250, 500, 1_000],
    ds_sizes=[1_000, 2_000, 5_000, 10_000],
    ds_slow_sizes=[250, 500, 1_000, 2_000],
    ds_repeats=2,
    mem_n=20_000,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

class Tee:
    """Print to the terminal and to benchmark_output.txt at the same time."""

    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def make_input(kind, n, rng):
    if kind == "random":
        return [rng.randrange(10 * n) for _ in range(n)]
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "few_unique":
        return [rng.randrange(10) for _ in range(n)]
    raise ValueError(kind)


DISTRIBUTIONS = ("random", "sorted", "reverse", "few_unique")


def timed(fn):
    """Run fn() once with the garbage collector off; return (seconds, result)."""
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        result = fn()
        return time.perf_counter() - start, result
    finally:
        gc.enable()


def write_csv(name, header, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:7.2f} ms"
    return f"{seconds:7.3f} s "


def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------
# Part 1: selection
# ----------------------------------------------------------------------

def bench_selection_vs_n(cfg):
    header("S1. Selection: time and comparisons vs n (k = lower median)")
    rows = []
    for kind in DISTRIBUTIONS:
        print(f"\n-- {kind} --")
        print(f"{'n':>8} | {'deterministic':>13} {'cmp/n':>6} | "
              f"{'randomized':>11} {'cmp/n':>6} | {'sorted()':>10} | det/rand")
        for n in cfg["sizes"]:
            data = make_input(kind, n, random.Random(SEED + n))
            k = (n + 1) // 2
            answer = sorted(data)[k - 1]
            stats = {}
            for name in ("Deterministic", "Randomized", "sorted()"):
                times, comps, exams = [], [], []
                for rep in range(cfg["repeats"]):
                    t = SelectTracker()
                    if name == "Deterministic":
                        sec, got = timed(lambda: deterministic_select(
                            data, k, tracker=t))
                    elif name == "Randomized":
                        rng = random.Random(SEED * 1000 + rep)
                        sec, got = timed(lambda: randomized_select(
                            data, k, rng=rng, tracker=t))
                    else:
                        sec, got = timed(lambda: sorted(data)[k - 1])
                    assert got == answer, (name, kind, n)
                    times.append(sec)
                    comps.append(t.comparisons)
                    exams.append(t.examined)
                med = statistics.median(times)
                c = statistics.mean(comps) if name != "sorted()" else ""
                e = statistics.mean(exams) if name != "sorted()" else ""
                stats[name] = (med, c)
                rows.append([kind, n, name, f"{med:.6e}", c, e])
            d, r, s = stats["Deterministic"], stats["Randomized"], stats["sorted()"]
            print(f"{n:>8} | {fmt_time(d[0]):>13} {d[1] / n:6.2f} | "
                  f"{fmt_time(r[0]):>11} {r[1] / n:6.2f} | "
                  f"{fmt_time(s[0]):>10} | {d[0] / r[0]:5.2f}x")
    write_csv("selection_times.csv",
              ["distribution", "n", "algorithm", "median_seconds",
               "mean_comparisons", "mean_examined"], rows)


def bench_selection_spread(cfg):
    n, trials = cfg["spread_n"], cfg["spread_trials"]
    header(f"S2. Run-to-run variation: {trials} trials, n = {n:,}, "
           f"new random input each trial")
    rows = []
    rng = random.Random(SEED)
    results = {"Deterministic": [], "Randomized": []}
    for trial in range(trials):
        data = make_input("random", n, rng)
        k = (n + 1) // 2
        for name, fn in (("Deterministic", deterministic_select),
                         ("Randomized", randomized_select)):
            t = SelectTracker()
            kwargs = {"rng": random.Random(trial)} if name == "Randomized" else {}
            fn(data, k, tracker=t, **kwargs)
            results[name].append(t)
            rows.append([name, trial, n, t.comparisons, t.examined, t.partitions])
    print(f"{'':15} {'cmp/n mean':>10} {'sd':>6} {'min':>6} {'max':>6} | "
          f"{'examined/n mean':>15} {'max':>6}")
    for name, ts in results.items():
        c = [t.comparisons / n for t in ts]
        e = [t.examined / n for t in ts]
        print(f"{name:15} {statistics.mean(c):10.3f} {statistics.stdev(c):6.3f} "
              f"{min(c):6.3f} {max(c):6.3f} | {statistics.mean(e):15.3f} "
              f"{max(e):6.3f}")
    expected = 2 + 2 * math.log(2)
    print(f"Theory for randomized, examined/n at the median: 2 + 2 ln 2 = "
          f"{expected:.3f}")
    write_csv("selection_spread.csv",
              ["algorithm", "trial", "n", "comparisons", "examined", "partitions"],
              rows)


def knuth_expected(n, k):
    """Expected items examined by randomized quickselect (Knuth 1971).

    About 2n + 2k ln(n/k) + 2(n-k) ln(n/(n-k)); exact harmonic form used.
    """
    def H(m):
        return sum(1.0 / i for i in range(1, m + 1)) if m > 0 else 0.0
    return 2 * ((n + 1) * H(n) - (n + 3 - k) * H(n - k + 1)
                - (k + 2) * H(k) + n + 3)


def bench_selection_vs_k(cfg):
    n, trials = cfg["k_n"], cfg["k_trials"]
    header(f"S3. Work vs k: n = {n:,} random items, {trials} randomized "
           f"trials per k")
    fractions = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    data = make_input("random", n, random.Random(SEED))
    rows = []
    print(f"{'k':>8} {'k/n':>6} | {'rand examined/n':>15} {'min':>6} {'max':>6}"
          f" {'theory':>7} | {'det examined/n':>14} {'det cmp/n':>9}")
    for f in fractions:
        k = min(n, max(1, round(f * n)))
        exams = []
        for trial in range(trials):
            t = SelectTracker()
            randomized_select(data, k, rng=random.Random(k * 10_000 + trial),
                              tracker=t)
            exams.append(t.examined / n)
        td = SelectTracker()
        deterministic_select(data, k, tracker=td)
        theory = knuth_expected(n, k) / n
        rows.append([k, f"{k / n:.4f}", f"{statistics.mean(exams):.4f}",
                     f"{min(exams):.4f}", f"{max(exams):.4f}", f"{theory:.4f}",
                     f"{td.examined / n:.4f}", f"{td.comparisons / n:.4f}"])
        print(f"{k:>8} {k / n:6.3f} | {statistics.mean(exams):15.3f} "
              f"{min(exams):6.3f} {max(exams):6.3f} {theory:7.3f} | "
              f"{td.examined / n:14.3f} {td.comparisons / n:9.3f}")
    write_csv("selection_vs_k.csv",
              ["k", "k_over_n", "rand_examined_mean", "rand_examined_min",
               "rand_examined_max", "theory_examined", "det_examined",
               "det_comparisons"], rows)


def bench_fixed_pivot(cfg):
    header("S4. Why randomize? Fixed (last-item) pivot vs the others, "
           "median of sorted input")
    rows = []
    print(f"{'n':>6} | {'fixed pivot':>11} {'exam/n':>7} | {'randomized':>11} "
          f"{'exam/n':>7} | {'deterministic':>13} {'exam/n':>7}")
    for n in cfg["fixed_sizes"]:
        data = list(range(n))
        k = (n + 1) // 2
        line = []
        for name, fn in (("Fixed pivot", fixed_pivot_select),
                         ("Randomized", randomized_select),
                         ("Deterministic", deterministic_select)):
            times, exams = [], []
            for rep in range(cfg["repeats"]):
                t = SelectTracker()
                kwargs = ({"rng": random.Random(rep)}
                          if name == "Randomized" else {})
                sec, got = timed(lambda: fn(data, k, tracker=t, **kwargs))
                assert got == k - 1
                times.append(sec)
                exams.append(t.examined)
            med = statistics.median(times)
            e = statistics.mean(exams)
            rows.append(["sorted", n, name, f"{med:.6e}", e])
            line.append((med, e / n))
        print(f"{n:>6} | {fmt_time(line[0][0]):>11} {line[0][1]:7.1f} | "
              f"{fmt_time(line[1][0]):>11} {line[1][1]:7.2f} | "
              f"{fmt_time(line[2][0]):>13} {line[2][1]:7.2f}")
    write_csv("fixed_pivot.csv",
              ["distribution", "n", "algorithm", "median_seconds",
               "mean_examined"], rows)


# ----------------------------------------------------------------------
# Part 2: data structures
# ----------------------------------------------------------------------

def best_time(fn, repeats):
    return min(timed(fn)[0] for _ in range(repeats))


def stack_run(make, push, pop, n):
    def go():
        s = make()
        p, q = getattr(s, push), getattr(s, pop)
        for i in range(n):
            p(i)
        for _ in range(n):
            q()
    return go


def bench_ds(cfg):
    rows = []
    reps = cfg["ds_repeats"]

    def record(experiment, name, n, sec):
        rows.append([experiment, name, n, f"{sec:.6e}"])
        return sec

    def table(experiment, title, sizes, cases):
        header(title)
        names = [c[0] for c in cases]
        print(f"{'n':>8} | " + " | ".join(f"{nm:>26}" for nm in names))
        for n in sizes:
            cells = []
            for name, factory in cases:
                limit = factory.limit if hasattr(factory, "limit") else None
                if limit is not None and n > limit:
                    cells.append(f"{'(skipped)':>26}")
                    continue
                sec = record(experiment, name, n, best_time(factory(n), reps))
                cells.append(f"{fmt_time(sec)} ({sec / n * 1e9:6.0f} ns/op)")
            print(f"{n:>8} | " + " | ".join(f"{c:>26}" for c in cells))

    sizes, slow = cfg["ds_sizes"], cfg["ds_slow_sizes"]

    table("stack", "D1. Stack: n pushes then n pops", sizes, [
        ("ArrayStack", lambda n: stack_run(ArrayStack, "push", "pop", n)),
        ("LinkedStack", lambda n: stack_run(LinkedStack, "push", "pop", n)),
        ("Python list", lambda n: stack_run(list, "append", "pop", n)),
    ])

    def naive_queue(n):
        def go():
            q = []
            for i in range(n):
                q.append(i)
            for _ in range(n):
                q.pop(0)
        return go

    table("queue", "D2. Queue: n enqueues then n dequeues", sizes, [
        ("ArrayQueue (ring)",
         lambda n: stack_run(ArrayQueue, "enqueue", "dequeue", n)),
        ("LinkedQueue",
         lambda n: stack_run(LinkedQueue, "enqueue", "dequeue", n)),
        ("list + pop(0)", naive_queue),
        ("collections.deque",
         lambda n: stack_run(collections.deque, "append", "popleft", n)),
    ])

    def front_inserts(make, method):
        def factory(n):
            def go():
                s = make()
                if method == "insert0":
                    for i in range(n):
                        s.insert(0, i)
                else:
                    for i in range(n):
                        s.push_front(i)
            return go
        return factory

    table("front_insert", "D3. n inserts at the front", slow, [
        ("DynamicArray.insert(0)", front_inserts(DynamicArray, "insert0")),
        ("SinglyLinkedList", front_inserts(SinglyLinkedList, "push_front")),
        ("Python list.insert(0)", front_inserts(list, "insert0")),
    ])

    def random_reads(kind):
        def factory(n):
            idx = [random.Random(SEED).randrange(n) for _ in range(n)]
            if kind == "array":
                s = DynamicArray(range(n))
                return lambda: [s[i] for i in idx]
            if kind == "linked":
                s = SinglyLinkedList(range(n))
                return lambda: [s.get(i) for i in idx]
            s = list(range(n))
            return lambda: [s[i] for i in idx]
        return factory

    table("random_access", "D4. n reads at random positions", slow, [
        ("DynamicArray[i]", random_reads("array")),
        ("SinglyLinkedList.get(i)", random_reads("linked")),
        ("Python list[i]", random_reads("list")),
    ])

    def appends(make, method):
        def factory(n):
            def go():
                s = make()
                f = getattr(s, method)
                for i in range(n):
                    f(i)
            return go
        return factory

    table("append", "D5. n appends at the back", sizes, [
        ("DynamicArray.append", appends(DynamicArray, "append")),
        ("SinglyLinkedList", appends(SinglyLinkedList, "push_back")),
        ("Python list.append", appends(list, "append")),
    ])

    write_csv("ds_times.csv", ["experiment", "structure", "n", "seconds"], rows)


def bench_memory(cfg):
    n = cfg["mem_n"]
    header(f"D6. Memory per item, n = {n:,} (payload excluded: every slot "
           f"holds the same object)")

    def fill(make, method):
        def go():
            s = make()
            f = getattr(s, method)
            for _ in range(n):
                f(None)
            return s
        return go

    cases = [
        ("DynamicArray", fill(DynamicArray, "append")),
        ("ArrayStack", fill(ArrayStack, "push")),
        ("ArrayQueue", fill(ArrayQueue, "enqueue")),
        ("SinglyLinkedList", fill(SinglyLinkedList, "push_back")),
        ("LinkedStack", fill(LinkedStack, "push")),
        ("LinkedQueue", fill(LinkedQueue, "enqueue")),
        ("Python list", fill(list, "append")),
        ("collections.deque", fill(collections.deque, "append")),
    ]
    rows = []
    for name, go in cases:
        gc.collect()
        tracemalloc.start()
        obj = go()
        current, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        per = current / n
        rows.append([name, n, current, f"{per:.2f}"])
        print(f"{name:20} {current / 1024:9.1f} KiB  {per:6.1f} bytes/item")
        del obj
    write_csv("ds_memory.csv", ["structure", "n", "bytes", "bytes_per_item"],
              rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true",
                        help="small sizes, finishes in seconds")
    parser.add_argument("--no-plots", action="store_true",
                        help="don't draw the charts afterwards")
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL
    global RESULTS
    if args.quick:            # never overwrite the full-run results
        RESULTS = os.path.join(RESULTS, "quick")

    os.makedirs(RESULTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS, "benchmark_output.txt"))
    print(f"Python {sys.version.split()[0]} on {sys.platform}, "
          f"seed {SEED}, {'quick' if args.quick else 'full'} run")
    started = time.perf_counter()

    bench_selection_vs_n(cfg)
    bench_selection_spread(cfg)
    bench_selection_vs_k(cfg)
    bench_fixed_pivot(cfg)
    bench_ds(cfg)
    bench_memory(cfg)

    print(f"\nTotal benchmark time: {time.perf_counter() - started:.1f} s")
    sys.stdout.flush()
    sys.stdout = sys.stdout.stdout

    if not args.no_plots:
        try:
            import plot_results
            plot_results.main(RESULTS)
        except ImportError as e:
            print(f"Skipping charts ({e}). pip install matplotlib to get them.")


if __name__ == "__main__":
    main()
