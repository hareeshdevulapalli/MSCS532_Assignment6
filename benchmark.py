"""
benchmark.py - runs all the timing experiments for the assignment.

Selection (part 1):
    A1  time + comparisons vs n, for random / sorted / reversed / few-values
        input. Compares select_mom, select_random and sorted() as a
        reference.
    A2  how much quickselect's work jumps around between runs
    A3  work as a function of k (min ... median ... max)
    A4  last-element pivot vs the other two on sorted input

Data structures (part 2):
    B1  stack:  n pushes, then n pops
    B2  queue:  n enqueues, then n dequeues
    B3  n inserts at the front
    B4  n lookups at random indexes
    B5  n appends at the end
    B6  memory used per item

    python3 benchmark.py          full run (~30 s)
    python3 benchmark.py --small  tiny sizes, a few seconds; writes to
                                  results/small/ so the real results stay

Output: CSV files + log.txt in results/, then the charts via make_charts.py
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

from data_structures import (ArrStack, LinkedList, NodeQueue, NodeStack,
                             ResizableArray, RingQueue)
from selection import WorkLog, select_last_pivot, select_mom, select_random

SEED = 77
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

BIG = {
    "sizes": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
    "reps": 5,
    "var_n": 10000, "var_runs": 500,
    "k_n": 50000, "k_runs": 100,
    "pivot_sizes": [250, 500, 1000, 2000, 4000],
    "ds_sizes": [1000, 2000, 5000, 10000, 20000, 50000, 100000],
    "slow_sizes": [250, 500, 1000, 2000, 4000, 8000],
    "ds_reps": 3,
    "mem_n": 100000,
}
SMALL = {
    "sizes": [1000, 2000, 5000, 10000],
    "reps": 2,
    "var_n": 5000, "var_runs": 80,
    "k_n": 10000, "k_runs": 10,
    "pivot_sizes": [250, 500, 1000],
    "ds_sizes": [1000, 2000, 5000],
    "slow_sizes": [250, 500, 1000],
    "ds_reps": 2,
    "mem_n": 20000,
}

INPUT_KINDS = ["random", "sorted", "reversed", "few_values"]


class Both:
    """Sends print() output to the screen and to log.txt."""

    def __init__(self, path):
        self.screen = sys.stdout
        self.fh = open(path, "w")

    def write(self, s):
        self.screen.write(s)
        self.fh.write(s)

    def flush(self):
        self.screen.flush()
        self.fh.flush()


def build_input(kind, n, rng):
    if kind == "random":
        return [rng.randrange(n * 10) for _ in range(n)]
    if kind == "sorted":
        return list(range(n))
    if kind == "reversed":
        return list(range(n, 0, -1))
    if kind == "few_values":
        return [rng.randrange(10) for _ in range(n)]
    raise ValueError(kind)


def clock(fn):
    """Time one call of fn() with gc switched off. Returns (secs, result)."""
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        res = fn()
        return time.perf_counter() - t0, res
    finally:
        gc.enable()


def save_csv(fname, cols, rows):
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def nice(secs):
    if secs < 1e-3:
        return "%.1f us" % (secs * 1e6)
    if secs < 1:
        return "%.2f ms" % (secs * 1e3)
    return "%.3f s" % secs


def title(text):
    print("\n" + text)
    print("-" * len(text))


# ---------------------------------------------------------------------------
# part 1
# ---------------------------------------------------------------------------

def a1_time_vs_n(cfg):
    title("A1: finding the median, time and comparisons vs n")
    rows = []
    for kind in INPUT_KINDS:
        print("\n[%s]" % kind)
        print("%8s  %12s %6s  %12s %6s  %10s  %s"
              % ("n", "MoM", "cmp/n", "Quickselect", "cmp/n", "sorted()",
                 "MoM/QS"))
        for n in cfg["sizes"]:
            data = build_input(kind, n, random.Random(SEED + n))
            k = (n + 1) // 2
            right_answer = sorted(data)[k - 1]
            res = {}
            for algo in ("MoM", "Quickselect", "sorted()"):
                times, cmps, touched = [], [], []
                for rep in range(cfg["reps"]):
                    log = WorkLog()
                    if algo == "MoM":
                        secs, got = clock(lambda: select_mom(data, k, log=log))
                    elif algo == "Quickselect":
                        g = random.Random(SEED * 31 + rep)
                        secs, got = clock(
                            lambda: select_random(data, k, rng=g, log=log))
                    else:
                        secs, got = clock(lambda: sorted(data)[k - 1])
                    assert got == right_answer
                    times.append(secs)
                    cmps.append(log.compares)
                    touched.append(log.touched)
                t = statistics.median(times)
                c = statistics.mean(cmps)
                res[algo] = (t, c)
                if algo == "sorted()":
                    rows.append([kind, n, algo, "%.6e" % t, "", ""])
                else:
                    rows.append([kind, n, algo, "%.6e" % t, c,
                                 statistics.mean(touched)])
            m, q, s = res["MoM"], res["Quickselect"], res["sorted()"]
            print("%8d  %12s %6.2f  %12s %6.2f  %10s  %.2fx"
                  % (n, nice(m[0]), m[1] / n, nice(q[0]), q[1] / n,
                     nice(s[0]), m[0] / q[0]))
    save_csv("a1_time_vs_n.csv",
             ["input", "n", "algo", "median_secs", "avg_compares",
              "avg_touched"], rows)


def a2_variation(cfg):
    n, runs = cfg["var_n"], cfg["var_runs"]
    title("A2: %d runs at n=%d (new random list each run)" % (runs, n))
    gen = random.Random(SEED)
    rows = []
    logs = {"MoM": [], "Quickselect": []}
    for run in range(runs):
        data = build_input("random", n, gen)
        k = (n + 1) // 2
        a = WorkLog()
        select_mom(data, k, log=a)
        b = WorkLog()
        select_random(data, k, rng=random.Random(run), log=b)
        for algo, lg in (("MoM", a), ("Quickselect", b)):
            logs[algo].append(lg)
            rows.append([algo, run, n, lg.compares, lg.touched, lg.passes])
    print("%-12s %9s %6s %6s %6s   %11s %6s"
          % ("", "cmp/n avg", "sd", "min", "max", "touched/n", "max"))
    for algo, lst in logs.items():
        c = [x.compares / n for x in lst]
        t = [x.touched / n for x in lst]
        print("%-12s %9.3f %6.3f %6.3f %6.3f   %11.3f %6.3f"
              % (algo, statistics.mean(c), statistics.stdev(c), min(c),
                 max(c), statistics.mean(t), max(t)))
    print("expected touched/n for quickselect at the median: 2+2ln2 = %.3f"
          % (2 + 2 * math.log(2)))
    save_csv("a2_variation.csv",
             ["algo", "run", "n", "compares", "touched", "passes"], rows)


def expected_touched(n, k):
    """Knuth's exact expected work for quickselect, harmonic number form."""
    def harm(m):
        return sum(1.0 / i for i in range(1, m + 1)) if m > 0 else 0.0
    return 2 * ((n + 1) * harm(n) - (n + 3 - k) * harm(n - k + 1)
                - (k + 2) * harm(k) + n + 3)


def a3_by_k(cfg):
    n, runs = cfg["k_n"], cfg["k_runs"]
    title("A3: work vs k (n=%d, %d quickselect runs per k)" % (n, runs))
    data = build_input("random", n, random.Random(SEED))
    fracs = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
             0.9, 0.95, 0.99, 1.0]
    rows = []
    print("%7s %6s  %9s %6s %6s %7s   %9s %8s"
          % ("k", "k/n", "QS avg", "min", "max", "theory", "MoM", "MoM cmp"))
    for f in fracs:
        k = min(n, max(1, round(f * n)))
        vals = []
        for run in range(runs):
            lg = WorkLog()
            select_random(data, k, rng=random.Random(k * 7919 + run), log=lg)
            vals.append(lg.touched / n)
        mlog = WorkLog()
        select_mom(data, k, log=mlog)
        th = expected_touched(n, k) / n
        avg = statistics.mean(vals)
        rows.append([k, "%.4f" % (k / n), "%.4f" % avg, "%.4f" % min(vals),
                     "%.4f" % max(vals), "%.4f" % th,
                     "%.4f" % (mlog.touched / n), "%.4f" % (mlog.compares / n)])
        print("%7d %6.3f  %9.3f %6.3f %6.3f %7.3f   %9.3f %8.3f"
              % (k, k / n, avg, min(vals), max(vals), th, mlog.touched / n,
                 mlog.compares / n))
    save_csv("a3_by_k.csv",
             ["k", "k_frac", "qs_avg", "qs_min", "qs_max", "theory",
              "mom_touched", "mom_compares"], rows)


def a4_pivot_choice(cfg):
    title("A4: median of a sorted list with different pivot rules")
    rows = []
    print("%6s  %11s %8s  %11s %8s  %11s %8s"
          % ("n", "last pivot", "touch/n", "Quickselect", "touch/n", "MoM",
             "touch/n"))
    algos = [("Last pivot", select_last_pivot), ("Quickselect", select_random),
             ("MoM", select_mom)]
    for n in cfg["pivot_sizes"]:
        data = list(range(n))
        k = (n + 1) // 2
        line = []
        for algo, fn in algos:
            times, touched = [], []
            for rep in range(cfg["reps"]):
                lg = WorkLog()
                extra = {"rng": random.Random(rep)} if algo == "Quickselect" else {}
                secs, got = clock(lambda: fn(data, k, log=lg, **extra))
                assert got == k - 1
                times.append(secs)
                touched.append(lg.touched)
            t = statistics.median(times)
            tc = statistics.mean(touched)
            rows.append([n, algo, "%.6e" % t, tc])
            line.append((t, tc / n))
        print("%6d  %11s %8.1f  %11s %8.2f  %11s %8.2f"
              % (n, nice(line[0][0]), line[0][1], nice(line[1][0]),
                 line[1][1], nice(line[2][0]), line[2][1]))
    save_csv("a4_pivot_choice.csv", ["n", "algo", "median_secs", "avg_touched"],
             rows)


# ---------------------------------------------------------------------------
# part 2
# ---------------------------------------------------------------------------

def fastest(fn, reps):
    return min(clock(fn)[0] for _ in range(reps))


def fill_then_drain(make, add, take, n):
    def run():
        obj = make()
        put, get = getattr(obj, add), getattr(obj, take)
        for i in range(n):
            put(i)
        for _ in range(n):
            get()
    return run


def run_table(rows, cfg, exp_id, heading, sizes, contenders):
    title(heading)
    print("%8s  " % "n" + "  ".join("%24s" % name for name, _ in contenders))
    for n in sizes:
        cells = []
        for name, setup in contenders:
            secs = fastest(setup(n), cfg["ds_reps"])
            rows.append([exp_id, name, n, "%.6e" % secs])
            cells.append("%10s %6.0f ns/op" % (nice(secs), secs / n * 1e9))
        print("%8d  " % n + "  ".join("%24s" % c for c in cells))


def part2(cfg):
    rows = []
    sz, slow = cfg["ds_sizes"], cfg["slow_sizes"]

    run_table(rows, cfg, "B1", "B1: stack, n push then n pop", sz, [
        ("ArrStack", lambda n: fill_then_drain(ArrStack, "push", "pop", n)),
        ("NodeStack", lambda n: fill_then_drain(NodeStack, "push", "pop", n)),
        ("list (built-in)", lambda n: fill_then_drain(list, "append", "pop", n)),
    ])

    def pop0_queue(n):
        def run():
            q = []
            for i in range(n):
                q.append(i)
            for _ in range(n):
                q.pop(0)
        return run

    run_table(rows, cfg, "B2", "B2: queue, n enqueue then n dequeue", sz, [
        ("RingQueue", lambda n: fill_then_drain(RingQueue, "enqueue", "dequeue", n)),
        ("NodeQueue", lambda n: fill_then_drain(NodeQueue, "enqueue", "dequeue", n)),
        ("list.pop(0)", pop0_queue),
        ("deque (built-in)",
         lambda n: fill_then_drain(collections.deque, "append", "popleft", n)),
    ])

    def front(make, how):
        def setup(n):
            def run():
                obj = make()
                if how == "insert":
                    for i in range(n):
                        obj.insert(0, i)
                else:
                    for i in range(n):
                        obj.add_front(i)
            return run
        return setup

    run_table(rows, cfg, "B3", "B3: n inserts at the front", slow, [
        ("ResizableArray", front(ResizableArray, "insert")),
        ("LinkedList", front(LinkedList, "add_front")),
        ("list.insert(0)", front(list, "insert")),
    ])

    def lookups(kind):
        def setup(n):
            g = random.Random(SEED)
            idx = [g.randrange(n) for _ in range(n)]
            if kind == "array":
                obj = ResizableArray(range(n))
                return lambda: [obj[i] for i in idx]
            if kind == "linked":
                obj = LinkedList(range(n))
                return lambda: [obj.get(i) for i in idx]
            obj = list(range(n))
            return lambda: [obj[i] for i in idx]
        return setup

    run_table(rows, cfg, "B4", "B4: n lookups at random indexes", slow, [
        ("ResizableArray", lookups("array")),
        ("LinkedList", lookups("linked")),
        ("list (built-in)", lookups("list")),
    ])

    def appends(make, meth):
        def setup(n):
            def run():
                obj = make()
                f = getattr(obj, meth)
                for i in range(n):
                    f(i)
            return run
        return setup

    run_table(rows, cfg, "B5", "B5: n appends at the end", sz, [
        ("ResizableArray", appends(ResizableArray, "append")),
        ("LinkedList", appends(LinkedList, "add_back")),
        ("list (built-in)", appends(list, "append")),
    ])

    save_csv("b_times.csv", ["exp", "structure", "n", "secs"], rows)


def b6_memory(cfg):
    n = cfg["mem_n"]
    title("B6: memory per item at n=%d (every slot holds None, so only the "
          "structure itself is counted)" % n)

    def filled(make, meth):
        def run():
            obj = make()
            f = getattr(obj, meth)
            for _ in range(n):
                f(None)
            return obj
        return run

    cases = [
        ("ResizableArray", filled(ResizableArray, "append")),
        ("ArrStack", filled(ArrStack, "push")),
        ("RingQueue", filled(RingQueue, "enqueue")),
        ("LinkedList", filled(LinkedList, "add_back")),
        ("NodeStack", filled(NodeStack, "push")),
        ("NodeQueue", filled(NodeQueue, "enqueue")),
        ("list (built-in)", filled(list, "append")),
        ("deque (built-in)", filled(collections.deque, "append")),
    ]
    rows = []
    for name, run in cases:
        gc.collect()
        tracemalloc.start()
        obj = run()
        used, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append([name, n, used, "%.2f" % (used / n)])
        print("%-18s %9.1f KiB   %5.1f bytes/item" % (name, used / 1024, used / n))
        del obj
    save_csv("b6_memory.csv", ["structure", "n", "bytes", "bytes_per_item"], rows)


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="Assignment 6 benchmarks")
    ap.add_argument("--small", action="store_true", help="tiny sizes")
    ap.add_argument("--no-charts", action="store_true", help="skip the charts")
    args = ap.parse_args()
    cfg = SMALL if args.small else BIG
    if args.small:
        OUT_DIR = os.path.join(OUT_DIR, "small")
    os.makedirs(OUT_DIR, exist_ok=True)

    sys.stdout = Both(os.path.join(OUT_DIR, "log.txt"))
    print("python %s, %s, seed=%d, %s run"
          % (sys.version.split()[0], sys.platform, SEED,
             "small" if args.small else "full"))
    t0 = time.perf_counter()
    a1_time_vs_n(cfg)
    a2_variation(cfg)
    a3_by_k(cfg)
    a4_pivot_choice(cfg)
    part2(cfg)
    b6_memory(cfg)
    print("\ndone in %.1f s" % (time.perf_counter() - t0))
    sys.stdout.flush()
    sys.stdout = sys.stdout.screen

    if not args.no_charts:
        try:
            import make_charts
        except ImportError as err:
            print("no charts (%s) - install matplotlib" % err)
        else:
            make_charts.main(OUT_DIR)


if __name__ == "__main__":
    main()
