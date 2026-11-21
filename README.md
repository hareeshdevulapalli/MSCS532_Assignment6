# MSCS532 Assignment 6: Medians, Order Statistics & Elementary Data Structures

This repo has:

* **Part 1:** two linear-time selection algorithms for finding the *k*-th smallest item:
  * **median of medians** (deterministic, O(*n*) worst case)
  * **randomized quickselect** (O(*n*) expected)

  Both use a three-way partition so they handle duplicates correctly. The repo also has a
  benchmark that compares them on random, sorted, reverse-sorted and few-unique inputs.
* **Part 2:** arrays, matrices, stacks, queues, singly linked lists and rooted trees, all
  written from scratch. A benchmark compares array-based and linked versions.

The algorithms follow *Introduction to Algorithms* (Cormen et al., 4th ed., ch. 9 and 10). The
full write-up is in **[REPORT.md](REPORT.md)**. It covers the design choices, the complexity
proofs, the space and overhead analysis, the empirical results, the array-vs-linked trade-offs
and real-world applications.

| File | What's in it |
|---|---|
| `selection.py` | `deterministic_select`, `randomized_select`, `fixed_pivot_select` (baseline), `median`, `three_way_partition`, `SelectTracker` |
| `data_structures.py` | `DynamicArray`, `Matrix`, `ArrayStack`, `ArrayQueue`, `SinglyLinkedList`, `LinkedStack`, `LinkedQueue`, `RootedTree` |
| `test_selection.py` | 20 unit tests for Part 1 |
| `test_data_structures.py` | 21 unit tests for Part 2 |
| `run_benchmarks.py` | All experiments (S1–S4 selection, D1–D6 data structures) |
| `plot_results.py` / `plot_theme.py` | Draw the charts in `results/` from the saved CSVs |
| `results/` | CSV data, charts and `benchmark_output.txt` from the last full run |

## Requirements

Python 3.8 or newer. The algorithms, data structures and tests only use the standard library.
The charts need `matplotlib`:

```bash
pip install matplotlib
```

## How to run it

Run the tests:

```bash
python3 -m unittest -v test_selection test_data_structures
```

Run the small demos built into each module:

```bash
python3 selection.py
```

```bash
python3 data_structures.py
```

Run the full benchmark (about 35 seconds). It writes the CSV files, `benchmark_output.txt` and
all the charts into `results/`:

```bash
python3 run_benchmarks.py
```

Or run a quick version with small sizes, done in a few seconds. It writes to `results/quick/`
(git-ignored), so it never overwrites the full results:

```bash
python3 run_benchmarks.py --quick
```

Redraw the charts from the existing CSV files without rerunning the benchmark:

```bash
python3 plot_results.py
```

### Using the code

```python
import random
from selection import deterministic_select, randomized_select, median, SelectTracker

data = [7, 2, 9, 4, 4, 1, 8, 3, 6, 5, 4]
deterministic_select(data, 3)                    # 3rd smallest -> 3 (k is 1-based)
randomized_select(data, 6, rng=random.Random(1)) # -> 4, same pivots every run
median(data)                                     # lower median -> 4

t = SelectTracker()
deterministic_select(list(range(100_000)), 50_000, tracker=t)
print(t)   # comparisons, items examined, partition passes, recursion depth
```

```python
from data_structures import DynamicArray, ArrayQueue, SinglyLinkedList, RootedTree

a = DynamicArray([10, 20, 30]); a.insert(1, 15); a.delete(0)   # [15, 20, 30]
q = ArrayQueue(); q.enqueue("x"); q.dequeue()
ll = SinglyLinkedList([1, 2, 3]); ll.push_front(0); ll.reverse()
t = RootedTree("root"); child = t.add_child(t.root, "child"); t.preorder()
```

## Summary of findings

**Part 1: selection** (median of *n* items, Python 3.13, Apple Silicon)

* **Both algorithms are linear on every input.** Time per item and comparisons per item stay
  flat from *n* = 1,000 to 200,000 on random, sorted, reverse-sorted and few-unique inputs.
* **Quickselect is 3.0–4.5× faster** (35.8 ms vs 124.7 ms for 200,000 random items) and makes about
  half the comparisons: ~5*n* vs ~10*n*. Median of medians pays for its worst-case guarantee with
  a large constant factor.
* **Theory matches measurement.** Over 500 runs, quickselect examined 3.403*n* items on average,
  against Knuth's predicted 2*n*(1 + ln 2) = 3.386*n*. Its cost also follows his formula as *k*
  changes. Median of medians costs ~3.2*n* for every *k*.
* **Guarantee vs average.** Median of medians is very consistent (sd 0.48*n* comparisons).
  Quickselect varies 3.3× more, and its worst run ≈ a typical deterministic run.
* **Randomizing the pivot matters.** A fixed last-item pivot on sorted input is quadratic
  (exactly 3*n*²/8 items examined), ~410× slower than quickselect at *n* = 4,000.
* **Duplicates.** The three-way partition keeps the worst-case proof valid when values repeat,
  and made few-unique input the *fastest* case.
* **Constant factors are real.** C's O(*n* log *n*) `sorted()` still beat both pure-Python O(*n*)
  selections at every size tested.

**Part 2: data structures**

| Scenario | Array-based | Linked | Winner |
|---|---|---|---|
| Stack push/pop | 121 ns/op | 132 ns/op | tie (both O(1)) |
| Queue: ring buffer vs linked vs `list.pop(0)` | 205 ns/op | 139 ns/op | linked/ring (naive `pop(0)`: 6,311 ns/op, O(*n*)) |
| 8,000 inserts at the front | 525 ms | 0.77 ms | linked, ~690× |
| 8,000 reads at random positions | 0.48 ms | 118 ms | array, ~250× |
| Appends at the back | 66 ns/op | 103 ns/op | array |
| Memory per item | 10.5 B | 48 B | array, 4.6× smaller |

In real Python code, use `list` for stacks and `collections.deque` for queues. They're the
array-stack and block-linked-queue designs from this assignment, written in C.
