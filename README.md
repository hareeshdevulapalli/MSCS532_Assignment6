# MSCS532 Assignment 6 - Medians, Order Statistics and Elementary Data Structures

**Student:** Hareesh Devulapalli | **Instructor:** Michael Solomon

Code for assignment 6. There are two parts:

1. **Selection** - finding the k-th smallest element in linear time, two ways:
   median of medians (worst-case O(n)) and quickselect with a random pivot (O(n) on average).
   Both use a three-way partition so duplicate values are handled properly.
2. **Data structures** - arrays, matrices, stacks, queues, a singly linked list and a rooted
   tree, all written by hand, plus timing comparisons between the array and linked versions.

The write-up with the analysis and results is in [REPORT.md](REPORT.md). I used CLRS (4th
edition), chapters 9 and 10, as the main reference.

## Files

- `selection.py` - `select_mom`, `select_random`, `select_last_pivot` (baseline), `median`,
  `partition3`, `WorkLog`
- `data_structures.py` - `ResizableArray`, `Matrix`, `ArrStack`, `RingQueue`, `LinkedList`,
  `NodeStack`, `NodeQueue`, `GeneralTree`
- `test_selection.py`, `test_data_structures.py` - unit tests (41 total)
- `benchmark.py` - all the experiments (A1-A4 for selection, B1-B6 for data structures)
- `make_charts.py`, `chart_style.py` - draw the figures from the CSV output
- `results/` - CSVs, figures and `log.txt` from my last full run

## Setup

Needs Python 3.8+. Everything except the charts uses only the standard library. For the charts:

```bash
pip install matplotlib
```

## Running

Tests:

```bash
python3 -m unittest -v test_selection test_data_structures
```

Small demos of each file:

```bash
python3 selection.py
```

```bash
python3 data_structures.py
```

Full benchmark (around 30-35 seconds on my laptop). Writes the CSVs, `log.txt` and the six
figures to `results/`:

```bash
python3 benchmark.py
```

Quick version with small sizes. This one goes to `results/small/` so it doesn't overwrite the
real results:

```bash
python3 benchmark.py --small
```

Redraw the figures from existing CSVs without re-running anything:

```bash
python3 make_charts.py
```

## Example

```python
import random
from selection import select_mom, select_random, median, WorkLog

nums = [7, 2, 9, 4, 4, 1, 8, 3, 6, 5, 4]
select_mom(nums, 3)                              # 3rd smallest -> 3
select_random(nums, 6, rng=random.Random(1))     # -> 4
median(nums)                                     # -> 4

log = WorkLog()
select_mom(list(range(100000)), 50000, log=log)
print(log)        # compares, touched, passes, max_depth
```

```python
from data_structures import ResizableArray, RingQueue, LinkedList, GeneralTree

a = ResizableArray([10, 20, 30]); a.insert(1, 15); a.remove_at(0)   # [15, 20, 30]
q = RingQueue(); q.enqueue("x"); q.dequeue()
ll = LinkedList([1, 2, 3]); ll.add_front(0); ll.reverse()
t = GeneralTree("root"); t.add_child(t.root, "child"); t.preorder()
```

## Main results

Selection (median of n elements):

- Both algorithms scaled linearly on every input type I tried (random, sorted, reversed, only 10
  distinct values), from n = 1,000 up to 200,000.
- Quickselect was usually 3-4.5x faster than median of medians (33.7 ms vs 147.9 ms at 200,000
  random elements) and used about half as many comparisons (~5n vs ~10n).
- Over 500 runs quickselect touched 3.377n elements on average. The theoretical value is
  2n(1 + ln 2) = 3.386n.
- Median of medians was much more consistent between runs (sd 0.64n vs 1.57n comparisons).
- A fixed last-element pivot is quadratic on sorted input: about 390x slower than quickselect at
  n = 4,000.
- Python's built-in `sorted()` was still faster than both, since it runs in C.

Data structures:

| test | array based | linked | result |
|---|---|---|---|
| stack push/pop | 121 ns/op | 130 ns/op | about the same |
| queue (ring buffer / linked / list.pop(0)) | 205 ns/op | 138 ns/op | pop(0) version: 6,104 ns/op and growing |
| 8,000 inserts at the front | 507 ms | 0.78 ms | linked ~650x faster |
| 8,000 random index lookups | 0.54 ms | 285 ms | array ~520x faster |
| append at the end | 68 ns/op | 105 ns/op | array faster |
| memory per item | 10.5 bytes | 48 bytes | array 4.6x smaller |
