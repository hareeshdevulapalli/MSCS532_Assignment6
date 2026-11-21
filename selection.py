"""
Finding the k-th smallest item (order statistics) without sorting.

    deterministic_select(items, k)   median of medians, O(n) in the worst case
    randomized_select(items, k)      quickselect with a random pivot,
                                     O(n) expected time
    fixed_pivot_select(items, k)     quickselect that always uses the last
                                     item as the pivot. Only here as a
                                     baseline for the benchmark: it is
                                     O(n^2) on sorted input.
    median(items)                    lower median, using either of the above

I followed SELECT and RANDOMIZED-SELECT from CLRS chapter 9 (4th edition).
Like the book, k is 1-based: k = 1 is the smallest item, k = len(items) is
the largest. Inside the code all indexes are 0-based and ranges are
inclusive, so a[lo..hi] means a[lo] up to and including a[hi].

Duplicates
----------
Every version splits the range three ways, into  < pivot, == pivot and
> pivot  (Dijkstra's "Dutch national flag" partition). Two reasons:

  * If k lands inside the == block we can stop right away. On a list with
    only a handful of distinct values this ends the search after one or
    two passes.
  * The worst-case proof for median of medians needs "at least ~3n/10
    items are on each side of the pivot". With a two-way partition and lots
    of copies of the pivot value, all the copies can pile up on one side
    and the guarantee breaks (an all-equal list would go quadratic). With
    the three-way split, the copies go in the middle block and the proof
    holds as written.

The only operation used on the values is <, so ints, floats, strings,
tuples etc. all work.

By default every function works on a copy and leaves your list alone. Pass
copy=False to let it rearrange your list in place instead (saves the O(n)
copy).

You can pass a SelectTracker to count comparisons, partition passes and
recursion depth. The benchmark uses this for the "work done" charts.
"""

import random


class SelectTracker:
    """Optional counters a selection fills in while it runs.

    comparisons      every < between two items: the ones inside the
                     partition step plus the ones used to sort the groups
                     of 5 (median of medians only)
    examined         total number of items looked at by partition passes.
                     For quickselect this is the textbook "cost" measure,
                     one per item per pass.
    partitions       how many partition passes ran
    deepest          deepest level of real recursion reached (top call = 0).
                     Quickselect is written as a loop so it stays at 0.
    """

    __slots__ = ("comparisons", "examined", "partitions", "deepest")

    def __init__(self):
        self.comparisons = 0
        self.examined = 0
        self.partitions = 0
        self.deepest = 0

    def __repr__(self):
        return (f"SelectTracker(comparisons={self.comparisons}, "
                f"examined={self.examined}, partitions={self.partitions}, "
                f"deepest={self.deepest})")


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------

def three_way_partition(a, lo, hi, pivot):
    """Rearrange a[lo..hi] around the value `pivot`.

    Returns (lt, gt, comparisons) so that afterwards
        a[lo..lt-1]   < pivot
        a[lt..gt]     == pivot   (empty if pivot isn't in the range)
        a[gt+1..hi]   > pivot

    While the loop runs, with `i` as the current position:
        a[lo..lt-1]   < pivot
        a[lt..i-1]    == pivot
        a[i..gt]      not checked yet
        a[gt+1..hi]   > pivot
    Each item is checked exactly once. An item smaller than the pivot costs
    one comparison and any other item costs two, so the count is
    2m - (number smaller), with m = hi - lo + 1. Working it out at the end
    means the loop doesn't need a counter.
    """
    lt, i, gt = lo, lo, hi
    while i <= gt:
        value = a[i]
        if value < pivot:
            a[lt], a[i] = value, a[lt]
            lt += 1
            i += 1
        elif pivot < value:
            a[i], a[gt] = a[gt], value
            gt -= 1
        else:
            i += 1
    m = hi - lo + 1
    return lt, gt, 2 * m - (lt - lo)


def insertion_sort_range(a, lo, hi):
    """Sort a[lo..hi] in place and return how many comparisons it took.

    Only used on groups of at most 5 items, where it does at most 10
    comparisons, so the whole thing is O(1) per group.
    """
    comparisons = 0
    for i in range(lo + 1, hi + 1):
        value = a[i]
        j = i - 1
        while j >= lo:
            comparisons += 1
            if value < a[j]:
                a[j + 1] = a[j]
                j -= 1
            else:
                break
        a[j + 1] = value
    return comparisons


def _prepare(items, k, copy):
    """Shared argument checks. Returns the list to work on."""
    n = len(items)
    if n == 0:
        raise ValueError("cannot select from an empty sequence")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    if copy or not isinstance(items, list):
        return list(items)
    return items


# ----------------------------------------------------------------------
# Deterministic: median of medians
# ----------------------------------------------------------------------

def _mom_select(a, lo, hi, target, tracker, level):
    """Return the item that belongs at index `target` of sorted a[lo..hi].

    The outer `while` replaces the "recurse into one side" step with a
    loop. The only real recursion is the call that finds the median of
    the group medians, and that call is on a list 5 times smaller, so the
    stack never gets deeper than about log_5(n).
    """
    if tracker is not None and level > tracker.deepest:
        tracker.deepest = level
    comparisons = 0
    while True:
        n = hi - lo + 1
        if n <= 5:
            comparisons += insertion_sort_range(a, lo, hi)
            if tracker is not None:
                tracker.comparisons += comparisons
            return a[target]

        # 1. Split into groups of 5 (the last one may be shorter), sort
        #    each group, and move its median to the front of the range.
        #    `slot` never passes `start`, so we only ever overwrite spots
        #    in groups we've already finished with.
        slot = lo
        for start in range(lo, hi + 1, 5):
            end = start + 4 if start + 4 <= hi else hi
            comparisons += insertion_sort_range(a, start, end)
            mid = (start + end) // 2
            a[slot], a[mid] = a[mid], a[slot]
            slot += 1

        # 2. Recursively find the median of the medians in a[lo..slot-1].
        count = slot - lo
        if tracker is not None:
            tracker.comparisons += comparisons
        comparisons = 0
        pivot = _mom_select(a, lo, slot - 1, lo + (count - 1) // 2,
                            tracker, level + 1)

        # 3. Partition the whole range around it and keep the side that
        #    holds `target`.
        lt, gt, c = three_way_partition(a, lo, hi, pivot)
        comparisons += c
        if tracker is not None:
            tracker.examined += n
            tracker.partitions += 1
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            if tracker is not None:
                tracker.comparisons += comparisons
            return pivot


def deterministic_select(items, k, tracker=None, copy=True):
    """Return the k-th smallest item (k = 1 is the minimum), worst-case O(n).

    This is the median-of-medians algorithm (Blum, Floyd, Pratt, Rivest and
    Tarjan, 1973). The pivot is chosen so that at least ~3n/10 items are
    smaller than it and at least ~3n/10 are bigger, so every pass throws
    away a fixed fraction of the list whatever the input looks like.
    """
    a = _prepare(items, k, copy)
    return _mom_select(a, 0, len(a) - 1, k - 1, tracker, 0)


# ----------------------------------------------------------------------
# Randomized: quickselect
# ----------------------------------------------------------------------

def _quickselect(a, target, choose_pivot, tracker):
    lo, hi = 0, len(a) - 1
    comparisons = examined = partitions = 0
    while lo < hi:
        pivot = a[choose_pivot(lo, hi)]
        lt, gt, c = three_way_partition(a, lo, hi, pivot)
        comparisons += c
        examined += hi - lo + 1
        partitions += 1
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            break
    if tracker is not None:
        tracker.comparisons += comparisons
        tracker.examined += examined
        tracker.partitions += partitions
    return a[target]


def randomized_select(items, k, rng=None, tracker=None, copy=True):
    """Return the k-th smallest item using quickselect with a random pivot.

    Expected O(n) time on every input, because the pivot is random and not
    something an input can be arranged against. Worst case is still
    O(n^2), but you'd need to be extremely unlucky (see REPORT.md, 2.2).

    Pass rng=random.Random(seed) to get the same pivots every run.
    """
    a = _prepare(items, k, copy)
    randint = (rng if rng is not None else random).randint
    return _quickselect(a, k - 1, randint, tracker)


def fixed_pivot_select(items, k, tracker=None, copy=True):
    """Quickselect that always takes the last item as the pivot.

    Not one of the required algorithms; it's here so the benchmark can show
    what randomizing the pivot buys. On sorted or reverse-sorted input each
    pass only removes one item, so finding the median costs ~n^2/2 work.
    """
    a = _prepare(items, k, copy)
    return _quickselect(a, k - 1, lambda lo, hi: hi, tracker)


def median(items, method="deterministic", **kwargs):
    """Lower median of `items` (the ceil(n/2)-th smallest, like CLRS)."""
    pick = {
        "deterministic": deterministic_select,
        "randomized": randomized_select,
    }[method]
    return pick(items, (len(items) + 1) // 2, **kwargs)


if __name__ == "__main__":
    data = [7, 2, 9, 4, 4, 1, 8, 3, 6, 5, 4]
    print("data:", data)
    print("sorted:", sorted(data))
    for k in (1, 3, 6, len(data)):
        d = deterministic_select(data, k)
        r = randomized_select(data, k, rng=random.Random(1))
        print(f"k={k:2d}  deterministic={d}  randomized={r}")
    print("lower median:", median(data))

    big = list(range(100_000))
    random.Random(532).shuffle(big)
    for name, fn in (("deterministic", deterministic_select),
                     ("randomized", randomized_select)):
        t = SelectTracker()
        value = fn(big, 50_000, tracker=t)
        print(f"{name:13s} median of 0..99999 -> {value}   {t}")
