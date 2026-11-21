"""
selection.py - find the k-th smallest value of a list without sorting it.

Functions:
    select_mom(seq, k)          median of medians (BFPRT). Linear even in the
                                worst case.
    select_random(seq, k)       quickselect with a random pivot. Linear on
                                average.
    select_last_pivot(seq, k)   quickselect that always picks the last value.
                                I only use this one in the benchmarks to show
                                why a fixed pivot is a bad idea (sorted input
                                makes it quadratic).
    median(seq)                 lower median, using one of the two above

k starts at 1 (k=1 -> smallest, k=len(seq) -> largest), the same way CLRS
numbers order statistics in chapter 9. Indexes in the code are 0-based and
the ranges [left, right] include both ends.

A note on duplicates: every version splits the range into three parts
(less than / equal to / greater than the pivot). If k falls in the "equal"
part we're done right away. It also keeps the median-of-medians proof
valid when the list has repeated values. With a normal two-way split, a list
where every value is the same would go quadratic.

By default the functions copy the input first so your list isn't touched.
copy=False skips the copy and works directly on the list you pass in.

Passing a WorkLog lets you see how much work was done (comparisons, how
many elements the partition step went over, number of passes, and how deep
the recursion went). The benchmark uses it.
"""

import random


class WorkLog:
    """Counters filled in while a selection runs (all optional).

    compares   number of `<` checks between elements (partition + the small
               sorts of 5-element groups in median of medians)
    touched    elements gone over by partition passes, added up. This is
               the usual cost measure for quickselect.
    passes     how many partition passes there were
    max_depth  deepest recursion level reached. The top call is level 0.
               Quickselect is a loop, so for it this stays 0.
    """

    __slots__ = ("compares", "touched", "passes", "max_depth")

    def __init__(self):
        self.compares = 0
        self.touched = 0
        self.passes = 0
        self.max_depth = 0

    def __repr__(self):
        return ("WorkLog(compares=%d, touched=%d, passes=%d, max_depth=%d)"
                % (self.compares, self.touched, self.passes, self.max_depth))


# --- helpers ---------------------------------------------------------------

def partition3(arr, left, right, pivot):
    """Three-way partition of arr[left..right] around the value `pivot`.

    Gives back (eq_start, eq_end, compares). After it returns:
        arr[left .. eq_start-1]    are < pivot
        arr[eq_start .. eq_end]    are == pivot (can be empty)
        arr[eq_end+1 .. right]     are > pivot

    lo_ptr, scan and hi_ptr split the range into four parts while the loop
    runs: smaller | equal | still unseen | bigger. Each value gets looked at
    once. A smaller value costs 1 comparison and anything else costs 2, so I
    can work out the total at the end instead of counting inside the loop:
    2*size - (number of smaller values).
    """
    lo_ptr = scan = left
    hi_ptr = right
    while scan <= hi_ptr:
        cur = arr[scan]
        if cur < pivot:
            arr[lo_ptr], arr[scan] = cur, arr[lo_ptr]
            lo_ptr += 1
            scan += 1
        elif pivot < cur:
            arr[scan], arr[hi_ptr] = arr[hi_ptr], cur
            hi_ptr -= 1
        else:
            scan += 1
    size = right - left + 1
    return lo_ptr, hi_ptr, 2 * size - (lo_ptr - left)


def small_sort(arr, left, right):
    """Insertion sort on arr[left..right]. Returns the number of compares.

    Only ever called on groups of 5 or fewer, so it's constant work
    (10 compares max).
    """
    count = 0
    for i in range(left + 1, right + 1):
        item = arr[i]
        j = i - 1
        while j >= left:
            count += 1
            if item < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            else:
                break
        arr[j + 1] = item
    return count


def _check_args(seq, k, copy):
    n = len(seq)
    if n == 0:
        raise ValueError("can't select from an empty list")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k has to be an int, got " + type(k).__name__)
    if k < 1 or k > n:
        raise ValueError("k should be in 1..%d but was %d" % (n, k))
    if copy or not isinstance(seq, list):
        return list(seq)
    return seq


# --- median of medians -----------------------------------------------------

def _mom(arr, left, right, want, log, depth):
    """Value that would be at index `want` if arr[left..right] were sorted.

    The "go into one side" step is just the while loop, so the only real
    recursive call is the one that picks the pivot. That call works on a
    range 5 times smaller, which keeps the call stack around log5(n) deep.
    """
    if log is not None and depth > log.max_depth:
        log.max_depth = depth
    cmp_count = 0
    while True:
        size = right - left + 1
        if size <= 5:
            cmp_count += small_sort(arr, left, right)
            if log is not None:
                log.compares += cmp_count
            return arr[want]

        # sort every group of 5 and swap its middle value up to the front.
        # `dest` grows by 1 per group and `g_start` by 5, so dest never gets
        # ahead of g_start and we only overwrite groups we already handled.
        dest = left
        for g_start in range(left, right + 1, 5):
            g_end = min(g_start + 4, right)
            cmp_count += small_sort(arr, g_start, g_end)
            g_mid = (g_start + g_end) // 2
            arr[dest], arr[g_mid] = arr[g_mid], arr[dest]
            dest += 1

        # median of the group medians (they sit in arr[left..dest-1] now)
        n_meds = dest - left
        if log is not None:
            log.compares += cmp_count
        cmp_count = 0
        pivot = _mom(arr, left, dest - 1, left + (n_meds - 1) // 2,
                     log, depth + 1)

        eq_start, eq_end, c = partition3(arr, left, right, pivot)
        cmp_count += c
        if log is not None:
            log.touched += size
            log.passes += 1

        if want < eq_start:
            right = eq_start - 1
        elif want > eq_end:
            left = eq_end + 1
        else:
            if log is not None:
                log.compares += cmp_count
            return pivot


def select_mom(seq, k, log=None, copy=True):
    """k-th smallest value (k=1 is the min) in worst-case linear time.

    Blum, Floyd, Pratt, Rivest & Tarjan (1973). The pivot always has at
    least about 3n/10 values on each side of it, so every pass gets rid of
    a fixed share of the list no matter what the input looks like.
    """
    arr = _check_args(seq, k, copy)
    return _mom(arr, 0, len(arr) - 1, k - 1, log, 0)


# --- quickselect -----------------------------------------------------------

def _qselect(arr, want, pick_pivot, log):
    left, right = 0, len(arr) - 1
    cmp_total = seen = rounds = 0
    while left < right:
        pivot = arr[pick_pivot(left, right)]
        eq_start, eq_end, c = partition3(arr, left, right, pivot)
        cmp_total += c
        seen += right - left + 1
        rounds += 1
        if want < eq_start:
            right = eq_start - 1
        elif want > eq_end:
            left = eq_end + 1
        else:
            break
    if log is not None:
        log.compares += cmp_total
        log.touched += seen
        log.passes += rounds
    return arr[want]


def select_random(seq, k, rng=None, log=None, copy=True):
    """k-th smallest value using quickselect with a random pivot.

    Expected linear time for every input, since the pivot comes from the
    random generator and not from the data. The worst case is still
    quadratic but you would need an absurd run of bad luck to hit it.

    Give it rng=random.Random(seed) if you want the same pivots every time.
    """
    arr = _check_args(seq, k, copy)
    randint = (rng or random).randint
    return _qselect(arr, k - 1, randint, log)


def select_last_pivot(seq, k, log=None, copy=True):
    """Quickselect that always uses the last value as pivot (baseline only).

    On sorted or reverse sorted input each pass only removes one element,
    so finding the median takes about n^2/2 work.
    """
    arr = _check_args(seq, k, copy)
    return _qselect(arr, k - 1, lambda l, r: r, log)


def median(seq, method="mom", **kw):
    """Lower median, i.e. the ceil(n/2)-th smallest value (CLRS convention)."""
    if method == "mom":
        fn = select_mom
    elif method == "random":
        fn = select_random
    else:
        raise ValueError("method must be 'mom' or 'random'")
    return fn(seq, (len(seq) + 1) // 2, **kw)


if __name__ == "__main__":
    nums = [7, 2, 9, 4, 4, 1, 8, 3, 6, 5, 4]
    print("list:  ", nums)
    print("sorted:", sorted(nums))
    for k in (1, 3, 6, len(nums)):
        a = select_mom(nums, k)
        b = select_random(nums, k, rng=random.Random(1))
        print("k=%-2d  mom=%d  random=%d" % (k, a, b))
    print("median:", median(nums))

    big = list(range(100000))
    random.Random(532).shuffle(big)
    for label, fn in (("mom", select_mom), ("random", select_random)):
        log = WorkLog()
        val = fn(big, 50000, log=log)
        print("%-6s -> %d   %r" % (label, val, log))
