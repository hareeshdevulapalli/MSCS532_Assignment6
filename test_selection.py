# Tests for selection.py
# run:  python3 -m unittest -v test_selection

import math
import random
import unittest

from selection import (WorkLog, median, partition3, select_last_pivot,
                       select_mom, select_random, small_sort)

FUNCS = [
    ("mom", select_mom),
    ("random", select_random),
    ("last pivot", select_last_pivot),
]


def check_all_k(tc, nums):
    """Every k for every function should match sorted(nums)[k-1]."""
    want = sorted(nums)
    for name, fn in FUNCS:
        for k in range(1, len(nums) + 1):
            with tc.subTest(fn=name, n=len(nums), k=k):
                tc.assertEqual(fn(nums, k), want[k - 1])


class Correctness(unittest.TestCase):

    def test_small_example(self):
        nums = [7, 2, 9, 4, 4, 1, 8, 3, 6, 5, 4]
        for _, fn in FUNCS:
            self.assertEqual(fn(nums, 1), 1)
            self.assertEqual(fn(nums, 6), 4)
            self.assertEqual(fn(nums, 11), 9)

    def test_one_element(self):
        check_all_k(self, [42])

    def test_random_lists_of_many_sizes(self):
        r = random.Random(1)
        # 5/6 and 25/26 are where the group-of-5 logic changes
        for n in (2, 3, 5, 6, 7, 10, 24, 25, 26, 51, 125, 200):
            check_all_k(self, [r.randint(-1000, 1000) for _ in range(n)])

    def test_lots_of_duplicates(self):
        r = random.Random(2)
        for n in (10, 57, 300):
            check_all_k(self, [r.randint(0, 3) for _ in range(n)])

    def test_every_value_the_same(self):
        check_all_k(self, [7] * 103)

    def test_already_sorted_and_reversed(self):
        check_all_k(self, list(range(130)))
        check_all_k(self, list(range(130, 0, -1)))

    def test_other_types(self):
        check_all_k(self, [3.5, -1.25, 2.0, 2.0, 1e9, -1e-9])
        check_all_k(self, list("selectionalgorithms"))
        check_all_k(self, [(2, "b"), (1, "z"), (2, "a"), (0, "q")])

    def test_bigger_inputs(self):
        r = random.Random(3)
        for n in (1000, 10001):
            nums = [r.random() for _ in range(n)]
            s = sorted(nums)
            for k in (1, 2, n // 3, (n + 1) // 2, n - 1, n):
                self.assertEqual(select_mom(nums, k), s[k - 1])
                self.assertEqual(select_random(nums, k), s[k - 1])

    def test_tuple_and_range_input(self):
        self.assertEqual(select_mom((5, 1, 3), 2), 3)
        self.assertEqual(select_random(range(10, 0, -1), 4), 4)


class Behaviour(unittest.TestCase):

    def test_original_list_left_alone(self):
        nums = [5, 3, 9, 1, 7, 3, 8, 2, 6]
        copy_before = nums[:]
        for _, fn in FUNCS:
            fn(nums, 4)
            self.assertEqual(nums, copy_before)

    def test_in_place_mode(self):
        nums = list(range(50, 0, -1))
        self.assertEqual(select_random(nums, 10, copy=False), 10)
        self.assertEqual(sorted(nums), list(range(1, 51)))
        self.assertEqual(nums[9], 10)

    def test_bad_input(self):
        for _, fn in FUNCS:
            self.assertRaises(ValueError, fn, [], 1)
            self.assertRaises(ValueError, fn, [1, 2, 3], 0)
            self.assertRaises(ValueError, fn, [1, 2, 3], 4)
            self.assertRaises(TypeError, fn, [1, 2, 3], 1.0)
            self.assertRaises(TypeError, fn, [1, 2, 3], True)

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2)   # lower median
        self.assertEqual(median([4, 1, 3, 2], method="random"), 2)
        self.assertRaises(ValueError, median, [1, 2], method="nope")

    def test_same_seed_same_work(self):
        nums = list(range(5000))
        random.Random(9).shuffle(nums)
        a, b = WorkLog(), WorkLog()
        select_random(nums, 2500, rng=random.Random(4), log=a)
        select_random(nums, 2500, rng=random.Random(4), log=b)
        self.assertEqual(repr(a), repr(b))


class Bounds(unittest.TestCase):
    """Sanity checks that the work grows the way the analysis says."""

    def test_mom_linear_on_all_shapes(self):
        n = 20000
        r = random.Random(5)
        inputs = {
            "random": [r.random() for _ in range(n)],
            "sorted": list(range(n)),
            "reversed": list(range(n, 0, -1)),
            "few values": [r.randint(0, 4) for _ in range(n)],
            "all same": [1] * n,
        }
        for label, nums in inputs.items():
            log = WorkLog()
            select_mom(nums, n // 2, log=log)
            with self.subTest(input=label):
                self.assertLess(log.compares, 20 * n)

    def test_mom_recursion_is_shallow(self):
        n = 100000
        log = WorkLog()
        select_mom(list(range(n)), n // 2, log=log)
        self.assertLessEqual(log.max_depth, math.ceil(math.log(n, 5)) + 1)

    def test_last_pivot_blows_up_on_sorted(self):
        n = 2000
        log = WorkLog()
        select_last_pivot(list(range(n)), n // 2, log=log)
        self.assertGreater(log.touched, n * n // 4)

    def test_random_pivot_fine_on_sorted(self):
        n = 20000
        log = WorkLog()
        select_random(list(range(n)), n // 2, rng=random.Random(6), log=log)
        self.assertLess(log.touched, 10 * n)


class Helpers(unittest.TestCase):

    def test_partition3(self):
        r = random.Random(7)
        for _ in range(200):
            arr = [r.randint(0, 6) for _ in range(r.randint(1, 30))]
            pv = r.randint(-1, 7)
            before = arr[:]
            a, b, cmps = partition3(arr, 0, len(arr) - 1, pv)
            self.assertEqual(sorted(arr), sorted(before))
            self.assertTrue(all(x < pv for x in arr[:a]))
            self.assertTrue(all(x == pv for x in arr[a:b + 1]))
            self.assertTrue(all(x > pv for x in arr[b + 1:]))
            n_less = sum(1 for x in arr if x < pv)
            self.assertEqual(cmps, 2 * len(arr) - n_less)

    def test_small_sort_stays_in_range(self):
        arr = [9, 5, 4, 3, 2, 1, 0]
        small_sort(arr, 1, 5)
        self.assertEqual(arr, [9, 1, 2, 3, 4, 5, 0])
        self.assertLessEqual(small_sort([5, 4, 3, 2, 1], 0, 4), 10)


if __name__ == "__main__":
    unittest.main()
