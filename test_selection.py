"""Unit tests for selection.py. Run with:  python3 -m unittest -v test_selection"""

import math
import random
import unittest

from selection import (SelectTracker, deterministic_select, fixed_pivot_select,
                       insertion_sort_range, median, randomized_select,
                       three_way_partition)

ALL = (
    ("deterministic", deterministic_select),
    ("randomized", randomized_select),
    ("fixed pivot", fixed_pivot_select),
)


def every_k_matches_sorted(test, data):
    expected = sorted(data)
    for name, fn in ALL:
        for k in range(1, len(data) + 1):
            with test.subTest(algo=name, n=len(data), k=k):
                test.assertEqual(fn(data, k), expected[k - 1])


class TestSelectionCorrectness(unittest.TestCase):

    def test_small_hand_checked(self):
        data = [7, 2, 9, 4, 4, 1, 8, 3, 6, 5, 4]
        for _, fn in ALL:
            self.assertEqual(fn(data, 1), 1)
            self.assertEqual(fn(data, 6), 4)
            self.assertEqual(fn(data, 11), 9)

    def test_single_item(self):
        every_k_matches_sorted(self, [42])

    def test_every_k_on_random_lists(self):
        rng = random.Random(1)
        for n in (2, 3, 5, 6, 7, 10, 24, 25, 26, 51, 125, 200):
            every_k_matches_sorted(self, [rng.randint(-1000, 1000)
                                          for _ in range(n)])

    def test_heavy_duplicates(self):
        rng = random.Random(2)
        for n in (10, 57, 300):
            every_k_matches_sorted(self, [rng.randint(0, 3) for _ in range(n)])

    def test_all_equal(self):
        every_k_matches_sorted(self, [7] * 103)

    def test_sorted_and_reverse_sorted(self):
        every_k_matches_sorted(self, list(range(130)))
        every_k_matches_sorted(self, list(range(130, 0, -1)))

    def test_floats_strings_tuples(self):
        every_k_matches_sorted(self, [3.5, -1.25, 2.0, 2.0, 1e9, -1e-9])
        every_k_matches_sorted(self, list("selectionalgorithms"))
        every_k_matches_sorted(self, [(2, "b"), (1, "z"), (2, "a"), (0, "q")])

    def test_larger_random_inputs(self):
        rng = random.Random(3)
        for n in (1_000, 10_001):
            data = [rng.random() for _ in range(n)]
            expected = sorted(data)
            for k in (1, 2, n // 3, (n + 1) // 2, n - 1, n):
                self.assertEqual(deterministic_select(data, k), expected[k - 1])
                self.assertEqual(randomized_select(data, k), expected[k - 1])

    def test_accepts_tuples_and_ranges(self):
        self.assertEqual(deterministic_select((5, 1, 3), 2), 3)
        self.assertEqual(randomized_select(range(10, 0, -1), 4), 4)


class TestSelectionBehaviour(unittest.TestCase):

    def test_input_not_changed_by_default(self):
        data = [5, 3, 9, 1, 7, 3, 8, 2, 6]
        before = data[:]
        for _, fn in ALL:
            fn(data, 4)
            self.assertEqual(data, before)

    def test_copy_false_works_in_place(self):
        data = list(range(50, 0, -1))
        self.assertEqual(randomized_select(data, 10, copy=False), 10)
        self.assertEqual(sorted(data), list(range(1, 51)))   # same items
        self.assertEqual(data[9], 10)      # the answer ends up at index k-1

    def test_bad_arguments(self):
        for _, fn in ALL:
            with self.assertRaises(ValueError):
                fn([], 1)
            with self.assertRaises(ValueError):
                fn([1, 2, 3], 0)
            with self.assertRaises(ValueError):
                fn([1, 2, 3], 4)
            with self.assertRaises(TypeError):
                fn([1, 2, 3], 1.0)
            with self.assertRaises(TypeError):
                fn([1, 2, 3], True)

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2)          # lower median
        self.assertEqual(median([4, 1, 3, 2], method="randomized"), 2)

    def test_seeded_randomized_is_repeatable(self):
        data = list(range(5000))
        random.Random(9).shuffle(data)
        t1, t2 = SelectTracker(), SelectTracker()
        randomized_select(data, 2500, rng=random.Random(4), tracker=t1)
        randomized_select(data, 2500, rng=random.Random(4), tracker=t2)
        self.assertEqual(repr(t1), repr(t2))


class TestWorstCaseBounds(unittest.TestCase):
    """Checks that the numbers behave the way the analysis says they should."""

    def test_deterministic_work_is_linear_on_every_input_shape(self):
        n = 20_000
        rng = random.Random(5)
        shapes = {
            "random": [rng.random() for _ in range(n)],
            "sorted": list(range(n)),
            "reverse": list(range(n, 0, -1)),
            "few unique": [rng.randint(0, 4) for _ in range(n)],
            "all equal": [1] * n,
        }
        for name, data in shapes.items():
            t = SelectTracker()
            deterministic_select(data, n // 2, tracker=t)
            with self.subTest(shape=name):
                # the proof gives a constant; in practice it's well under 20
                self.assertLess(t.comparisons, 20 * n)

    def test_deterministic_recursion_depth_is_logarithmic(self):
        n = 100_000
        t = SelectTracker()
        deterministic_select(list(range(n)), n // 2, tracker=t)
        self.assertLessEqual(t.deepest, math.ceil(math.log(n, 5)) + 1)

    def test_fixed_pivot_is_quadratic_on_sorted_input(self):
        n = 2_000
        t = SelectTracker()
        fixed_pivot_select(list(range(n)), n // 2, tracker=t)
        self.assertGreater(t.examined, n * n // 4)

    def test_randomized_is_linear_on_sorted_input(self):
        n = 20_000
        t = SelectTracker()
        randomized_select(list(range(n)), n // 2, rng=random.Random(6),
                          tracker=t)
        self.assertLess(t.examined, 10 * n)


class TestHelpers(unittest.TestCase):

    def test_three_way_partition(self):
        rng = random.Random(7)
        for _ in range(200):
            a = [rng.randint(0, 6) for _ in range(rng.randint(1, 30))]
            pivot = rng.randint(-1, 7)
            original = a[:]
            lt, gt, comps = three_way_partition(a, 0, len(a) - 1, pivot)
            self.assertEqual(sorted(a), sorted(original))
            self.assertTrue(all(x < pivot for x in a[:lt]))
            self.assertTrue(all(x == pivot for x in a[lt:gt + 1]))
            self.assertTrue(all(x > pivot for x in a[gt + 1:]))
            less = sum(x < pivot for x in a)
            self.assertEqual(comps, 2 * len(a) - less)

    def test_insertion_sort_range_only_touches_range(self):
        a = [9, 5, 4, 3, 2, 1, 0]
        insertion_sort_range(a, 1, 5)
        self.assertEqual(a, [9, 1, 2, 3, 4, 5, 0])
        self.assertLessEqual(insertion_sort_range([5, 4, 3, 2, 1], 0, 4), 10)


if __name__ == "__main__":
    unittest.main()
