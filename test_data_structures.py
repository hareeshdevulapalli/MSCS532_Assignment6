"""Unit tests for data_structures.py.  python3 -m unittest -v test_data_structures"""

import random
import unittest

from data_structures import (ArrayQueue, ArrayStack, DynamicArray, LinkedQueue,
                             LinkedStack, Matrix, RootedTree, SinglyLinkedList)


class TestDynamicArray(unittest.TestCase):

    def test_append_access_and_growth(self):
        a = DynamicArray()
        for i in range(100):
            a.append(i * i)
        self.assertEqual(len(a), 100)
        self.assertEqual(a[0], 0)
        self.assertEqual(a[99], 99 * 99)
        self.assertEqual(a[-1], 99 * 99)
        self.assertGreaterEqual(a.capacity, 100)
        self.assertEqual(a.capacity & (a.capacity - 1), 0)  # grew by doubling
        a[5] = "five"
        self.assertEqual(a[5], "five")

    def test_insert_and_delete_match_python_list(self):
        rng = random.Random(1)
        a, ref = DynamicArray(), []
        for _ in range(2000):
            if ref and rng.random() < 0.45:
                i = rng.randrange(len(ref))
                self.assertEqual(a.delete(i), ref.pop(i))
            else:
                i = rng.randint(0, len(ref))
                v = rng.randint(0, 99)
                a.insert(i, v)
                ref.insert(i, v)
            self.assertEqual(list(a), ref)

    def test_shrinks_after_many_deletes(self):
        a = DynamicArray(range(1024))
        big = a.capacity
        while len(a) > 3:
            a.pop()
        self.assertLess(a.capacity, big)
        self.assertEqual(list(a), [0, 1, 2])

    def test_search_and_remove(self):
        a = DynamicArray([4, 8, 15, 16, 23, 42])
        self.assertEqual(a.index_of(16), 3)
        self.assertEqual(a.index_of(99), -1)
        self.assertIn(23, a)
        a.remove(15)
        self.assertEqual(list(a), [4, 8, 16, 23, 42])
        with self.assertRaises(ValueError):
            a.remove(15)

    def test_errors(self):
        a = DynamicArray([1, 2])
        with self.assertRaises(IndexError):
            a[2]
        with self.assertRaises(IndexError):
            a.insert(5, 0)
        with self.assertRaises(IndexError):
            a.delete(-3)
        with self.assertRaises(IndexError):
            DynamicArray().pop()


class TestMatrix(unittest.TestCase):

    def test_access_and_shape(self):
        m = Matrix(2, 3, fill=0)
        m[1, 2] = 9
        self.assertEqual(m[1, 2], 9)
        self.assertEqual(m.shape, (2, 3))
        with self.assertRaises(IndexError):
            m[2, 0]

    def test_row_and_column_insert_delete(self):
        m = Matrix.from_lists([[1, 2], [3, 4]])
        m.insert_row(1, [9, 9])
        self.assertEqual(m.to_lists(), [[1, 2], [9, 9], [3, 4]])
        m.insert_col(2, [7, 7, 7])
        self.assertEqual(m.to_lists(), [[1, 2, 7], [9, 9, 7], [3, 4, 7]])
        self.assertEqual(m.delete_row(1), [9, 9, 7])
        self.assertEqual(m.delete_col(0), [1, 3])
        self.assertEqual(m.to_lists(), [[2, 7], [4, 7]])
        with self.assertRaises(ValueError):
            m.insert_row(0, [1, 2, 3])

    def test_build_from_empty(self):
        m = Matrix(0, 0)
        m.insert_row(0, [1, 2, 3])
        m.insert_row(1, [4, 5, 6])
        self.assertEqual(m.to_lists(), [[1, 2, 3], [4, 5, 6]])

    def test_arithmetic(self):
        a = Matrix.from_lists([[1, 2, 3], [4, 5, 6]])
        b = Matrix.from_lists([[7, 8], [9, 10], [11, 12]])
        self.assertEqual((a @ b).to_lists(), [[58, 64], [139, 154]])
        self.assertEqual(a.transpose().to_lists(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual((a + a).to_lists(), [[2, 4, 6], [8, 10, 12]])
        self.assertEqual(a @ Matrix.identity(3), a)
        with self.assertRaises(ValueError):
            a @ a


class StackContract:
    make = None

    def test_lifo(self):
        s = self.make()
        self.assertTrue(s.is_empty())
        for i in range(1000):
            s.push(i)
        self.assertEqual(len(s), 1000)
        self.assertEqual(s.peek(), 999)
        self.assertEqual([s.pop() for _ in range(1000)], list(range(999, -1, -1)))
        self.assertTrue(s.is_empty())
        with self.assertRaises(IndexError):
            s.pop()
        with self.assertRaises(IndexError):
            s.peek()


class QueueContract:
    make = None

    def test_fifo_with_wraparound(self):
        q = self.make()
        ref = []
        rng = random.Random(2)
        for i in range(3000):        # mixed ops so the ring wraps many times
            if ref and rng.random() < 0.5:
                self.assertEqual(q.dequeue(), ref.pop(0))
            else:
                q.enqueue(i)
                ref.append(i)
            self.assertEqual(len(q), len(ref))
            if ref:
                self.assertEqual(q.peek(), ref[0])
        while ref:
            self.assertEqual(q.dequeue(), ref.pop(0))
        self.assertTrue(q.is_empty())
        with self.assertRaises(IndexError):
            q.dequeue()
        with self.assertRaises(IndexError):
            q.peek()


class TestArrayStack(StackContract, unittest.TestCase):
    make = ArrayStack


class TestLinkedStack(StackContract, unittest.TestCase):
    make = LinkedStack


class TestArrayQueue(QueueContract, unittest.TestCase):
    make = staticmethod(lambda: ArrayQueue(capacity=2))

    def test_ring_does_not_grow_when_it_does_not_need_to(self):
        q = ArrayQueue(capacity=4)
        for i in range(1000):
            q.enqueue(i)
            q.dequeue()
        self.assertEqual(len(q._data), 4)


class TestLinkedQueue(QueueContract, unittest.TestCase):
    make = LinkedQueue


class TestSinglyLinkedList(unittest.TestCase):

    def test_front_and_back(self):
        ll = SinglyLinkedList()
        ll.push_back(2)
        ll.push_front(1)
        ll.push_back(3)
        self.assertEqual(ll.to_list(), [1, 2, 3])
        self.assertEqual(ll.pop_back(), 3)
        self.assertEqual(ll.pop_front(), 1)
        self.assertEqual(ll.pop_back(), 2)
        self.assertTrue(ll.is_empty())
        self.assertIsNone(ll.head)
        self.assertIsNone(ll.tail)
        with self.assertRaises(IndexError):
            ll.pop_front()

    def test_random_ops_match_python_list(self):
        rng = random.Random(3)
        ll, ref = SinglyLinkedList(), []
        for _ in range(2000):
            r = rng.random()
            if ref and r < 0.3:
                i = rng.randrange(len(ref))
                self.assertEqual(ll.delete(i), ref.pop(i))
            elif ref and r < 0.4:
                v = rng.choice(ref)
                ll.remove(v)
                ref.remove(v)
            else:
                i = rng.randint(0, len(ref))
                v = rng.randint(0, 50)
                ll.insert(i, v)
                ref.insert(i, v)
            self.assertEqual(ll.to_list(), ref)
            self.assertEqual(len(ll), len(ref))
            # the tail pointer must always be right
            if ref:
                self.assertEqual(ll.tail.value, ref[-1])
                self.assertIsNone(ll.tail.next)

    def test_search_traverse_reverse(self):
        ll = SinglyLinkedList("abcde")
        self.assertEqual(ll.find("c").value, "c")
        self.assertIsNone(ll.find("z"))
        self.assertEqual(ll.index_of("d"), 3)
        self.assertEqual(ll[-1], "e")
        seen = []
        ll.traverse(seen.append)
        self.assertEqual(seen, list("abcde"))
        ll.reverse()
        self.assertEqual(ll.to_list(), list("edcba"))
        ll.push_back("!")
        self.assertEqual(ll.to_list(), list("edcba!"))
        node = ll.find("c")
        ll.insert_after(node, "C")
        self.assertEqual(ll.to_list(), list("edcCba!"))
        with self.assertRaises(ValueError):
            ll.remove("zzz")


class TestRootedTree(unittest.TestCase):

    def setUp(self):
        #        A
        #      / | \
        #     B  C  D
        #    / \     \
        #   E   F     G
        #             |
        #             H
        self.t = RootedTree("A")
        r = self.t.root
        b = self.t.add_child(r, "B")
        self.t.add_child(r, "C")
        d = self.t.add_child(r, "D")
        self.t.add_child(b, "E")
        self.t.add_child(b, "F")
        g = self.t.add_child(d, "G")
        self.h = self.t.add_child(g, "H")

    def test_traversals(self):
        self.assertEqual(self.t.preorder(), list("ABEFCDGH"))
        self.assertEqual(self.t.level_order(), list("ABCDEFGH"))

    def test_measures(self):
        self.assertEqual(len(self.t), 8)
        self.assertEqual(self.t.height(), 3)
        self.assertEqual(self.t.depth(self.h), 3)
        self.assertEqual([c.value for c in self.t.children(self.t.root)],
                         list("BCD"))
        self.assertTrue(self.t.is_leaf(self.h))

    def test_find_and_remove(self):
        d = self.t.find("D")
        self.assertEqual(self.t.remove_subtree(d), 3)
        self.assertEqual(self.t.preorder(), list("ABEFC"))
        self.assertEqual(len(self.t), 5)
        self.assertIsNone(self.t.find("H"))
        with self.assertRaises(ValueError):
            self.t.remove_subtree(self.t.root)

    def test_deep_tree_does_not_hit_recursion_limit(self):
        t = RootedTree(0)
        node = t.root
        for i in range(1, 5000):
            node = t.add_child(node, i)
        self.assertEqual(t.height(), 4999)
        self.assertEqual(len(t.preorder()), 5000)


if __name__ == "__main__":
    unittest.main()
