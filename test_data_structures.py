# Tests for data_structures.py
# run:  python3 -m unittest -v test_data_structures

import random
import unittest

from data_structures import (ArrStack, GeneralTree, LinkedList, Matrix,
                             NodeQueue, NodeStack, ResizableArray, RingQueue)


class ResizableArrayTests(unittest.TestCase):

    def test_append_get_set(self):
        arr = ResizableArray()
        for i in range(100):
            arr.append(i * i)
        self.assertEqual(len(arr), 100)
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[99], 9801)
        self.assertEqual(arr[-1], 9801)
        self.assertGreaterEqual(arr.capacity, 100)
        # capacity should be a power of two since it only ever doubles
        self.assertEqual(arr.capacity & (arr.capacity - 1), 0)
        arr[5] = "five"
        self.assertEqual(arr[5], "five")

    def test_insert_remove_vs_python_list(self):
        r = random.Random(1)
        arr, ref = ResizableArray(), []
        for _ in range(2000):
            if ref and r.random() < 0.45:
                i = r.randrange(len(ref))
                self.assertEqual(arr.remove_at(i), ref.pop(i))
            else:
                i = r.randint(0, len(ref))
                v = r.randint(0, 99)
                arr.insert(i, v)
                ref.insert(i, v)
            self.assertEqual(list(arr), ref)

    def test_gives_memory_back(self):
        arr = ResizableArray(range(1024))
        before = arr.capacity
        while len(arr) > 3:
            arr.pop()
        self.assertLess(arr.capacity, before)
        self.assertEqual(list(arr), [0, 1, 2])

    def test_find_and_remove(self):
        arr = ResizableArray([4, 8, 15, 16, 23, 42])
        self.assertEqual(arr.find(16), 3)
        self.assertEqual(arr.find(99), -1)
        self.assertIn(23, arr)
        arr.remove(15)
        self.assertEqual(list(arr), [4, 8, 16, 23, 42])
        self.assertRaises(ValueError, arr.remove, 15)

    def test_errors(self):
        arr = ResizableArray([1, 2])
        self.assertRaises(IndexError, lambda: arr[2])
        self.assertRaises(IndexError, arr.insert, 5, 0)
        self.assertRaises(IndexError, arr.remove_at, -3)
        self.assertRaises(IndexError, ResizableArray().pop)


class MatrixTests(unittest.TestCase):

    def test_get_set(self):
        m = Matrix(2, 3)
        m[1, 2] = 9
        self.assertEqual(m[1, 2], 9)
        self.assertEqual(m.shape, (2, 3))
        self.assertRaises(IndexError, lambda: m[2, 0])

    def test_rows_and_columns(self):
        m = Matrix.from_rows([[1, 2], [3, 4]])
        m.add_row(1, [9, 9])
        self.assertEqual(m.as_rows(), [[1, 2], [9, 9], [3, 4]])
        m.add_col(2, [7, 7, 7])
        self.assertEqual(m.as_rows(), [[1, 2, 7], [9, 9, 7], [3, 4, 7]])
        self.assertEqual(m.drop_row(1), [9, 9, 7])
        self.assertEqual(m.drop_col(0), [1, 3])
        self.assertEqual(m.as_rows(), [[2, 7], [4, 7]])
        self.assertRaises(ValueError, m.add_row, 0, [1, 2, 3])

    def test_start_empty(self):
        m = Matrix(0, 0)
        m.add_row(0, [1, 2, 3])
        m.add_row(1, [4, 5, 6])
        self.assertEqual(m.as_rows(), [[1, 2, 3], [4, 5, 6]])

    def test_math(self):
        a = Matrix.from_rows([[1, 2, 3], [4, 5, 6]])
        b = Matrix.from_rows([[7, 8], [9, 10], [11, 12]])
        self.assertEqual((a @ b).as_rows(), [[58, 64], [139, 154]])
        self.assertEqual(a.transposed().as_rows(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual((a + a).as_rows(), [[2, 4, 6], [8, 10, 12]])
        self.assertEqual(a @ Matrix.identity(3), a)
        self.assertRaises(ValueError, lambda: a @ a)


class StackChecks:
    factory = None

    def test_last_in_first_out(self):
        s = self.factory()
        self.assertTrue(s.is_empty())
        for i in range(1000):
            s.push(i)
        self.assertEqual(len(s), 1000)
        self.assertEqual(s.peek(), 999)
        popped = [s.pop() for _ in range(1000)]
        self.assertEqual(popped, list(range(999, -1, -1)))
        self.assertTrue(s.is_empty())
        self.assertRaises(IndexError, s.pop)
        self.assertRaises(IndexError, s.peek)


class QueueChecks:
    factory = None

    def test_first_in_first_out(self):
        q = self.factory()
        ref = []
        r = random.Random(2)
        # mix enqueues and dequeues so the ring wraps around a lot
        for i in range(3000):
            if ref and r.random() < 0.5:
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
        self.assertRaises(IndexError, q.dequeue)
        self.assertRaises(IndexError, q.peek)


class ArrStackTests(StackChecks, unittest.TestCase):
    factory = ArrStack


class NodeStackTests(StackChecks, unittest.TestCase):
    factory = NodeStack


class RingQueueTests(QueueChecks, unittest.TestCase):
    factory = staticmethod(lambda: RingQueue(cap=2))

    def test_no_growth_when_not_needed(self):
        q = RingQueue(cap=4)
        for i in range(1000):
            q.enqueue(i)
            q.dequeue()
        self.assertEqual(len(q._slots), 4)


class NodeQueueTests(QueueChecks, unittest.TestCase):
    factory = NodeQueue


class LinkedListTests(unittest.TestCase):

    def test_both_ends(self):
        ll = LinkedList()
        ll.add_back(2)
        ll.add_front(1)
        ll.add_back(3)
        self.assertEqual(ll.to_list(), [1, 2, 3])
        self.assertEqual(ll.pop_back(), 3)
        self.assertEqual(ll.pop_front(), 1)
        self.assertEqual(ll.pop_back(), 2)
        self.assertTrue(ll.is_empty())
        self.assertIsNone(ll.head)
        self.assertIsNone(ll.tail)
        self.assertRaises(IndexError, ll.pop_front)

    def test_random_ops_vs_python_list(self):
        r = random.Random(3)
        ll, ref = LinkedList(), []
        for _ in range(2000):
            x = r.random()
            if ref and x < 0.3:
                i = r.randrange(len(ref))
                self.assertEqual(ll.remove_at(i), ref.pop(i))
            elif ref and x < 0.4:
                v = r.choice(ref)
                ll.remove_value(v)
                ref.remove(v)
            else:
                i = r.randint(0, len(ref))
                v = r.randint(0, 50)
                ll.insert_at(i, v)
                ref.insert(i, v)
            self.assertEqual(ll.to_list(), ref)
            self.assertEqual(len(ll), len(ref))
            if ref:   # tail pointer has to stay correct
                self.assertEqual(ll.tail.val, ref[-1])
                self.assertIsNone(ll.tail.nxt)

    def test_find_walk_reverse(self):
        ll = LinkedList("abcde")
        self.assertEqual(ll.find("c").val, "c")
        self.assertIsNone(ll.find("z"))
        self.assertEqual(ll.index_of("d"), 3)
        self.assertEqual(ll[-1], "e")
        seen = []
        ll.walk(seen.append)
        self.assertEqual(seen, list("abcde"))
        ll.reverse()
        self.assertEqual(ll.to_list(), list("edcba"))
        ll.add_back("!")
        self.assertEqual(ll.to_list(), list("edcba!"))
        ll.insert_after(ll.find("c"), "C")
        self.assertEqual(ll.to_list(), list("edcCba!"))
        self.assertRaises(ValueError, ll.remove_value, "zzz")


class TreeTests(unittest.TestCase):

    def setUp(self):
        #         A
        #       / | \
        #      B  C  D
        #     / \     \
        #    E   F     G
        #              |
        #              H
        t = GeneralTree("A")
        b = t.add_child(t.root, "B")
        t.add_child(t.root, "C")
        d = t.add_child(t.root, "D")
        t.add_child(b, "E")
        t.add_child(b, "F")
        g = t.add_child(d, "G")
        self.h = t.add_child(g, "H")
        self.t = t

    def test_orders(self):
        self.assertEqual(self.t.preorder(), list("ABEFCDGH"))
        self.assertEqual(self.t.bfs(), list("ABCDEFGH"))

    def test_sizes(self):
        self.assertEqual(len(self.t), 8)
        self.assertEqual(self.t.height(), 3)
        self.assertEqual(self.t.depth(self.h), 3)
        self.assertEqual([k.val for k in self.t.kids(self.t.root)], list("BCD"))
        self.assertTrue(self.t.is_leaf(self.h))

    def test_find_and_cut(self):
        d = self.t.find("D")
        self.assertEqual(self.t.cut(d), 3)
        self.assertEqual(self.t.preorder(), list("ABEFC"))
        self.assertEqual(len(self.t), 5)
        self.assertIsNone(self.t.find("H"))
        self.assertRaises(ValueError, self.t.cut, self.t.root)

    def test_very_deep_tree(self):
        t = GeneralTree(0)
        node = t.root
        for i in range(1, 5000):
            node = t.add_child(node, i)
        self.assertEqual(t.height(), 4999)
        self.assertEqual(len(t.preorder()), 5000)


if __name__ == "__main__":
    unittest.main()
