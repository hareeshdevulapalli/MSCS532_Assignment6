"""
Elementary data structures, written from scratch (CLRS chapter 10).

    DynamicArray       resizable array: access, insert, delete, search
    Matrix             2-D array stored row-major in one flat block
    ArrayStack         stack on top of a growable array
    ArrayQueue         queue on a circular buffer (ring) that grows
    SinglyLinkedList   head + tail pointers: insert, delete, search, traverse
    LinkedStack        stack built on linked nodes
    LinkedQueue        queue built on linked nodes
    RootedTree         general rooted tree, "left-child, right-sibling" form

"From scratch" here means none of these lean on Python's list methods to
do the real work. The array-based classes get a block of fixed size with
`[None] * capacity` (the closest thing Python has to a raw C array) and
then manage the size, the shifting and the resizing themselves. The only
things done on that block are reading and writing a slot by index. That
way the costs in the report are the costs of *this* code, not of CPython's
built-in list.

Indexes are 0-based. Errors follow Python's own conventions: IndexError
for a bad position or an empty structure, ValueError for "value not found".
"""


# ----------------------------------------------------------------------
# Arrays
# ----------------------------------------------------------------------

def _new_block(capacity):
    """A fixed-size block of `capacity` empty slots."""
    return [None] * capacity


class DynamicArray:
    """A growable array.

    Keeps `_size` items in a block of `_capacity` slots. When the block is
    full it is replaced by one twice as big (so n appends cost O(n) in
    total, O(1) amortized each). When it drops to a quarter full it is
    halved, so a long run of deletes gives the memory back without
    "thrashing" between growing and shrinking.

        access a[i] / a[i] = v     O(1)
        append(v)                  O(1) amortized
        insert(i, v)               O(n - i)   items after i shift right
        delete(i)                  O(n - i)   items after i shift left
        index_of(v)                O(n)
    """

    MIN_CAPACITY = 4

    def __init__(self, values=(), capacity=MIN_CAPACITY):
        self._capacity = max(self.MIN_CAPACITY, capacity)
        self._data = _new_block(self._capacity)
        self._size = 0
        for v in values:
            self.append(v)

    # -- basic protocol -------------------------------------------------
    def __len__(self):
        return self._size

    def _check(self, i):
        if not isinstance(i, int):
            raise TypeError("array index must be an int")
        if i < 0:
            i += self._size
        if not 0 <= i < self._size:
            raise IndexError("array index out of range")
        return i

    def __getitem__(self, i):
        return self._data[self._check(i)]

    def __setitem__(self, i, value):
        self._data[self._check(i)] = value

    def __iter__(self):
        for i in range(self._size):
            yield self._data[i]

    def __repr__(self):
        return f"DynamicArray({list(self)})"

    @property
    def capacity(self):
        return self._capacity

    # -- resizing -------------------------------------------------------
    def _resize(self, new_capacity):
        block = _new_block(new_capacity)
        for i in range(self._size):
            block[i] = self._data[i]
        self._data = block
        self._capacity = new_capacity

    def _maybe_shrink(self):
        if (self._capacity > self.MIN_CAPACITY
                and self._size <= self._capacity // 4):
            self._resize(max(self.MIN_CAPACITY, self._capacity // 2))

    # -- insertion ------------------------------------------------------
    def append(self, value):
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._data[self._size] = value
        self._size += 1

    def insert(self, i, value):
        """Put `value` at position i (0 <= i <= len), shifting the rest right."""
        if not 0 <= i <= self._size:
            raise IndexError("insert position out of range")
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        data = self._data
        for j in range(self._size, i, -1):     # walk from the back
            data[j] = data[j - 1]
        data[i] = value
        self._size += 1

    # -- deletion -------------------------------------------------------
    def delete(self, i):
        """Remove and return the item at position i, shifting the rest left."""
        i = self._check(i)
        data = self._data
        value = data[i]
        for j in range(i, self._size - 1):
            data[j] = data[j + 1]
        self._size -= 1
        data[self._size] = None                # don't keep a stale reference
        self._maybe_shrink()
        return value

    def pop(self):
        """Remove and return the last item, O(1) amortized."""
        if self._size == 0:
            raise IndexError("pop from empty array")
        return self.delete(self._size - 1)

    def remove(self, value):
        """Delete the first item equal to `value`."""
        i = self.index_of(value)
        if i < 0:
            raise ValueError(f"{value!r} not in array")
        self.delete(i)

    # -- search ---------------------------------------------------------
    def index_of(self, value):
        """Position of the first item equal to `value`, or -1."""
        for i in range(self._size):
            if self._data[i] == value:
                return i
        return -1

    def __contains__(self, value):
        return self.index_of(value) >= 0


class Matrix:
    """A rows x cols matrix stored row-major in one flat block.

    Item (r, c) lives at slot r * cols + c, so reading or writing any item
    is one multiply and one add: O(1). This is how C and NumPy lay out 2-D
    arrays. The downside is that changing the shape means rebuilding the
    block, so inserting or deleting a row or a column costs O(rows * cols).

        m[r, c] / m[r, c] = v               O(1)
        insert_row / delete_row             O(rows * cols)
        insert_col / delete_col             O(rows * cols)
        transpose, +                        O(rows * cols)
        @ (multiply n x m by m x p)         O(n * m * p)
    """

    def __init__(self, rows, cols, fill=0):
        if rows < 0 or cols < 0:
            raise ValueError("matrix dimensions must be non-negative")
        self.rows = rows
        self.cols = cols
        self._data = _new_block(rows * cols)
        for i in range(rows * cols):
            self._data[i] = fill

    @classmethod
    def from_lists(cls, rows):
        rows = [list(r) for r in rows]
        width = len(rows[0]) if rows else 0
        if any(len(r) != width for r in rows):
            raise ValueError("all rows must have the same length")
        m = cls(len(rows), width)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                m._data[r * width + c] = v
        return m

    @classmethod
    def identity(cls, n):
        m = cls(n, n, 0)
        for i in range(n):
            m[i, i] = 1
        return m

    def to_lists(self):
        return [[self._data[r * self.cols + c] for c in range(self.cols)]
                for r in range(self.rows)]

    @property
    def shape(self):
        return (self.rows, self.cols)

    # -- access ---------------------------------------------------------
    def _slot(self, key):
        r, c = key
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"matrix index {key} out of range for shape "
                             f"{self.shape}")
        return r * self.cols + c

    def __getitem__(self, key):
        return self._data[self._slot(key)]

    def __setitem__(self, key, value):
        self._data[self._slot(key)] = value

    def row(self, r):
        return [self[r, c] for c in range(self.cols)]

    def col(self, c):
        return [self[r, c] for r in range(self.rows)]

    def __eq__(self, other):
        return (isinstance(other, Matrix) and self.shape == other.shape
                and self.to_lists() == other.to_lists())

    def __repr__(self):
        return f"Matrix({self.to_lists()})"

    # -- shape changes --------------------------------------------------
    def _rebuild(self, rows, cols, source):
        """Make a new block of the given shape; source(r, c) gives each item."""
        block = _new_block(rows * cols)
        for r in range(rows):
            for c in range(cols):
                block[r * cols + c] = source(r, c)
        self.rows, self.cols, self._data = rows, cols, block

    def insert_row(self, i, values):
        if not 0 <= i <= self.rows:
            raise IndexError("row position out of range")
        values = list(values)
        if self.rows and len(values) != self.cols:
            raise ValueError(f"row must have {self.cols} items")
        if not self.rows and not self.cols:
            self.cols = len(values)
        old, cols = self._data, self.cols
        self._rebuild(self.rows + 1, cols,
                      lambda r, c: values[c] if r == i
                      else old[(r if r < i else r - 1) * cols + c])

    def delete_row(self, i):
        if not 0 <= i < self.rows:
            raise IndexError("row index out of range")
        removed = self.row(i)
        old, cols = self._data, self.cols
        self._rebuild(self.rows - 1, cols,
                      lambda r, c: old[(r if r < i else r + 1) * cols + c])
        return removed

    def insert_col(self, j, values):
        if not 0 <= j <= self.cols:
            raise IndexError("column position out of range")
        values = list(values)
        if self.cols and len(values) != self.rows:
            raise ValueError(f"column must have {self.rows} items")
        if not self.rows and not self.cols:
            self.rows = len(values)
        old, cols = self._data, self.cols
        self._rebuild(self.rows, cols + 1,
                      lambda r, c: values[r] if c == j
                      else old[r * cols + (c if c < j else c - 1)])

    def delete_col(self, j):
        if not 0 <= j < self.cols:
            raise IndexError("column index out of range")
        removed = self.col(j)
        old, cols = self._data, self.cols
        self._rebuild(self.rows, cols - 1,
                      lambda r, c: old[r * cols + (c if c < j else c + 1)])
        return removed

    # -- arithmetic -----------------------------------------------------
    def transpose(self):
        t = Matrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                t._data[c * self.rows + r] = self._data[r * self.cols + c]
        return t

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("matrix shapes don't match for +")
        out = Matrix(self.rows, self.cols)
        for i in range(self.rows * self.cols):
            out._data[i] = self._data[i] + other._data[i]
        return out

    def __matmul__(self, other):
        if self.cols != other.rows:
            raise ValueError(f"can't multiply {self.shape} by {other.shape}")
        n, m, p = self.rows, self.cols, other.cols
        out = Matrix(n, p)
        a, b, o = self._data, other._data, out._data
        for i in range(n):
            for k in range(m):          # i-k-j order walks b row by row
                aik = a[i * m + k]
                if aik == 0:
                    continue
                for j in range(p):
                    o[i * p + j] += aik * b[k * p + j]
        return out


# ----------------------------------------------------------------------
# Stacks and queues on arrays
# ----------------------------------------------------------------------

class ArrayStack:
    """LIFO stack on a growable array. The top is the last used slot.

        push, pop, peek     O(1) amortized (O(1) worst case except when
                            the block has to grow)
        is_empty, len       O(1)
    """

    def __init__(self, capacity=4):
        self._data = _new_block(max(1, capacity))
        self._top = 0                       # number of items = next free slot

    def __len__(self):
        return self._top

    def is_empty(self):
        return self._top == 0

    def push(self, value):
        if self._top == len(self._data):
            block = _new_block(2 * len(self._data))
            for i in range(self._top):
                block[i] = self._data[i]
            self._data = block
        self._data[self._top] = value
        self._top += 1

    def pop(self):
        if self._top == 0:
            raise IndexError("pop from empty stack")
        self._top -= 1
        value = self._data[self._top]
        self._data[self._top] = None
        return value

    def peek(self):
        if self._top == 0:
            raise IndexError("peek at empty stack")
        return self._data[self._top - 1]

    def __repr__(self):
        items = [self._data[i] for i in range(self._top)]
        return f"ArrayStack(bottom->top {items})"


class ArrayQueue:
    """FIFO queue on a circular buffer.

    `_head` is where the next item will be taken from; the tail is
    (_head + _size) mod capacity. Both ends just move forward and wrap
    around, so no item ever has to be shifted. That is the whole point:
    the obvious "list.pop(0)" queue shifts every remaining item on every
    dequeue and costs O(n) per operation.

        enqueue             O(1) amortized (grows by doubling when full)
        dequeue, peek       O(1)
        is_empty, len       O(1)
    """

    def __init__(self, capacity=4):
        self._data = _new_block(max(1, capacity))
        self._head = 0
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def _grow(self):
        old, cap = self._data, len(self._data)
        block = _new_block(2 * cap)
        for i in range(self._size):        # unwrap so the head is at 0
            block[i] = old[(self._head + i) % cap]
        self._data = block
        self._head = 0

    def enqueue(self, value):
        if self._size == len(self._data):
            self._grow()
        self._data[(self._head + self._size) % len(self._data)] = value
        self._size += 1

    def dequeue(self):
        if self._size == 0:
            raise IndexError("dequeue from empty queue")
        value = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % len(self._data)
        self._size -= 1
        return value

    def peek(self):
        if self._size == 0:
            raise IndexError("peek at empty queue")
        return self._data[self._head]

    def __iter__(self):
        cap = len(self._data)
        for i in range(self._size):
            yield self._data[(self._head + i) % cap]

    def __repr__(self):
        return f"ArrayQueue(front->back {list(self)})"


# ----------------------------------------------------------------------
# Linked lists
# ----------------------------------------------------------------------

class Node:
    """One cell of a singly linked list. __slots__ keeps each node small."""

    __slots__ = ("value", "next")

    def __init__(self, value, next=None):
        self.value = value
        self.next = next

    def __repr__(self):
        return f"Node({self.value!r})"


class SinglyLinkedList:
    """Singly linked list with head and tail pointers.

    Keeping a tail pointer makes adding at the back O(1). Removing from
    the back is still O(n): to unhook the last node we need the one before
    it, and the only way to reach it is to walk from the head.

        push_front, pop_front           O(1)
        push_back                       O(1)
        pop_back                        O(n)
        insert_after(node, v)           O(1)
        insert(i, v), delete(i), get(i) O(i)
        find(v), remove(v)              O(n)
        traversal, reverse              O(n)
    """

    def __init__(self, values=()):
        self.head = None
        self.tail = None
        self._size = 0
        for v in values:
            self.push_back(v)

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    # -- traversal ------------------------------------------------------
    def __iter__(self):
        node = self.head
        while node is not None:
            yield node.value
            node = node.next

    def traverse(self, visit):
        """Call visit(value) on every item from head to tail."""
        node = self.head
        while node is not None:
            visit(node.value)
            node = node.next

    def to_list(self):
        return list(self)

    def __repr__(self):
        return "SinglyLinkedList(" + " -> ".join(map(repr, self)) + ")"

    def _node_at(self, i):
        if not 0 <= i < self._size:
            raise IndexError("list index out of range")
        node = self.head
        for _ in range(i):
            node = node.next
        return node

    def get(self, i):
        return self._node_at(i).value

    def __getitem__(self, i):
        if i < 0:
            i += self._size
        return self._node_at(i).value

    # -- insertion ------------------------------------------------------
    def push_front(self, value):
        self.head = Node(value, self.head)
        if self.tail is None:
            self.tail = self.head
        self._size += 1

    def push_back(self, value):
        node = Node(value)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self._size += 1

    def insert_after(self, node, value):
        """Splice a new node in right after `node`. O(1)."""
        new = Node(value, node.next)
        node.next = new
        if node is self.tail:
            self.tail = new
        self._size += 1
        return new

    def insert(self, i, value):
        """Insert so the new item ends up at position i (0 <= i <= len)."""
        if not 0 <= i <= self._size:
            raise IndexError("insert position out of range")
        if i == 0:
            self.push_front(value)
        elif i == self._size:
            self.push_back(value)
        else:
            self.insert_after(self._node_at(i - 1), value)

    # -- deletion -------------------------------------------------------
    def pop_front(self):
        if self.head is None:
            raise IndexError("pop from empty list")
        node = self.head
        self.head = node.next
        if self.head is None:
            self.tail = None
        self._size -= 1
        return node.value

    def pop_back(self):
        if self.head is None:
            raise IndexError("pop from empty list")
        if self.head is self.tail:
            return self.pop_front()
        prev = self.head
        while prev.next is not self.tail:   # the O(n) walk
            prev = prev.next
        value = self.tail.value
        prev.next = None
        self.tail = prev
        self._size -= 1
        return value

    def delete(self, i):
        """Remove and return the item at position i."""
        if not 0 <= i < self._size:
            raise IndexError("list index out of range")
        if i == 0:
            return self.pop_front()
        prev = self._node_at(i - 1)
        node = prev.next
        prev.next = node.next
        if node is self.tail:
            self.tail = prev
        self._size -= 1
        return node.value

    def remove(self, value):
        """Delete the first node holding `value`. Raises ValueError if none."""
        prev, node = None, self.head
        while node is not None and node.value != value:
            prev, node = node, node.next
        if node is None:
            raise ValueError(f"{value!r} not in list")
        if prev is None:
            self.head = node.next
        else:
            prev.next = node.next
        if node is self.tail:
            self.tail = prev
        self._size -= 1

    # -- search and misc ------------------------------------------------
    def find(self, value):
        """First node holding `value`, or None."""
        node = self.head
        while node is not None:
            if node.value == value:
                return node
            node = node.next
        return None

    def index_of(self, value):
        for i, v in enumerate(self):
            if v == value:
                return i
        return -1

    def __contains__(self, value):
        return self.find(value) is not None

    def reverse(self):
        """Reverse the list in place by flipping every `next` pointer."""
        prev, node = None, self.head
        self.tail = node
        while node is not None:
            node.next, prev, node = prev, node, node.next
        self.head = prev


class LinkedStack:
    """LIFO stack on linked nodes. The top is the head.

    push, pop, peek are O(1) in the worst case: there is never a resize.
    The price is one node object per item.
    """

    def __init__(self):
        self._head = None
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._head is None

    def push(self, value):
        self._head = Node(value, self._head)
        self._size += 1

    def pop(self):
        if self._head is None:
            raise IndexError("pop from empty stack")
        node = self._head
        self._head = node.next
        self._size -= 1
        return node.value

    def peek(self):
        if self._head is None:
            raise IndexError("peek at empty stack")
        return self._head.value


class LinkedQueue:
    """FIFO queue on linked nodes: enqueue at the tail, dequeue at the head.

    Both are O(1) worst case. (Doing it the other way round would make
    dequeue O(n), for the same reason pop_back is O(n) above.)
    """

    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._head is None

    def enqueue(self, value):
        node = Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def dequeue(self):
        if self._head is None:
            raise IndexError("dequeue from empty queue")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return node.value

    def peek(self):
        if self._head is None:
            raise IndexError("peek at empty queue")
        return self._head.value


# ----------------------------------------------------------------------
# Rooted trees
# ----------------------------------------------------------------------

class TreeNode:
    """A node in the left-child, right-sibling representation (CLRS 10.3).

    Each node has exactly three pointers however many children it has:
    its parent, its first (leftmost) child, and its next sibling. The
    children of a node form a singly linked list through `right_sibling`.
    """

    __slots__ = ("value", "parent", "left_child", "right_sibling")

    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.left_child = None
        self.right_sibling = None

    def __repr__(self):
        return f"TreeNode({self.value!r})"


class RootedTree:
    """A rooted tree where a node can have any number of children.

        add_child(parent, v)     O(number of children of parent),
                                 because it walks to the end of the
                                 sibling list to keep children in order
        children(node)           O(number of children)
        depth(node)              O(depth)
        height(), size, find()   O(n)
        preorder / level_order   O(n)
        remove_subtree(node)     O(number of siblings + size of subtree)
    """

    def __init__(self, root_value):
        self.root = TreeNode(root_value)
        self._size = 1

    def __len__(self):
        return self._size

    def add_child(self, parent, value):
        child = TreeNode(value, parent)
        if parent.left_child is None:
            parent.left_child = child
        else:
            last = parent.left_child
            while last.right_sibling is not None:
                last = last.right_sibling
            last.right_sibling = child
        self._size += 1
        return child

    def children(self, node):
        child = node.left_child
        while child is not None:
            yield child
            child = child.right_sibling

    def is_leaf(self, node):
        return node.left_child is None

    def depth(self, node):
        d = 0
        while node.parent is not None:
            node = node.parent
            d += 1
        return d

    def height(self, node=None):
        """Number of edges on the longest path from `node` down to a leaf."""
        node = self.root if node is None else node
        best = 0
        stack = ArrayStack()
        stack.push((node, 0))
        while not stack.is_empty():
            current, d = stack.pop()
            if d > best:
                best = d
            for child in self.children(current):
                stack.push((child, d + 1))
        return best

    def preorder(self, node=None):
        """Values in preorder (node, then each child's subtree, left to right).

        Done with an explicit stack instead of recursion so a very deep tree
        can't hit Python's recursion limit.
        """
        node = self.root if node is None else node
        out = []
        stack = ArrayStack()
        stack.push(node)
        while not stack.is_empty():
            current = stack.pop()
            out.append(current.value)
            kids = list(self.children(current))
            for child in reversed(kids):    # so the leftmost comes off first
                stack.push(child)
        return out

    def level_order(self):
        """Values level by level (breadth-first), using LinkedQueue."""
        out = []
        queue = LinkedQueue()
        queue.enqueue(self.root)
        while not queue.is_empty():
            current = queue.dequeue()
            out.append(current.value)
            for child in self.children(current):
                queue.enqueue(child)
        return out

    def find(self, value):
        queue = LinkedQueue()
        queue.enqueue(self.root)
        while not queue.is_empty():
            current = queue.dequeue()
            if current.value == value:
                return current
            for child in self.children(current):
                queue.enqueue(child)
        return None

    def remove_subtree(self, node):
        """Detach `node` and everything below it. The root can't be removed."""
        if node is self.root:
            raise ValueError("cannot remove the root")
        parent = node.parent
        if parent.left_child is node:
            parent.left_child = node.right_sibling
        else:
            prev = parent.left_child
            while prev.right_sibling is not node:
                prev = prev.right_sibling
            prev.right_sibling = node.right_sibling
        removed = len(self.preorder(node))
        node.parent = node.right_sibling = None
        self._size -= removed
        return removed


if __name__ == "__main__":
    a = DynamicArray([10, 20, 30])
    a.insert(1, 15)
    a.delete(0)
    print(a, "capacity", a.capacity)

    m = Matrix.from_lists([[1, 2], [3, 4]])
    m.insert_row(2, [5, 6])
    m.insert_col(0, [0, 0, 0])
    print(m, "shape", m.shape)
    print("transpose", m.transpose())

    s = ArrayStack()
    for x in "abc":
        s.push(x)
    print(s)
    print("pop ->", s.pop(), "peek ->", s.peek())

    q = ArrayQueue(capacity=2)
    for x in range(5):
        q.enqueue(x)
    q.dequeue()
    print(q)

    ll = SinglyLinkedList([1, 2, 3])
    ll.push_front(0)
    ll.insert(2, 1.5)
    ll.remove(3)
    ll.reverse()
    print(ll)

    t = RootedTree("CEO")
    cto = t.add_child(t.root, "CTO")
    cfo = t.add_child(t.root, "CFO")
    t.add_child(cto, "Dev lead")
    t.add_child(cto, "QA lead")
    t.add_child(cfo, "Accountant")
    print("preorder   ", t.preorder())
    print("level order", t.level_order())
    print("height", t.height(), "size", len(t))
