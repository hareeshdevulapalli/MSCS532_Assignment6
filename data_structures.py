"""
data_structures.py - basic data structures from CLRS chapter 10, written
by hand.

    ResizableArray   array that grows/shrinks (get, set, insert, delete, find)
    Matrix           2D grid kept row by row in one flat block
    ArrStack         stack on a growable array
    RingQueue        queue on a circular buffer
    LinkedList       singly linked list with head and tail
    NodeStack        stack made of linked nodes
    NodeQueue        queue made of linked nodes
    GeneralTree      rooted tree, any number of children per node
                     (left-child / right-sibling layout)

I didn't want Python's list doing the real work, so the array classes only
ask for a fixed block once (`[None] * cap`, which is about as close as Python
gets to a plain C array) and after that they only read and write single
slots. Growing, shifting and wrapping around is all done in this file. That
way when I say insert is O(n) you can actually see the loop that does it.

Indexes start at 0. Bad index / empty structure -> IndexError, value not
there -> ValueError, same as Python's built-ins.
"""


def _block(cap):
    # fixed size chunk of empty slots
    return [None] * cap


# ===========================================================================
# arrays
# ===========================================================================

class ResizableArray:
    """Array that resizes itself.

    `_count` items live in a block with `_cap` slots. Full block -> make one
    twice as big (n appends cost O(n) total). Only a quarter used -> halve
    it, so memory gets freed but we don't keep bouncing between sizes.

    get/set O(1), append O(1) amortized, insert/delete at i O(n-i),
    find O(n)
    """

    SMALLEST = 4

    def __init__(self, items=(), cap=SMALLEST):
        self._cap = max(self.SMALLEST, cap)
        self._slots = _block(self._cap)
        self._count = 0
        for x in items:
            self.append(x)

    def __len__(self):
        return self._count

    def _fix_index(self, i):
        if not isinstance(i, int):
            raise TypeError("index must be an int")
        if i < 0:
            i += self._count
        if i < 0 or i >= self._count:
            raise IndexError("index out of range")
        return i

    def __getitem__(self, i):
        return self._slots[self._fix_index(i)]

    def __setitem__(self, i, val):
        self._slots[self._fix_index(i)] = val

    def __iter__(self):
        for i in range(self._count):
            yield self._slots[i]

    def __repr__(self):
        return "ResizableArray(%s)" % list(self)

    @property
    def capacity(self):
        return self._cap

    def _resize(self, new_cap):
        fresh = _block(new_cap)
        for i in range(self._count):
            fresh[i] = self._slots[i]
        self._slots = fresh
        self._cap = new_cap

    def _shrink_if_sparse(self):
        if self._cap > self.SMALLEST and self._count <= self._cap // 4:
            self._resize(max(self.SMALLEST, self._cap // 2))

    def append(self, val):
        if self._count == self._cap:
            self._resize(self._cap * 2)
        self._slots[self._count] = val
        self._count += 1

    def insert(self, pos, val):
        """Put val at index pos (0..len) and push everything after it right."""
        if pos < 0 or pos > self._count:
            raise IndexError("insert position out of range")
        if self._count == self._cap:
            self._resize(self._cap * 2)
        s = self._slots
        # go backwards so nothing gets overwritten
        for j in range(self._count, pos, -1):
            s[j] = s[j - 1]
        s[pos] = val
        self._count += 1

    def remove_at(self, pos):
        """Take out the item at pos and close the gap. Returns the item."""
        pos = self._fix_index(pos)
        s = self._slots
        out = s[pos]
        for j in range(pos, self._count - 1):
            s[j] = s[j + 1]
        self._count -= 1
        s[self._count] = None      # drop the old reference
        self._shrink_if_sparse()
        return out

    def pop(self):
        if self._count == 0:
            raise IndexError("pop from an empty array")
        return self.remove_at(self._count - 1)

    def remove(self, val):
        """Delete the first item equal to val."""
        pos = self.find(val)
        if pos == -1:
            raise ValueError("%r is not in the array" % (val,))
        self.remove_at(pos)

    def find(self, val):
        """Index of the first match, -1 if not found (linear scan)."""
        for i in range(self._count):
            if self._slots[i] == val:
                return i
        return -1

    def __contains__(self, val):
        return self.find(val) != -1


class Matrix:
    """n_rows x n_cols matrix stored row-major in one flat block.

    Entry (r, c) is slot r*n_cols + c, so reading/writing an entry is O(1).
    That's how C and NumPy store 2D arrays. The catch is that adding or
    dropping a row or column means rebuilding the whole block: O(rows*cols).
    Multiplying (n x m) by (m x p) is O(n*m*p).
    """

    def __init__(self, n_rows, n_cols, fill=0):
        if n_rows < 0 or n_cols < 0:
            raise ValueError("matrix size can't be negative")
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._cells = _block(n_rows * n_cols)
        for i in range(n_rows * n_cols):
            self._cells[i] = fill

    @classmethod
    def from_rows(cls, rows):
        rows = [list(r) for r in rows]
        width = len(rows[0]) if rows else 0
        for r in rows:
            if len(r) != width:
                raise ValueError("rows have different lengths")
        m = cls(len(rows), width)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                m._cells[r * width + c] = v
        return m

    @classmethod
    def identity(cls, n):
        m = cls(n, n, 0)
        for i in range(n):
            m[i, i] = 1
        return m

    def as_rows(self):
        w = self.n_cols
        return [[self._cells[r * w + c] for c in range(w)]
                for r in range(self.n_rows)]

    @property
    def shape(self):
        return (self.n_rows, self.n_cols)

    def _where(self, rc):
        r, c = rc
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            raise IndexError("%s is outside a %dx%d matrix"
                             % (rc, self.n_rows, self.n_cols))
        return r * self.n_cols + c

    def __getitem__(self, rc):
        return self._cells[self._where(rc)]

    def __setitem__(self, rc, val):
        self._cells[self._where(rc)] = val

    def get_row(self, r):
        return [self[r, c] for c in range(self.n_cols)]

    def get_col(self, c):
        return [self[r, c] for r in range(self.n_rows)]

    def __eq__(self, other):
        return (isinstance(other, Matrix) and self.shape == other.shape
                and self.as_rows() == other.as_rows())

    def __repr__(self):
        return "Matrix(%s)" % self.as_rows()

    def _rebuild(self, rows, cols, value_at):
        # new block of the new shape, value_at(r, c) says what goes where
        fresh = _block(rows * cols)
        for r in range(rows):
            for c in range(cols):
                fresh[r * cols + c] = value_at(r, c)
        self.n_rows, self.n_cols, self._cells = rows, cols, fresh

    def add_row(self, at, vals):
        if at < 0 or at > self.n_rows:
            raise IndexError("row position out of range")
        vals = list(vals)
        if self.n_rows and len(vals) != self.n_cols:
            raise ValueError("row needs %d values" % self.n_cols)
        if self.n_rows == 0 and self.n_cols == 0:
            self.n_cols = len(vals)
        old, w = self._cells, self.n_cols

        def pick(r, c):
            if r == at:
                return vals[c]
            src = r if r < at else r - 1
            return old[src * w + c]
        self._rebuild(self.n_rows + 1, w, pick)

    def drop_row(self, at):
        if at < 0 or at >= self.n_rows:
            raise IndexError("row index out of range")
        gone = self.get_row(at)
        old, w = self._cells, self.n_cols
        self._rebuild(self.n_rows - 1, w,
                      lambda r, c: old[(r if r < at else r + 1) * w + c])
        return gone

    def add_col(self, at, vals):
        if at < 0 or at > self.n_cols:
            raise IndexError("column position out of range")
        vals = list(vals)
        if self.n_cols and len(vals) != self.n_rows:
            raise ValueError("column needs %d values" % self.n_rows)
        if self.n_rows == 0 and self.n_cols == 0:
            self.n_rows = len(vals)
        old, w = self._cells, self.n_cols

        def pick(r, c):
            if c == at:
                return vals[r]
            src = c if c < at else c - 1
            return old[r * w + src]
        self._rebuild(self.n_rows, w + 1, pick)

    def drop_col(self, at):
        if at < 0 or at >= self.n_cols:
            raise IndexError("column index out of range")
        gone = self.get_col(at)
        old, w = self._cells, self.n_cols
        self._rebuild(self.n_rows, w - 1,
                      lambda r, c: old[r * w + (c if c < at else c + 1)])
        return gone

    def transposed(self):
        t = Matrix(self.n_cols, self.n_rows)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                t._cells[c * self.n_rows + r] = self._cells[r * self.n_cols + c]
        return t

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("can't add matrices of different shapes")
        out = Matrix(self.n_rows, self.n_cols)
        for i in range(self.n_rows * self.n_cols):
            out._cells[i] = self._cells[i] + other._cells[i]
        return out

    def __matmul__(self, other):
        if self.n_cols != other.n_rows:
            raise ValueError("shapes %s and %s don't line up"
                             % (self.shape, other.shape))
        n, m, p = self.n_rows, self.n_cols, other.n_cols
        out = Matrix(n, p)
        A, B, C = self._cells, other._cells, out._cells
        # i-k-j loop order: the inner loop runs along rows of B and C,
        # which is the friendly order for row-major storage
        for i in range(n):
            for k in range(m):
                a = A[i * m + k]
                if a == 0:
                    continue
                for j in range(p):
                    C[i * p + j] += a * B[k * p + j]
        return out


# ===========================================================================
# stack and queue on arrays
# ===========================================================================

class ArrStack:
    """Stack on a growable array; the top is the last used slot.

    push/pop/peek are O(1) (push is amortized because of the occasional
    resize).
    """

    def __init__(self, cap=4):
        self._slots = _block(max(1, cap))
        self._top = 0                  # also the number of items

    def __len__(self):
        return self._top

    def is_empty(self):
        return self._top == 0

    def push(self, val):
        if self._top == len(self._slots):
            bigger = _block(len(self._slots) * 2)
            for i in range(self._top):
                bigger[i] = self._slots[i]
            self._slots = bigger
        self._slots[self._top] = val
        self._top += 1

    def pop(self):
        if self._top == 0:
            raise IndexError("pop from an empty stack")
        self._top -= 1
        val = self._slots[self._top]
        self._slots[self._top] = None
        return val

    def peek(self):
        if self._top == 0:
            raise IndexError("peek on an empty stack")
        return self._slots[self._top - 1]

    def __repr__(self):
        return "ArrStack(%s <- top)" % [self._slots[i] for i in range(self._top)]


class RingQueue:
    """Queue on a circular buffer.

    `_front` is where the next dequeue happens and the back is
    (_front + _count) % cap. Both ends just move forward and wrap around,
    so nothing ever gets shifted. (Compare with using list.pop(0), which
    shifts everything left on every dequeue -> O(n) each.)

    enqueue O(1) amortized, dequeue/peek O(1)
    """

    def __init__(self, cap=4):
        self._slots = _block(max(1, cap))
        self._front = 0
        self._count = 0

    def __len__(self):
        return self._count

    def is_empty(self):
        return self._count == 0

    def _grow(self):
        old = self._slots
        cap = len(old)
        bigger = _block(cap * 2)
        # copy in queue order so the front ends up at index 0 again
        for i in range(self._count):
            bigger[i] = old[(self._front + i) % cap]
        self._slots = bigger
        self._front = 0

    def enqueue(self, val):
        if self._count == len(self._slots):
            self._grow()
        back = (self._front + self._count) % len(self._slots)
        self._slots[back] = val
        self._count += 1

    def dequeue(self):
        if self._count == 0:
            raise IndexError("dequeue from an empty queue")
        val = self._slots[self._front]
        self._slots[self._front] = None
        self._front = (self._front + 1) % len(self._slots)
        self._count -= 1
        return val

    def peek(self):
        if self._count == 0:
            raise IndexError("peek on an empty queue")
        return self._slots[self._front]

    def __iter__(self):
        cap = len(self._slots)
        for i in range(self._count):
            yield self._slots[(self._front + i) % cap]

    def __repr__(self):
        return "RingQueue(front %s back)" % list(self)


# ===========================================================================
# linked list
# ===========================================================================

class ListNode:
    __slots__ = ("val", "nxt")     # keeps each node small

    def __init__(self, val, nxt=None):
        self.val = val
        self.nxt = nxt

    def __repr__(self):
        return "ListNode(%r)" % (self.val,)


class LinkedList:
    """Singly linked list that keeps both a head and a tail pointer.

    The tail pointer makes add_back O(1). pop_back is still O(n) because we
    need the node before the tail and the only way to get there is to walk
    from the head.

    add_front / pop_front / add_back / insert_after(node) : O(1)
    pop_back : O(n)
    insert_at / remove_at / get : O(i)
    find / remove_value / walk / reverse : O(n)
    """

    def __init__(self, items=()):
        self.head = None
        self.tail = None
        self._count = 0
        for x in items:
            self.add_back(x)

    def __len__(self):
        return self._count

    def is_empty(self):
        return self._count == 0

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node.val
            node = node.nxt

    def walk(self, fn):
        """Traverse head -> tail calling fn(value) on each item."""
        node = self.head
        while node is not None:
            fn(node.val)
            node = node.nxt

    def to_list(self):
        return list(self)

    def __repr__(self):
        return "LinkedList(" + " -> ".join(repr(v) for v in self) + ")"

    def _node(self, i):
        if i < 0 or i >= self._count:
            raise IndexError("index out of range")
        node = self.head
        for _ in range(i):
            node = node.nxt
        return node

    def get(self, i):
        return self._node(i).val

    def __getitem__(self, i):
        if i < 0:
            i += self._count
        return self._node(i).val

    # adding
    def add_front(self, val):
        self.head = ListNode(val, self.head)
        if self.tail is None:
            self.tail = self.head
        self._count += 1

    def add_back(self, val):
        node = ListNode(val)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.nxt = node
            self.tail = node
        self._count += 1

    def insert_after(self, node, val):
        """Link a new node in right after `node` (O(1))."""
        new = ListNode(val, node.nxt)
        node.nxt = new
        if node is self.tail:
            self.tail = new
        self._count += 1
        return new

    def insert_at(self, i, val):
        """Insert so that the new value ends up at index i (0..len)."""
        if i < 0 or i > self._count:
            raise IndexError("insert position out of range")
        if i == 0:
            self.add_front(val)
        elif i == self._count:
            self.add_back(val)
        else:
            self.insert_after(self._node(i - 1), val)

    # removing
    def pop_front(self):
        if self.head is None:
            raise IndexError("pop from an empty list")
        node = self.head
        self.head = node.nxt
        if self.head is None:
            self.tail = None
        self._count -= 1
        return node.val

    def pop_back(self):
        if self.head is None:
            raise IndexError("pop from an empty list")
        if self.head is self.tail:
            return self.pop_front()
        before = self.head
        while before.nxt is not self.tail:     # this walk is the O(n) part
            before = before.nxt
        val = self.tail.val
        before.nxt = None
        self.tail = before
        self._count -= 1
        return val

    def remove_at(self, i):
        if i < 0 or i >= self._count:
            raise IndexError("index out of range")
        if i == 0:
            return self.pop_front()
        before = self._node(i - 1)
        node = before.nxt
        before.nxt = node.nxt
        if node is self.tail:
            self.tail = before
        self._count -= 1
        return node.val

    def remove_value(self, val):
        """Unlink the first node holding val, ValueError if there isn't one."""
        before, node = None, self.head
        while node is not None and node.val != val:
            before, node = node, node.nxt
        if node is None:
            raise ValueError("%r is not in the list" % (val,))
        if before is None:
            self.head = node.nxt
        else:
            before.nxt = node.nxt
        if node is self.tail:
            self.tail = before
        self._count -= 1

    # searching etc.
    def find(self, val):
        """First node with this value, or None."""
        node = self.head
        while node is not None:
            if node.val == val:
                return node
            node = node.nxt
        return None

    def index_of(self, val):
        for i, v in enumerate(self):
            if v == val:
                return i
        return -1

    def __contains__(self, val):
        return self.find(val) is not None

    def reverse(self):
        """Flip the list in place by turning every `nxt` pointer around."""
        prev, node = None, self.head
        self.tail = node
        while node is not None:
            node.nxt, prev, node = prev, node, node.nxt
        self.head = prev


class NodeStack:
    """Stack made from linked nodes, top = head. Every operation is O(1)
    with no resizing ever, but each item costs a whole node object."""

    def __init__(self):
        self._top = None
        self._count = 0

    def __len__(self):
        return self._count

    def is_empty(self):
        return self._top is None

    def push(self, val):
        self._top = ListNode(val, self._top)
        self._count += 1

    def pop(self):
        if self._top is None:
            raise IndexError("pop from an empty stack")
        node = self._top
        self._top = node.nxt
        self._count -= 1
        return node.val

    def peek(self):
        if self._top is None:
            raise IndexError("peek on an empty stack")
        return self._top.val


class NodeQueue:
    """Queue made from linked nodes: add at the tail, remove at the head.
    Both O(1). (The other way around, dequeue would be O(n), same reason
    pop_back is.)"""

    def __init__(self):
        self._head = None
        self._tail = None
        self._count = 0

    def __len__(self):
        return self._count

    def is_empty(self):
        return self._head is None

    def enqueue(self, val):
        node = ListNode(val)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.nxt = node
            self._tail = node
        self._count += 1

    def dequeue(self):
        if self._head is None:
            raise IndexError("dequeue from an empty queue")
        node = self._head
        self._head = node.nxt
        if self._head is None:
            self._tail = None
        self._count -= 1
        return node.val

    def peek(self):
        if self._head is None:
            raise IndexError("peek on an empty queue")
        return self._head.val


# ===========================================================================
# rooted tree
# ===========================================================================

class TreeNode:
    """Left-child / right-sibling node (CLRS 10.3). Three links no matter how
    many children: parent, first child, and the next sibling. A node's
    children are basically a linked list through next_sibling."""

    __slots__ = ("val", "parent", "first_child", "next_sibling")

    def __init__(self, val, parent=None):
        self.val = val
        self.parent = parent
        self.first_child = None
        self.next_sibling = None

    def __repr__(self):
        return "TreeNode(%r)" % (self.val,)


class GeneralTree:
    """Rooted tree where nodes can have any number of children.

    add_child is O(d) with d = number of children the parent already has
    (it walks to the end of the sibling chain so children stay in order).
    kids() O(d), depth O(depth), height/size/find/traversals O(n).
    """

    def __init__(self, root_val):
        self.root = TreeNode(root_val)
        self._count = 1

    def __len__(self):
        return self._count

    def add_child(self, parent, val):
        kid = TreeNode(val, parent)
        if parent.first_child is None:
            parent.first_child = kid
        else:
            last = parent.first_child
            while last.next_sibling is not None:
                last = last.next_sibling
            last.next_sibling = kid
        self._count += 1
        return kid

    def kids(self, node):
        k = node.first_child
        while k is not None:
            yield k
            k = k.next_sibling

    def is_leaf(self, node):
        return node.first_child is None

    def depth(self, node):
        d = 0
        while node.parent is not None:
            node = node.parent
            d += 1
        return d

    def height(self, node=None):
        """Edges on the longest downward path from node (root by default)."""
        start = self.root if node is None else node
        best = 0
        todo = ArrStack()
        todo.push((start, 0))
        while not todo.is_empty():
            cur, d = todo.pop()
            best = max(best, d)
            for k in self.kids(cur):
                todo.push((k, d + 1))
        return best

    def preorder(self, node=None):
        """Node first, then each child's subtree left to right.

        Uses an explicit stack, not recursion, so a really deep tree doesn't
        run into Python's recursion limit.
        """
        start = self.root if node is None else node
        order = []
        todo = ArrStack()
        todo.push(start)
        while not todo.is_empty():
            cur = todo.pop()
            order.append(cur.val)
            # push right to left so the leftmost child gets popped first
            for k in reversed(list(self.kids(cur))):
                todo.push(k)
        return order

    def bfs(self):
        """Level by level, using NodeQueue."""
        order = []
        q = NodeQueue()
        q.enqueue(self.root)
        while not q.is_empty():
            cur = q.dequeue()
            order.append(cur.val)
            for k in self.kids(cur):
                q.enqueue(k)
        return order

    def find(self, val):
        q = NodeQueue()
        q.enqueue(self.root)
        while not q.is_empty():
            cur = q.dequeue()
            if cur.val == val:
                return cur
            for k in self.kids(cur):
                q.enqueue(k)
        return None

    def cut(self, node):
        """Remove node and its whole subtree. Returns how many nodes went."""
        if node is self.root:
            raise ValueError("can't cut off the root")
        p = node.parent
        if p.first_child is node:
            p.first_child = node.next_sibling
        else:
            prev = p.first_child
            while prev.next_sibling is not node:
                prev = prev.next_sibling
            prev.next_sibling = node.next_sibling
        n_gone = len(self.preorder(node))
        node.parent = None
        node.next_sibling = None
        self._count -= n_gone
        return n_gone


if __name__ == "__main__":
    arr = ResizableArray([10, 20, 30])
    arr.insert(1, 15)
    arr.remove_at(0)
    print(arr, "capacity:", arr.capacity)

    m = Matrix.from_rows([[1, 2], [3, 4]])
    m.add_row(2, [5, 6])
    m.add_col(0, [0, 0, 0])
    print(m, "shape:", m.shape)
    print("transposed:", m.transposed())

    st = ArrStack()
    for ch in "abc":
        st.push(ch)
    print(st)
    print("pop:", st.pop(), " peek:", st.peek())

    q = RingQueue(cap=2)
    for i in range(5):
        q.enqueue(i)
    q.dequeue()
    print(q)

    ll = LinkedList([1, 2, 3])
    ll.add_front(0)
    ll.insert_at(2, 1.5)
    ll.remove_value(3)
    ll.reverse()
    print(ll)

    t = GeneralTree("CEO")
    cto = t.add_child(t.root, "CTO")
    cfo = t.add_child(t.root, "CFO")
    t.add_child(cto, "Dev lead")
    t.add_child(cto, "QA lead")
    t.add_child(cfo, "Accountant")
    print("preorder:", t.preorder())
    print("bfs:     ", t.bfs())
    print("height:", t.height(), " nodes:", len(t))
