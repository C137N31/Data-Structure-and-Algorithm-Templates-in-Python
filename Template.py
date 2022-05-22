from bisect import bisect_left, bisect_right
from heapq import heappop, heappush
from operator import itemgetter
from sortedcontainers import SortedList, SortedDict
from collections import defaultdict, Counter, OrderedDict, deque
from functools import lru_cache
import random
import typing

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def preorder(self, result):
        stack = [self]
        while stack:
            curr = stack.pop()
            result.append(curr.val)
            if curr.right: stack.append(curr.right)
            if curr.left: stack.append(curr.left)
        
    def inorder(self, result):
        stack = []
        curr = self
        while stack or curr:
            if curr:
                stack.append(curr)
                curr = curr.left
            else:
                curr = stack.pop()
                result.append(curr.val)
                curr = curr.right
        
    def postorder(self, result):
        stack = []
        curr = self
        prev = None
        while stack or curr:
            if curr:
                stack.append(curr)
                curr = curr.left
            else:
                curr = stack[-1]
                if curr.right and curr.right != prev:
                    curr = curr.right
                else:
                    curr = stack.pop()
                    result.append(curr.val)
                    prev, curr = curr, None

    def buildBST(self, node):   # build binary search tree by adding node
        if node.val < self.val:
            if not self.left:
                self.left = node
                return True
            return self.left.buildBST(node)
        elif node.val > self.val:
            if not self.right:
                self.right = node
                return True
            return self.right.buildBST(node)
        else:
            return False
    
    def solve1(self, root):
        if not root: return -1 # or something
        # if func0(root): return something
        left_return  = self.solve1(root.left)
        right_return = self.solve1(root.right)
        # func1 is a function on root, left_return, right_return
        func1 = max(root.val, left_return, right_return)
        return func1

    def solve2(self, root1, root2):
        if not root1 and not root2: return -1 # or something
        # if func0(root1, root2): return something
        left_return  = self.solve2(root1.left, root2.left)
        right_return = self.solve2(root1.right, root2.right)
        # func1 is a function on root, left_return, right_return
        func1 = max(root1.val, root2.val, left_return, right_return)
        return func1

class FenwickTree:  # binary indexed tree for range value: sum/count/max/min
    def __init__(self, nums):
        self.vals = [0] * (len(nums)+1) # vals[i] = sum(nums[:i]) or count/max/min
        for i in range(len(nums)):
            self.add(i+1, nums[i])
    
    def _lowbit(self, x):
        return x & (-x)
    
    def add(self, i, delta):    # add nums[i-1] with delta 
        while i < len(self.vals):
            self.vals[i] += delta
            i += self._lowbit(i)
            
    def query(self, i):         # query the sum of nums[:i]
        result = 0
        while i > 0:
            result += self.vals[i]
            i -= self._lowbit(i)
        return result

class SegmentTreeNode:
    def __init__(self, start, end, val, left=None, right=None):
        self.start = start
        self.end = end      
        self.val = val      # nums[start:end+1] range value: sum/count/max/min
        self.left = left    # left segment tree node
        self.right = right  # right segment tree node

class SegmentTree:
    def __init__(self, nums):
        self.root = self.buildSegmentTree(nums, 0, len(nums)-1)

    def buildSegmentTree(self, nums, start, end):
        if start == end:
            return SegmentTreeNode(start, end, nums[end])
        
        mid = (start + end) // 2
        left  = self.buildSegmentTree(nums, start, mid)
        right = self.buildSegmentTree(nums, mid+1, end)
        return SegmentTreeNode(start, end, left.val+right.val, left, right)

    def query(self, node, start, end):  # query range value of nums[start:end+1]
        if node.start == start and node.end == end: 
            return node.val

        mid = (node.start + node.end) // 2
        if mid < start:
            return self.query(node.right, start, end)
        elif mid >= end:
            return self.query(node.left, start, end)
        else:
            return self.query(node.left, start, mid) + self.query(node.right, mid+1, end)

    def update(self, node, i, val):     # update nums[i] with val
        if node.start == i == node.end:
            node.val = val
            return 

        mid = (node.start + node.end) // 2
        if mid < i: self.update(node.right, i, val)
        else:       self.update(node.left, i, val)
        node.val = node.left.val + node.right.val

class Trie:
    def __init__(self):
         self.children = {}
         self.isEnd = False

    def add(self, word):
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = Trie()
            node = node.children[char]
        node.isEnd = True

    def search(self, word):
        node = self
        for char in word:
            if char not in node.children: return False
            node = node.children[char]
        return node.isEnd

    def remove(self, word):
        self._delete(word, 0, self)

    def _delete(self, word, i, node):
        if i == len(word):
            if not node.isEnd: return False
            node.isEnd = False
            return len(node.children) == 0

        if word[i] not in node.children: return False

        if self._delete(word, i+1, node.children[word[i]]):
            node.children.pop(word[i])
            return len(node.children) == 0
        
        return False

class ListNode:     # singly linked list
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __len__(self):
        result = 0
        node = self
        while node:
            result += 1
            node = node.next
        return result
    
    def getMid(self):
        slow, fast = self, self
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def reverse(self):
        prev, curr = None, self
        while curr:
            temp = curr.next
            curr.next = prev
            prev, curr = curr, temp
        return prev

class LinkedList:
    def __init__(self, head):
        self.head = head

    def merge(self, head1, head2):  
        tail = dummy = ListNode()
        while head1 and head2:
            if head1.val < head2.val:
                tail.next = head1
                head1 = head1.next
            else:
                tail.next = head2
                head2 = head2.next
            tail = tail.next
        
        tail.next = head1 if head1 else head2
        while tail.next: 
            tail = tail.next

        return (dummy.next, tail)   # return (head, tail) of merged list

    def split(self, head, length):
        node = head
        while node and length > 1:
            node = node.next
            length -= 1
        
        nextHead = node.next if node else None  # head of next list
        if node: node.next = None   # cut current list
        return nextHead

class DLinkedListNode:      # doubly linked list
    def __init__(self, val=0):
        self.val = val
        self.prev = self.next = None

class DLinkedList:
    def __init__(self):
        self.head = DLinkedListNode()
        self.tail = DLinkedListNode()
        self.size = 0

        self.head.next = self.tail
        self.tail.prev = self.head

    def __len__(self):
        return self.size

    def add(self, node):
        node.prev, node.next = self.head, self.head.next
        node.next.prev = self.head.next = node
        self.size += 1

    def pop(self, node=None):
        if self.size == 0: return None
        self.size -= 1
        if not node: node = self.tail.prev
        prev, next = node.prev, node.next
        prev.next, next.prev = next, prev
        return node

class DirectedGraphNode:        # for topological sorting
    def __init__(self):
        self.inEdges = 0
        self.outNodes = set()   # list or set or dictionary

class DirectedGraph:
    def __init__(self, nodes, edges):  # edges [(weight, u, v)]
        self.graph = {}     # need to set up for all nodes. DON'T use defaultdict(list) based on edges

        for u in nodes:
            self.graph[u] = self.graph.get(u, DirectedGraphNode())

        for weight, u, v in edges:
            if v not in self.graph[u].outNodes:
                self.graph[u].outNodes.add(v)
                self.graph[v].inEdges += 1

    def topologicalSort(self):
        self.queue = deque()
        for u, dgNode in self.graph.items():
            if dgNode.inEdges == 0 :
                self.queue.append(u)

        result = []
        while self.queue:
            u = self.queue.popleft()
            result.append(u)
            for _ in range(len(self.graph[u].outNodes)):
                v = self.graph[u].outNodes.pop()
                self.graph[v].inEdges -= 1
                if self.graph[v].inEdges == 0:
                    self.queue.append(v)

        return result

class UndirectedGraph:
    def __init__(self, edges):  # edges [(weight, u, v)]
        self.graph = defaultdict(list)

        for weight, u, v in edges:
            self.graph[u].append((weight,v))
            self.graph[v].append((weight,u))

    # Prim algorithm is Dijkstra algorithm
    def prim(self, nodeCount, start):      # for MST: minimum spanning tree
        result = 0
        visit = set()
        minHeap = [(0, start)]
        while len(visit) < nodeCount:   # while minHeap:
            weight, u = heappop(minHeap)
            if u in visit: continue
            result += weight
            visit.add(u)
            for weight, v in self.graph[u]:
                if v not in visit:
                    heappush(minHeap, (weight, v))

    def kruskal(self, nodeCount, edges):    # edges [(weight, u, v)]
        result = 0
        djs = DisjointSet(nodeCount)
        edges.sort()    # by weight
        for weight, u, v in edges:
            if djs.joint(u,v): continue
            djs.union(u,v)
            result += weight
        return result

class DisjointSet:
    def __init__(self, n):
        self.count = 0      # disjoint sets count
        self.root = [-1] * n
        self.size = [0] * n

        for i in range(n):
            self.add(i)
        
    def __len__(self):
        return self.count

    def add(self, a):
        self.root[a] = a
        self.size[a] = 1
        self.count += 1
    
    def union(self, a, b):
        rootA, rootB = self.find(a), self.find(b)
        if rootA == rootB: return False
        if self.size[rootA] >= self.size[rootB]:
            self.root[rootB] = rootA
            self.size[rootA] += self.size[rootB]
        else:
            self.root[rootA] = rootB
            self.size[rootB] += self.size[rootA]
        self.count -= 1
        return True

    def find(self, a):
        if self.root[a] != a:
            self.root[a] = self.find(self.root[a])
        return self.root[a]

    def joint(self, a, b):
        return self.find(a) == self.find(b)

class Sort:
    def __init__(self, nums):
        self.nums = nums

    def quickSort(self, nums, start, end):  # nums[start:end+1]
        if start >= end: return
        pivotIndex = random.randint(start,end)
        pivot = nums[pivotIndex]
        l, r = start, end
        while l <= r:
            while l <= r and nums[l] < pivot: l += 1
            while l <= r and nums[r] > pivot: r -= 1
            if l <= r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1
        self.quickSort(nums, start, r)
        self.quickSort(nums, l, end)

    def quickSelect(self, nums, start, end, k):     # select first k smallest nums
        if start >= end: return nums[:k]
        pivotIndex = random.randint(start,end)
        pivot = nums[pivotIndex]
        l, r = start, end
        while l <= r:
            while l <= r and nums[l] < pivot: l += 1
            while l <= r and nums[r] > pivot: r -= 1
            if l <= r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1
        if   l > k: return self.quickSelect(nums, start, r, k)
        elif l < k: return self.quickSelect(nums, l, end, k)
        else:       return nums[:k]

    def mergeSortTopDown(self, nums):
        if len(nums) < 2: return
        mid = len(nums)//2
        nums1 = nums[:mid]
        nums2 = nums[mid:]
        self.mergeSortTopDown(nums1)
        self.mergeSortTopDown(nums2)
        self.mergeTopDown(nums1, nums2, nums)

    def mergeTopDown(self, nums1, nums2, nums):
        i = j = 0
        while i+j < len(nums):
            if j == len(nums2) or (i < len(nums1) and nums1[i] < nums2[j]):
                nums[i+j] = nums1[i]
                i += 1
            else:
                nums[i+j] = nums2[j]
                j += 1

    def mergeSortBottomUP(self, nums):
        interval = 1
        while interval < len(nums):
            for start in range(0, len(nums)-interval, 2*interval):
                self.mergeBottomUP(nums, start, interval)
            interval *= 2

    def mergeBottomUP(self, nums, start, interval):
        temp = []
        mid = min(len(nums), start + interval)      # nums1 = [start:mid]
        end = min(len(nums), start + 2*interval)    # nums2 = [mid:end]
        i, j = start, mid
        while i+j < mid+end:
            if j == end or (i < mid and nums[i] < nums[j]):
                temp.append(nums[i])
                i += 1
            else: 
                temp.append(nums[j])
                j += 1
        nums[start:end] = temp

class Search:   # array search target
    def __init__(self, arr=None):
        self.arr = arr
    
    def binarySearch1(self, arr, target, left, right):   # arr[left:right]
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] <= target: 
                result = arr[mid]
                left = mid + 1
            else:
                right = mid - 1
        return result

    def binarySearch2(self, arr, target, left, right):   # arr[left:right+1]
        while left < right:
            mid = (left + right) // 2
            # >= is lower bound (minimal index whose value >= target) = bisect_left
            #       leftmost index to insert target
            # >  is upper bound (minimal index whose value > target) = bisect_right
            #       rightmost index to insert target
            if arr[mid] >= target: right = mid
            else: left = mid + 1
        return left

class Match:    # string match pattern
    def __init__(self, string=None, pattern = None):
        self.s = string
        self.p = pattern

    def buildPattern(self): 
        self.next = [0] * (len(self.p) + 1)    # longest (prefix == suffix) of p[:i] 
        j = 0
        for i in range(1, len(self.p)):
            while j > 0 and self.p[i] != self.p[j]: j = self.next[j]
            if self.p[i] == self.p[j]: j += 1
            self.next[i+1] = j
        return self.next

    def matchKMP(self, string=None, pattern=None):    # all starting index of substring matching pattern
        self.buildPattern()
        result = []
        j = 0
        for i in range(len(self.s)):
            while j > 0 and self.s[i] != self.p[j]: j = self.next[j]
            if self.s[i] == self.p[j]: j += 1
            if j == len(self.p): 
                result.append(i-j+1)
                j = self.next[j]
        return result

    def matchRabinKarp(self):
        # change string to numbers for computing
        s = [ord(c)-ord('a') for c in self.s]
        p = [ord(c)-ord('a') for c in self.p]
        n, m = len(s), len(p)
        # choose one set of base and modular for hash function
        # to avoid hash collision, we may choose two or more sets of bases and modulars 
        a1, a2 = random.randint(26,100), random.randint(26,100)
        mod1, mod2 = random.randint(10**9+7,2**31-1), random.randint(10**9+7,2**31-1)
        aL1, aL2 = pow(a1, m, mod1), pow(a2, m, mod2)
        # calculate pattern's hash value
        ph1, ph2 = 0, 0
        for i in range(m):
            ph1 = (ph1*a1 + p[i]) % mod1
            ph2 = (ph2*a2 + p[i]) % mod2
        seen = {(ph1, ph2)}
        # calculate string's sliding window  hash value (window length = pattern length)
        result = []
        h1, h2 = 0, 0
        for i in range(m):
            h1 = (h1*a1 + s[i]) % mod1
            h2 = (h2*a2 + s[i]) % mod2
        if (h1, h2) in seen: result.append(0)
        else: seen.add((h1, h2))
        # slide window
        for i in range(1, n-m+1):
            h1 = (h1*a1 - s[i-1]*aL1 + s[i+m-1]) % mod1
            h2 = (h2*a2 - s[i-1]*aL2 + s[i+m-1]) % mod2
            if (h1, h2) in seen: result.append(i)
            else: seen.add((h1, h2))

        return result

print("Fan's Data Structures and Algorithms")
