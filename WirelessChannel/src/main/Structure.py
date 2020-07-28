
import math
import mpmath
import cmath
import queue
import graphviz

import numpy as np

root_path = '' # change it for yourself
adr = root_path + '/out/bin/'


class Array:

    def __init__(self, dn=0.5):
        """initializing the structure generator"""
        self.__unit = dn
        return

    def linear(self, size: int) -> np.ndarray:
        """creating a linear structure of antennas"""
        distance = np.zeros(shape=(size, size))
        for row in range(0, size):
            for col in range(0, size):
                distance[row][col] = np.abs(row - col) * self.__unit
        return distance

    def rectangular(self, row_count: int, column_count: int) -> np.ndarray:
        """creating a rectangular structure of antennas"""
        size = row_count * column_count
        distance = np.zeros(shape=(size, size))
        row = 0
        for row_0 in range(0, row_count):
            for col_0 in range(0, column_count):
                col = 0
                for row_1 in range(0, row_count):
                    for col_1 in range(0, column_count):
                        distance[row][col] = np.sqrt((row_0 - row_1) ** 2 + (col_0 - col_1) ** 2) * self.__unit
        return distance


class Correlation:

    def __init__(self, alpha=0, eta=0, mu=0):
        """initializing the correlation generator"""
        self.__alpha = alpha
        self.__eta = eta
        self.__mu = mu
        return

    def exponential(self, distance: np.ndarray) -> np.ndarray:
        """exponential correlation pattern"""
        matrix = np.power(self.__alpha, distance)
        return matrix

    def bessel(self, distance: np.ndarray) -> np.ndarray:
        """Bessel function correlation pattern"""
        x = -4 * (math.pi ** 2)
        z = self.__eta ** 2
        n = len(distance)
        corr = []
        y = 4j * math.pi * self.__eta * math.sin(self.__mu)
        for i in range(0, n):
            temp = []
            for j in range(0, n):
                arg = cmath.sqrt(z + x * (distance[i][j] ** 2) + y * distance[i][j])
                arg = mpmath.besseli(0, arg) / mpmath.besseli(0, self.__eta)
                arg = complex(arg.real, arg.imag)
                temp.append(arg)
            corr.append(temp)
        matrix = np.array(corr)
        return matrix

    def bessel_angel(self, mu_list: list, distance: np.ndarray) -> np.ndarray:
        """Bessel function correlation pattern for multiple angel of arrival"""
        x = -4 * (math.pi ** 2)
        z = self.__eta ** 2
        n = len(distance)
        corr = []
        for angel in mu_list:
            local = []
            y = 4j + np.pi * self.__eta * np.sin(angel)
            for i in range(0, n):
                temp = []
                for j in range(0, n):
                    arg = cmath.sqrt(z + x * (distance[i][j] ** 2) + y * distance[i][j])
                    arg = mpmath.besseli(0, arg) / mpmath.besseli(0, self.__eta)
                    arg = complex(arg.real, arg.imag)
                    temp.append(arg)
                local.append(temp)
            corr.append(local)
        matrix = np.asarray(corr)
        return matrix

    def trace_exponential(self, distance: np.ndarray) -> float:
        """trace of exponential correlation pattern"""
        corr = self.exponential(distance=distance)
        corr = corr @ corr.transpose().conj()
        factor = float(np.trace(corr))
        return factor

    def trace_bessel(self, distance: np.ndarray) -> float:
        """trace of bessel correlation pattern"""
        corr = self.bessel(distance=distance)
        corr = corr @ corr.transpose().conj()
        factor = float(np.trace(corr))
        return factor


class ListNode:

    def __init__(self, data):
        """initializing ListNode"""
        self.val = data
        self.next = None
        self.__count = 0
        return

    def add(self, data):
        """adding value to tail"""
        curr = self
        while curr.next is not None:
            curr = curr.next
        curr.next = ListNode(data)
        self.__count += 1
        return

    def size(self) -> int:
        """returning the size of the structure"""
        return self.__count

    def to_string(self) -> str:
        """printing the structure"""
        curr = self
        string = ''
        while curr is not None:
            string += str(curr.val) + '->'
            curr = curr.next
        string += 'None'
        return string

    def to_graph(self, name='ListNode') -> graphviz:
        """visualizing the structure"""
        curr = self
        index = 0
        graph = graphviz.Graph()
        prev = str(index)
        graph.node(prev, str(curr.val), shape='circle')
        curr = curr.next
        while curr is not None:
            index += 1
            graph.node(str(index), str(curr.val), shape='circle')
            graph.edge(prev, str(index), constraint='false')
            prev = str(index)
            curr = curr.next
        graph.render(adr + name)
        return graph


class TreeNode:

    def __init__(self, data):
        """initializing TreeNode"""
        self.left = None
        self.right = None
        self.val = data
        return

    def add_to_left(self, data):
        """adding value to left"""
        self.left = TreeNode(data)
        return

    def add_to_right(self, data):
        """adding value to right"""
        self.right = TreeNode(data)
        return

    def to_graph(self, name='TreeNode') -> graphviz:
        """visualizing structure"""
        graph = graphviz.Graph()
        q1 = queue.Queue()
        q2 = queue.Queue()
        q1.put(self)
        index = 0
        graph.node(str(index), str(self.val), shape='circle')
        q2.put(str(index))
        while not q1.empty():
            head = q1.get()
            node = q2.get()
            if head.left is not None:
                left = index + 1
                curr = str(head.left.val)
                graph.node(str(left), curr, shape='circle')
                graph.edge(node, str(left))
                q1.put(head.left)
                q2.put(curr)
            if head.right is not None:
                right = index + 2
                curr = str(head.right.val)
                graph.node(str(right), curr, shape='circle')
                graph.edge(node, str(right))
                q1.put(head.right)
                q2.put(curr)
            index += 2
        graph.render(adr + name)
        return graph


class TrieNode:

    def __init__(self):
        """initializing TrieNode"""
        self.nodes = [None for _ in range(0, 256)]
        self.end = False
        return

    def add(self, word: str):
        """adding a word to the structure"""
        root = self
        for i in range(0, len(word)):
            c = ord(word[i])
            if root.nodes[c] is None:
                root.node[c] = TrieNode()
            root = root.nodes[c]
        root.end = True
        return

    def suffix(self, word: str) -> bool:
        """checking structure for a word as a suffix"""
        root = self
        for i in range(0, len(word)):
            c = ord(word[i])
            if root.nodes[c] is None:
                return False
            root = root.nodes[c]
        return True

    def contains(self, word: str) -> bool:
        """checking structure for a word"""
        root = self
        for i in range(0, len(word)):
            c = ord(word[i])
            if root.nodes[c] is None:
                return False
            root = root.nodes[c]
        return root.end
