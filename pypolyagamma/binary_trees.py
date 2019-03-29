"""
Represent a binary tree as a tuple of tuples.
Each tuple is length two, and the elements are either tuples or integers.
Integers denote leaf nodes; tuples denote internal choice nodes.

((1, 2), (3, 4)) denotes the tree
      root
     /    \
    -     -
   / \   / \
  1  2  3  4

Note that if the tree has N leaves then it must have N-1 internal
nodes including the root.
"""

import numpy as np
import numpy.random as npr
npr.seed(0)

def check_tree(tree):
    def _check_node(node):
        if np.isscalar(node):
            return True
        elif isinstance(node, tuple) and len(node) == 2:
            return _check_node(node[0]) & _check_node(node[1])
        else:
            raise Exception("Not a tree!")
    return _check_node(tree)


def balanced_binary_tree(n_leaves):
    """
    Create a balanced binary tree
    """
    def _balanced_subtree(leaves):
        if len(leaves) == 1:
            return leaves[0]
        elif len(leaves) == 2:
            return (leaves[0], leaves[1])
        else:
            split = len(leaves) // 2
            return (_balanced_subtree(leaves[:split]),
                    _balanced_subtree(leaves[split:]))

    return _balanced_subtree(np.arange(n_leaves))

def decision_list(n_leaves):
    """
    Create a decision list
    """
    def _list(leaves):
        if len(leaves) == 2:
            return (leaves[0], leaves[1])
        else:
            return (leaves[0], _list(leaves[1:]))
    return _list(np.arange(n_leaves))


def random_tree(n_leaves):
    """
    Randomly partition the nodes
    """
    def _random_subtree(leaves):
        if len(leaves) == 1:
            return leaves[0]
        elif len(leaves) == 2:
            return (leaves[0], leaves[1])
        else:
            split = npr.randint(1, len(leaves)-1)
            return (_random_subtree(leaves[:split]),
                    _random_subtree(leaves[split:]))

    return _random_subtree(np.arange(n_leaves))


def leaves(tree):
    """
    Return the leaves in this subtree.
    """
    lvs = []
    def _leaves(node):
        if np.isscalar(node):
            lvs.append(node)
        elif isinstance(node, tuple) and len(node) == 2:
            _leaves(node[0])
            _leaves(node[1])
        else:
            raise Exception("Not a tree!")

    _leaves(tree)
    return lvs


def choices(tree):
    """
    Get the 'address' of each leaf node in terms of internal
    node choices
    """
    n = len(leaves(tree))
    addr = np.nan * np.ones((n, n-1))
    def _addresses(node, index, choices):
        # index is the index of the current internal node
        # choices is a list of (indice, 0/1) choices made
        if np.isscalar(node):
            for i, choice in choices:
                addr[node, i] = choice
            return index

        elif isinstance(node, tuple) and len(node) == 2:
            newindex = _addresses(node[0], index+1, choices + [(index, 0)])
            newindex = _addresses(node[1], newindex, choices + [(index, 1)])
            return newindex

        else:
            raise Exception("Not a tree!")

    _addresses(tree, 0, [])
    return addr


def ids(tree):
    # keep track of node ids
    from itertools import count
    from collections import defaultdict
    dd = defaultdict(lambda c=count(): c.__next__())

    def _ids(node):
        dd[node]
        if isinstance(node, tuple) and len(node) == 2:
            _ids(node[0])
            _ids(node[1])
    _ids(tree)
    return dd


def adjacency(tree):
    """
    Construct the adjacency matrix of the tree
    :param tree:
    :return:
    """
    dd = ids(tree)
    N = len(dd)
    A = np.zeros((N, N))

    def _adj(node):
        if np.isscalar(node):
            return
        elif isinstance(node, tuple) and len(node) == 2:
            A[dd[node], dd[node[0]]] = 1
            A[dd[node[0]], dd[node]] = 1
            _adj(node[0])

            A[dd[node], dd[node[1]]] = 1
            A[dd[node[1]], dd[node]] = 1
            _adj(node[1])

    _adj(tree)
    return A


def depths(tree):
    _ids = ids(tree)
    out = {}
    def _depths(node, d):
        out[_ids[node]] = d
        if isinstance(node, tuple) and len(node) == 2:
            _depths(node[0], d+1)
            _depths(node[1], d+1)
    _depths(tree, 0)
    return out


def print_tree(tree):
    lines = {}
    def _collect_lines(node, depth, offset):
        if np.isscalar(node):
            if depth in lines:
                lines[depth] = lines[depth] + [(offset, node)]
            else:
                lines[depth] = [(offset, node)]

        elif isinstance(node, tuple) and len(node) == 2:
            lsize = len(leaves(node[0]))
            _collect_lines(node[0], depth+1, offset)
            _collect_lines(node[1], depth+1, offset + lsize)
        else:
            raise Exception("Not a tree!")

    _collect_lines(tree, 0, 0)

    # Print the tree
    height = np.max(depths(tree))
    n = len(leaves(tree))
    for depth in range(height + 1):
        line = [" "] * n
        if depth in lines:
            for (o, i) in lines[depth]:
                line[i] = str(i)
        print(' '.join(line))

if __name__ == "__main__":
    tree = balanced_binary_tree(8)
    check_tree(tree)
    print(tree)
    print(choices(tree))
    print(adjacency(tree))
    print("")

    tree = random_tree(10)
    check_tree(tree)
    print(tree)
    print(choices(tree))
    print(adjacency(tree))
    print("")


