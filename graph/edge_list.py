"""Edge-list utilities for simple undirected graphs."""

import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import njit
from scipy import sparse

from graph.type_aliases import EDGE, NODE


@njit(fastmath=True, inline="always")
def canonical_edge(node1: NODE, node2: NODE) -> EDGE:
    """Return an ordered edge tuple `(u, v)` where `u < v`."""
    if node1 < node2:
        return node1, node2
    return node2, node1


@njit(fastmath=True)
def edge_list_to_adjacency_matrix(
    edge_list: npt.NDArray[np.uint32],
) -> npt.NDArray[np.uint32]:
    """
    Build a dense adjacency matrix from an undirected edge list.

    Args:
        edge_list: Edge list with shape `[E, 2]`.

    Returns:
        Dense adjacency matrix with shape `[N, N]`, where `N = max(edge_list) + 1`.
        Returns shape `[0, 0]` when `E == 0`.
    """
    if len(edge_list) == 0:
        return np.zeros((0, 0), dtype=np.uint32)

    num_nodes = int(edge_list.max()) + 1
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.uint32)
    for node1, node2 in edge_list:
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1
    return adjacency_matrix


@njit(fastmath=True)
def edge_list_to_csr(
    num_nodes: int,
    edge_list: npt.NDArray[np.uint32],
    degrees: npt.NDArray[np.uint32],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """
    Build a CSR adjacency representation from an undirected edge list.

    Args:
        num_nodes: Number of nodes
        edge_list: [E, 2]
        degrees: [N, ]

    Returns:
        offsets: [N + 1, ]
        neighbors: [2E, ]
    """
    offsets = np.zeros(num_nodes + 1, dtype=np.uint32)
    for node_idx in range(num_nodes):
        offsets[node_idx + 1] = offsets[node_idx] + degrees[node_idx]

    neighbors = np.empty(offsets[num_nodes], dtype=np.uint32)
    cursor = offsets[:-1].copy()
    for node1, node2 in edge_list:
        pos1 = cursor[node1]
        neighbors[pos1] = node2
        cursor[node1] += 1

        pos2 = cursor[node2]
        neighbors[pos2] = node1
        cursor[node2] += 1

    return offsets, neighbors


def edge_list_to_sparse_adjacency(
    num_nodes: int, edge_list: npt.NDArray[np.uint32]
) -> sparse.csr_array:
    """Build symmetric sparse adjacency from an undirected edge list."""
    rows = edge_list[:, 0].astype(np.int64)
    cols = edge_list[:, 1].astype(np.int64)
    data = np.ones(len(edge_list), dtype=np.float64)

    return sparse.coo_array(  # coo_matrix → coo_array
        (
            np.concatenate([data, data]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    ).tocsr()


@njit(fastmath=True)
def edge_list_to_degrees(
    num_nodes: int, edge_list: npt.NDArray[np.uint32]
) -> npt.NDArray[np.uint32]:
    """
    Compute node degrees from an undirected edge list.

    Args:
        num_nodes: Number of nodes `N`.
        edge_list: Edge list with shape `[E, 2]`.

    Returns:
        Degree array with shape `[N]`.
    """
    degrees = np.zeros(num_nodes, dtype=np.uint32)
    for node1, node2 in edge_list:
        degrees[node1] += 1
        degrees[node2] += 1
    return degrees


@njit(fastmath=True)
def _find_root(parent: npt.NDArray[np.uint32], node: NODE) -> NODE:
    """Find a DSU root with path compression (path halving)."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(fastmath=True)
def _union(
    parent: npt.NDArray[np.uint32],
    size: npt.NDArray[np.uint32],
    node1: NODE,
    node2: NODE,
) -> None:
    """Union two DSU components using union-by-size."""
    root1 = _find_root(parent, node1)
    root2 = _find_root(parent, node2)
    if root1 == root2:
        return

    if size[root1] < size[root2]:
        parent[root1] = root2
        size[root2] += size[root1]
    else:
        parent[root2] = root1
        size[root1] += size[root2]


@njit(fastmath=True)
def is_connected(num_nodes: int, edge_list: npt.NDArray[np.uint32]) -> bool:
    """
    Check whether all nodes are in one connected component.

    Args:
        num_nodes: Number of nodes `N`.
        edge_list: Edge list with shape `[E, 2]`.

    Returns:
        True if connected, otherwise False.
    """
    if num_nodes <= 1:
        return True

    parent = np.arange(num_nodes, dtype=np.uint32)
    size = np.ones(num_nodes, dtype=np.uint32)

    for node1, node2 in edge_list:
        _union(parent, size, node1, node2)

    root0 = _find_root(parent, 0)
    for node_idx in range(1, num_nodes):
        if _find_root(parent, node_idx) != root0:
            return False
    return True


@njit(fastmath=True)
def has_selfloop(edge_list: npt.NDArray[np.uint32]) -> bool:
    """
    Check whether any edge is a self-loop.

    Args:
        edge_list: Edge list with shape `[E, 2]`.

    Returns:
        True if any `(u, u)` exists.
    """
    for node1, node2 in edge_list:
        if node1 == node2:
            return True
    return False


@njit(fastmath=True)
def has_multiedge(edge_list: npt.NDArray[np.uint32]) -> bool:
    """
    Check whether duplicate undirected edges exist.

    Args:
        edge_list: Canonical edge list with shape `[E, 2]`.

    Returns:
        True if duplicate edges exist.
    """
    seen = set()
    for node1, node2 in edge_list:
        if (node1, node2) in seen:
            return True
        seen.add((node1, node2))
    return False


def graph_to_edge_list(graph: nx.Graph) -> npt.NDArray[np.uint32]:
    """
    Convert a NetworkX graph into a canonical edge list.

    Args:
        graph: NetworkX undirected graph.

    Returns:
        Canonical edge list with shape `[E, 2]` and `u < v` per edge.
    """
    raw_edges = np.array(list(graph.edges), dtype=np.uint32)
    if raw_edges.size == 0:
        return np.empty((0, 2), dtype=np.uint32)

    edge_list = np.empty_like(raw_edges)
    edge_list[:, 0] = np.minimum(raw_edges[:, 0], raw_edges[:, 1])
    edge_list[:, 1] = np.maximum(raw_edges[:, 0], raw_edges[:, 1])
    return edge_list


def edge_list_to_graph(num_nodes: int, edge_list: npt.NDArray[np.uint32]) -> nx.Graph:
    """
    Convert an edge list into a NetworkX graph.

    Args:
        num_nodes: Number of nodes `N`.
        edge_list: Edge list with shape `[E, 2]`.

    Returns:
        NetworkX undirected graph with `N` nodes.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_list)
    return graph
