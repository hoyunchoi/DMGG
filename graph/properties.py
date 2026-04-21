"""Graph property metrics computed from edge lists."""

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy import sparse

from graph.edge_list import _find_root, _union, edge_list_to_sparse_adjacency


# ================= Giant cluster based properties =================
@njit(fastmath=True)
def get_giant_cluster_mask(
    num_nodes: int, edge_list: npt.NDArray[np.uint32]
) -> npt.NDArray[np.bool_]:
    """
    Compute giant-component membership of nodes.

    Args:
        num_nodes: Number of nodes
        edge_list: [E, 2]

    Returns:
        mask: [N, ]
    """
    mask = np.zeros(num_nodes, dtype=np.bool_)

    parent = np.arange(num_nodes, dtype=np.uint32)
    size = np.ones(num_nodes, dtype=np.uint32)

    for edge_idx in range(len(edge_list)):
        node1 = int(edge_list[edge_idx, 0])
        node2 = int(edge_list[edge_idx, 1])
        _union(parent, size, node1, node2)

    largest_root = 0
    largest_size = 1
    for node_idx in range(num_nodes):
        root = _find_root(parent, node_idx)
        if size[root] > largest_size:
            largest_size = size[root]
            largest_root = root

    for node_idx in range(num_nodes):
        if _find_root(parent, node_idx) == largest_root:
            mask[node_idx] = True

    return mask


@njit(fastmath=True)
def get_giant_cluster(
    num_nodes: int, edge_list: npt.NDArray[np.uint32], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.uint32]:
    """
    Extract the giant-component edge list and remap node indices to [0, gc_size]

    Args:
        num_nodes: Number of nodes
        edge_list: [E, 2]
        mask: [N, ]

    Return:
        gc_edge_list: [E_gc, 2], edge list of the giant cluster
    """
    remap = np.full(num_nodes, -1, dtype=np.uint32)
    next_idx = 0
    for node_idx in range(num_nodes):
        if mask[node_idx] == 1:
            remap[node_idx] = next_idx
            next_idx += 1

    gc_edge_list: list[list[np.uint32]] = []
    for node1, node2 in edge_list:
        if mask[node1] and mask[node2]:
            new_node1 = np.uint32(remap[node1])
            new_node2 = np.uint32(remap[node2])
            gc_edge_list.append([new_node1, new_node2])

    return np.array(gc_edge_list, dtype=np.uint32)


# ================= Clustering coefficient =================
@njit(fastmath=True)
def get_average_clustering_coefficient(
    degrees: npt.NDArray[np.uint32],
    offsets: npt.NDArray[np.uint32],
    neighbors: npt.NDArray[np.uint32],
) -> np.float32:
    """
    Compute the average local clustering coefficient over all nodes. The graph should be connected.

    Args:
        degrees: [N, ]. Node degrees.
        offsets: [N + 1, ]. CSR offsets.
        neighbors: [2E, ]. CSR neighbor list.

    Returns:
        Average clustering coefficient.
    """
    num_nodes = len(degrees)
    marker = np.zeros(num_nodes, dtype=np.bool_)

    local_sum = np.float64(0.0)
    for node, degree in enumerate(degrees):
        if degree < 2:
            continue
        start, end = offsets[node], offsets[node + 1]

        # Mark neighbors of the center node for O(1)-style triangle checks.
        for neighbor in neighbors[start:end]:
            marker[neighbor] = True

        links = 0
        for neighbor in neighbors[start:end]:
            neighbor_start = offsets[neighbor]
            neighbor_end = offsets[neighbor + 1]
            for nn in neighbors[neighbor_start:neighbor_end]:
                if marker[nn] and neighbor < nn:
                    links += 1

        for neighbor in neighbors[start:end]:
            marker[neighbor] = False

        local_sum += np.float64(2.0 * links / (degree * (degree - 1)))

    return np.float32(local_sum / num_nodes)


# ================= Spectral properties =================
def get_algebraic_connectivity(
    num_nodes: int,
    edge_list: npt.NDArray[np.uint32],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.float32:
    """
    Compute algebraic connectivity 2nd-smallest unnormalized Laplacian eigenvalue

    Args:
        num_nodes: Number of nodes.
        edge_list: [E, 2]
        max_iter: Maximum ARPACK iterations used by `eigsh`.
        tol: ARPACK convergence tolerance used by `eigsh`.

    Returns:
        Algebraic connectivity (`lambda_2`) of the input graph.
        If the input graph is disconnected, `lambda_2` is approximately 0.
    """
    adjacency = edge_list_to_sparse_adjacency(num_nodes, edge_list)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sparse.diags(degrees, dtype=np.float64) - adjacency

    eigenvalues = sparse.linalg.eigsh(
        laplacian,
        k=2,
        which="SM",  # smallest
        return_eigenvectors=False,
        tol=tol,
        maxiter=max_iter,
    )

    eigenvalues = np.sort(eigenvalues)
    return np.float32(max(0.0, float(eigenvalues[1])))


def get_spectral_radius(
    num_nodes: int,
    edge_list: npt.NDArray[np.uint32],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.float32:
    """
    Compute spectral radius largest adjacency eigenvalue

    Args:
        edge_list: Edge list with shape `[E, 2]`.
        num_nodes: Number of nodes `N`.
        max_iter: Maximum ARPACK iterations used by `eigsh`.
        tol: ARPACK convergence tolerance used by `eigsh`.

    Returns:
        Spectral radius of the adjacency matrix.
    """
    adjacency = edge_list_to_sparse_adjacency(num_nodes, edge_list)

    eigenvalues = sparse.linalg.eigsh(
        adjacency,
        k=1,
        which="LA",  # largest
        return_eigenvectors=False,
        tol=tol,
        maxiter=max_iter,
    )

    return np.float32(max(0.0, float(np.max(eigenvalues))))


# ================= Distance based properties =================
@njit(fastmath=True)
def _bfs_distance(
    start_node: int,
    offsets: npt.NDArray[np.uint32],
    neighbors: npt.NDArray[np.uint32],
    distances: npt.NDArray[np.int64],
    queue: npt.NDArray[np.int64],
) -> None:
    """
    Run BFS on a CSR graph and write shortest-path distances from one source.

    Args:
        start_node: BFS source node.
        offsets: [N+1, ], CSR offsets.
        neighbors: [2E, ], CSR neighbor list.
        distances: Output buffer with shape `[N]` (overwritten in-place).
        queue: Work buffer with shape `[N]`.
    """
    num_nodes = len(distances)
    for node_idx in range(num_nodes):
        distances[node_idx] = -1

    head = 0
    tail = 0
    queue[tail] = start_node
    tail += 1
    distances[start_node] = 0

    while head < tail:
        node = queue[head]
        head += 1
        base_distance = distances[node]

        begin = offsets[node]
        end = offsets[node + 1]
        for nbr_pos in range(begin, end):
            nbr = int(neighbors[nbr_pos])
            if distances[nbr] == -1:
                distances[nbr] = base_distance + 1
                queue[tail] = nbr
                tail += 1


@njit(fastmath=True)
def get_distances(
    offsets: npt.NDArray[np.uint32], neighbors: npt.NDArray[np.uint32]
) -> npt.NDArray[np.uint32]:
    """
    Compute all-pairs shortest-path distances from CSR graph. The graph should be connected.

    Args:
        offsets: [N + 1, ]. CSR offsets.
        neighbors: [2E, ]. CSR neighbor list.

    Returns:
        [N, N] shortest-path distance matrix.
    """
    num_nodes = len(offsets) - 1
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    distances = np.empty(num_nodes, dtype=np.int64)
    queue = np.empty(num_nodes, dtype=np.int64)

    for source_node in range(num_nodes):
        _bfs_distance(source_node, offsets, neighbors, distances, queue)
        for target_node in range(source_node, num_nodes):
            distance = distances[target_node]
            distance_matrix[source_node, target_node] = distance
            distance_matrix[target_node, source_node] = distance

    return distance_matrix.astype(np.uint32)
