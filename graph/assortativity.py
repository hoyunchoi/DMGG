import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import njit

from graph.edge_list import canonical_edge, edge_list_to_degrees, graph_to_edge_list
from graph.generator import _edge_key, _next_pow2

SWAP_TRIALS: float = 50.0
STAGNATION_LIMIT: float = 1.0
_EMPTY_KEY = np.int64(-1)
_DELETED_KEY = np.int64(-2)


@njit(fastmath=True)
def get_joint_degree_matrix(
    edge_list: npt.NDArray[np.uint32], degrees: npt.NDArray[np.uint32]
) -> npt.NDArray[np.uint32]:
    """
    Build the joint degree matrix for an undirected graph.

    Args:
        edge_list: [E, 2] undirected edges where each row is `(u, v)`.
        degrees: [N] degree of each node.

    Returns:
        [D, D] matrix where D is max degree and entry (i, j) counts
        oriented edge-ends between degree (i+1) and degree (j+1).
    """
    max_degree = degrees.max()
    joint_degree_matrix = np.zeros((max_degree, max_degree), dtype=np.uint32)

    for edge in edge_list:
        degree1, degree2 = degrees[edge]
        joint_degree_matrix[degree1 - 1, degree2 - 1] += 1
        joint_degree_matrix[degree2 - 1, degree1 - 1] += 1
    return joint_degree_matrix


@njit(fastmath=True)
def get_marginal_prob(
    joint_degree_matrix: npt.NDArray[np.uint32], num_edges: int
) -> npt.NDArray[np.float32]:
    """
    Compute degree marginals from a joint degree matrix.

    Args:
        joint_degree_matrix: [D, D].
        num_edges: scalar E.

    Returns:
        [D] marginal probability of each degree value.
    """
    return joint_degree_matrix.sum(axis=0).astype(np.float32) / np.float32(
        2 * num_edges
    )


@njit(fastmath=True)
def get_degree_variance(
    max_degree: int, marginal_prob: npt.NDArray[np.float32]
) -> np.float32:
    """
    Compute the degree variance used in assortativity normalization.

    Args:
        max_degree: scalar D.
        marginal_prob: [D, ] degree marginal probabilities.

    Returns:
        scalar variance term in Newman assortativity formula.
    """
    degree_range = np.arange(1, max_degree + 1, dtype=np.float32)

    return np.float32(
        np.average(degree_range**2, weights=marginal_prob)
        - np.average(degree_range, weights=marginal_prob) ** 2
    )


@njit(fastmath=True)
def get_assortativity(
    joint_degree_matrix: npt.NDArray[np.uint32],
    num_edges: int,
    marginal_prob: npt.NDArray[np.float32],
    degrees_variance: np.float32,
) -> np.float32:
    """
    Compute assortativity from cached joint degree statistics.

    Args:
        joint_degree_matrix: [D, D]
        num_edges: scalar E.
        marginal_prob: [D] degree marginal probabilities.
        degrees_variance: scalar normalization term.

    Returns:
        scalar assortativity coefficient.
    """
    max_degree = len(joint_degree_matrix)

    degree_product_sum = 0
    for degree1 in range(1, max_degree + 1):
        for degree2 in range(1, max_degree + 1):
            degree_product_sum += (
                degree1 * degree2 * joint_degree_matrix[degree1 - 1, degree2 - 1]
            )
    degree_product_mean = np.float32(0.5 * degree_product_sum / num_edges)

    mean_degree_term: int = 0
    for degree in range(1, max_degree + 1):
        mean_degree_term += degree * marginal_prob[degree - 1]

    return np.float32(
        (degree_product_mean - np.float32(mean_degree_term) ** 2.0) / degrees_variance
    )


@njit(fastmath=True)
def get_assortativity_from_edge_list(
    degrees: npt.NDArray[np.uint32],
    edge_list: npt.NDArray[np.uint32],
) -> np.float32:
    """
    Compute assortativity from edge list and authoritative node count.

    Args:
        num_nodes: Number of nodes in the graph.
        edge_list: [E, 2] edge list.

    Returns:
        Assortativity coefficient, or 0.0 when undefined.
    """
    num_edges = len(edge_list)

    joint_degree_matrix = get_joint_degree_matrix(edge_list, degrees)
    marginal_prob = get_marginal_prob(joint_degree_matrix, num_edges)
    degree_variance = get_degree_variance(int(degrees.max()), marginal_prob)

    return get_assortativity(
        joint_degree_matrix, num_edges, marginal_prob, degree_variance
    )


def get_assortativity_from_graph(graph: nx.Graph) -> np.float32:
    """
    Compute assortativity from a NetworkX graph.

    Args:
        graph: simple undirected graph.

    Returns:
        scalar assortativity coefficient.
    """
    edge_list = graph_to_edge_list(graph)
    degrees = edge_list_to_degrees(graph.number_of_nodes(), edge_list)
    joint_degree_matrix = get_joint_degree_matrix(edge_list, degrees)
    num_edges = len(edge_list)
    marginal_prob = get_marginal_prob(joint_degree_matrix, num_edges)
    degree_variance = get_degree_variance(degrees.max(), marginal_prob)
    return get_assortativity(
        joint_degree_matrix, num_edges, marginal_prob, degree_variance
    )


@njit(fastmath=True)
def _hash_contains(keys: npt.NDArray[np.int64], key: np.int64) -> bool:
    """Check whether key exists in open-addressing hash table."""
    mask = keys.size - 1
    slot_idx = key & np.int64(mask)
    while True:
        existing_key = keys[slot_idx]
        if existing_key == _EMPTY_KEY:
            return False
        if existing_key == key:
            return True
        slot_idx = (slot_idx + 1) & mask


@njit(fastmath=True)
def _hash_insert_or_keep(keys: npt.NDArray[np.int64], key: np.int64) -> bool:
    """Insert key into table. Return False only if key already exists."""
    mask = keys.size - 1
    slot_idx = key & np.int64(mask)
    first_deleted = -1
    while True:
        existing_key = keys[slot_idx]
        if existing_key == _EMPTY_KEY:
            target_idx = first_deleted if first_deleted >= 0 else slot_idx
            keys[target_idx] = key
            return True
        if existing_key == key:
            return False
        if existing_key == _DELETED_KEY and first_deleted < 0:
            first_deleted = slot_idx
        slot_idx = (slot_idx + 1) & mask


@njit(fastmath=True)
def _hash_remove(keys: npt.NDArray[np.int64], key: np.int64) -> bool:
    """Remove key from table if it exists."""
    mask = keys.size - 1
    slot_idx = key & np.int64(mask)
    while True:
        existing_key = keys[slot_idx]
        if existing_key == _EMPTY_KEY:
            return False
        if existing_key == key:
            keys[slot_idx] = _DELETED_KEY
            return True
        slot_idx = (slot_idx + 1) & mask


@njit(fastmath=True)
def _sample_rewire_candidate(
    num_nodes: int,
    edge_list: npt.NDArray[np.uint32],
    edge_keys: npt.NDArray[np.int64],
    degrees: npt.NDArray[np.uint32],
    maximize: bool,
    rng: np.random.Generator,
) -> tuple[int, int, int, np.int64]:
    """
    Sample one ERGM-style candidate and choose mode aligned with optimization goal.

    Returns:
        (edge1_idx, edge2_idx, mode, delta). If sampling fails, returns mode = -1.
    """
    num_edges = len(edge_list)

    while True:
        edge1_idx = int(rng.integers(0, num_edges))
        edge2_idx = int(rng.integers(0, num_edges - 1))
        if edge2_idx >= edge1_idx:
            edge2_idx += 1

        node1 = int(edge_list[edge1_idx, 0])
        node2 = int(edge_list[edge1_idx, 1])
        node3 = int(edge_list[edge2_idx, 0])
        node4 = int(edge_list[edge2_idx, 1])

        # Shared endpoints imply self-loop or pointless rewiring.
        if node1 == node3 or node1 == node4 or node2 == node3 or node2 == node4:
            continue

        mode0_edge1 = canonical_edge(node1, node3)
        mode0_edge2 = canonical_edge(node2, node4)
        mode0_key1 = _edge_key(num_nodes, mode0_edge1[0], mode0_edge1[1])
        mode0_key2 = _edge_key(num_nodes, mode0_edge2[0], mode0_edge2[1])
        mode0_invalid = (
            (mode0_edge1 == mode0_edge2)
            or _hash_contains(edge_keys, mode0_key1)
            or _hash_contains(edge_keys, mode0_key2)
        )

        mode1_edge1 = canonical_edge(node1, node4)
        mode1_edge2 = canonical_edge(node2, node3)
        mode1_key1 = _edge_key(num_nodes, mode1_edge1[0], mode1_edge1[1])
        mode1_key2 = _edge_key(num_nodes, mode1_edge2[0], mode1_edge2[1])
        mode1_invalid = (
            (mode1_edge1 == mode1_edge2)
            or _hash_contains(edge_keys, mode1_key1)
            or _hash_contains(edge_keys, mode1_key2)
        )

        if mode0_invalid and mode1_invalid:
            continue

        old_objective = np.int64(degrees[node1]) * np.int64(degrees[node2]) + np.int64(
            degrees[node3]
        ) * np.int64(degrees[node4])
        mode0_delta = (
            np.int64(degrees[node1]) * np.int64(degrees[node3])
            + np.int64(degrees[node2]) * np.int64(degrees[node4])
            - old_objective
        )
        mode1_delta = (
            np.int64(degrees[node1]) * np.int64(degrees[node4])
            + np.int64(degrees[node2]) * np.int64(degrees[node3])
            - old_objective
        )

        if mode0_invalid:
            return edge1_idx, edge2_idx, 1, mode1_delta
        if mode1_invalid:
            return edge1_idx, edge2_idx, 0, mode0_delta

        if maximize:
            if mode0_delta >= mode1_delta:
                return edge1_idx, edge2_idx, 0, mode0_delta
            return edge1_idx, edge2_idx, 1, mode1_delta
        if mode0_delta <= mode1_delta:
            return edge1_idx, edge2_idx, 0, mode0_delta
        return edge1_idx, edge2_idx, 1, mode1_delta


@njit(fastmath=True)
def _optimize_extreme_graph(
    degrees: npt.NDArray[np.uint32],
    edge_list: npt.NDArray[np.uint32],
    maximize: bool,
    rng: np.random.Generator,
    swap_trials: float,
    stagnation_limit: float,
) -> npt.NDArray[np.uint32]:
    """Numba-optimized core loop for degree-preserving extreme rewiring."""
    num_nodes = len(degrees)
    num_edges = len(edge_list)

    table_size = _next_pow2(max(8, int(num_edges * 4)))
    edge_keys = np.full(table_size, _EMPTY_KEY, dtype=np.int64)
    for edge_idx in range(num_edges):
        node1 = int(edge_list[edge_idx, 0])
        node2 = int(edge_list[edge_idx, 1])
        _hash_insert_or_keep(edge_keys, _edge_key(num_nodes, node1, node2))

    stagnation_limit = int(stagnation_limit * num_edges)
    no_accept_count = 0

    for _ in range(int(swap_trials * num_edges)):
        edge1_idx, edge2_idx, mode, delta = _sample_rewire_candidate(
            num_nodes=num_nodes,
            edge_list=edge_list,
            edge_keys=edge_keys,
            degrees=degrees,
            maximize=maximize,
            rng=rng,
        )

        # If delta goes in the opposite direction, do not apply this rewiring.
        is_improving = (delta > 0) if maximize else (delta < 0)
        if not is_improving:
            no_accept_count += 1
            if no_accept_count >= stagnation_limit:
                # print("Stagnation detected")
                break
            continue

        old_node1 = int(edge_list[edge1_idx, 0])
        old_node2 = int(edge_list[edge1_idx, 1])
        old_node3 = int(edge_list[edge2_idx, 0])
        old_node4 = int(edge_list[edge2_idx, 1])
        old_key1 = _edge_key(num_nodes, old_node1, old_node2)
        old_key2 = _edge_key(num_nodes, old_node3, old_node4)

        if mode == 0:
            new_edge1 = canonical_edge(old_node1, old_node3)
            new_edge2 = canonical_edge(old_node2, old_node4)
        else:
            new_edge1 = canonical_edge(old_node1, old_node4)
            new_edge2 = canonical_edge(old_node2, old_node3)
        new_key1 = _edge_key(num_nodes, new_edge1[0], new_edge1[1])
        new_key2 = _edge_key(num_nodes, new_edge2[0], new_edge2[1])

        _hash_remove(edge_keys, old_key1)
        _hash_remove(edge_keys, old_key2)
        _hash_insert_or_keep(edge_keys, new_key1)
        _hash_insert_or_keep(edge_keys, new_key2)

        edge_list[edge1_idx, 0] = np.uint32(new_edge1[0])
        edge_list[edge1_idx, 1] = np.uint32(new_edge1[1])
        edge_list[edge2_idx, 0] = np.uint32(new_edge2[0])
        edge_list[edge2_idx, 1] = np.uint32(new_edge2[1])
        no_accept_count = 0

    return edge_list


def maximally_assortative_graph(
    degrees: npt.NDArray[np.uint32],
    edge_list: npt.NDArray[np.uint32],
    rng: np.random.Generator | int | None = None,
    swap_trials: float = SWAP_TRIALS,
    stagnation_limit: float = STAGNATION_LIMIT,
) -> npt.NDArray[np.uint32]:
    """
    Create (approximately) maximally assortative graph from given degree sequence
    That is, high-degree nodes connect to other high-degree nodes first

    Args:
        degrees: [N, ] degree sequence
        edge_list: [E, 2] edge list of starting graph, specifying degree sequence
        rng: random number generator or seed (optional)

    Return:
        edge_list: [E, 2] edge list of (approximately) maximally assortative graph
                   If error occurs during finding such graph, return empty edge_list (E=0)
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    return _optimize_extreme_graph(
        degrees=degrees.copy(),
        edge_list=edge_list.copy(),
        maximize=True,
        rng=rng,
        swap_trials=swap_trials,
        stagnation_limit=stagnation_limit,
    )


def maximally_disassortative_graph(
    degrees: npt.NDArray[np.uint32],
    edge_list: npt.NDArray[np.uint32],
    rng: np.random.Generator | int | None = None,
    swap_trials: float = SWAP_TRIALS,
    stagnation_limit: float = STAGNATION_LIMIT,
) -> npt.NDArray[np.uint32]:
    """
    Create (approximately) maximally disassortative graph from given degree sequence
    That is, high-degree nodes connect to low-degree nodes first
    Args:
        degrees: [N, ] degree sequence
        edge_list: [E, 2] edge list of starting graph, specifying degree sequence
        rng: random number generator or seed (optional)

    Returns:
        edge_list: [E, 2] edge list of (approximately) maximally disassortative graph
                   If error occurs during finding such graph, return empty edge_list (E=0)
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    return _optimize_extreme_graph(
        degrees=degrees.copy(),
        edge_list=edge_list.copy(),
        maximize=False,
        rng=rng,
        swap_trials=swap_trials,
        stagnation_limit=stagnation_limit,
    )


def get_maximum_assortativity(
    degrees: npt.NDArray[np.uint32],
    edge_list: npt.NDArray[np.uint32],
    rng: np.random.Generator | int | None = None,
    swap_trials: float = SWAP_TRIALS,
    stagnation_limit: float = STAGNATION_LIMIT,
) -> np.float32 | None:
    """
    Compute (approximately) maximum assortativity from given degree sequence
    Args:
        degrees: [N, ] degree sequence
        edge_list: [E, 2] edge list of starting graph, specifying degree sequence
        rng: random number generator or seed (optional)

    Returns:
        max_assort: maximum assortativity value. If error occurs during finding such graph, return 0.0
    """
    max_graph_edge_list = maximally_assortative_graph(
        degrees, edge_list, rng, swap_trials, stagnation_limit
    )
    if len(max_graph_edge_list) == 0:
        return None
    return get_assortativity_from_edge_list(degrees, max_graph_edge_list)


def get_minimum_assortativity(
    degrees: npt.NDArray[np.uint32],
    edge_list: npt.NDArray[np.uint32],
    rng: np.random.Generator | int | None = None,
    swap_trials: float = SWAP_TRIALS,
    stagnation_limit: float = STAGNATION_LIMIT,
) -> np.float32 | None:
    """
    Compute (approximately) minimum assortativity from given degree sequence
    Args:
        degrees: [N, ] degree sequence
        edge_list: [E, 2] edge list of starting graph, specifying degree sequence
        rng: random number generator or seed (optional)

    Returns:
        min_assort: minimum assortativity value. If error occurs during finding such graph, return 0.0
    """
    min_graph_edge_list = maximally_disassortative_graph(
        degrees, edge_list, rng, swap_trials, stagnation_limit
    )
    if len(min_graph_edge_list) == 0:
        return None
    return get_assortativity_from_edge_list(degrees, min_graph_edge_list)
