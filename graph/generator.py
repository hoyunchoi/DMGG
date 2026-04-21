from typing import Callable

import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import njit

from graph.edge_list import canonical_edge
from graph.type_aliases import NODE


# ---------- Hash utilities (open addressing) -------
@njit(fastmath=True)
def _next_pow2(value: int) -> int:
    next_power = 1
    while next_power < value:
        next_power <<= 1
    return next_power


@njit(fastmath=True)
def _hash_insert(keys: np.ndarray, key: np.int64) -> bool:
    """Insert key if absent. Return True if inserted, False if already exists."""
    mask = keys.size - 1
    slot_index = np.int64(key) & np.int64(mask)
    while True:
        existing_key = keys[slot_index]
        if existing_key == -1:
            keys[slot_index] = key
            return True
        if existing_key == key:
            return False
        slot_index = (slot_index + 1) & mask


@njit(fastmath=True)
def _edge_key(num_nodes: int, node1: NODE, node2: NODE) -> np.int64:
    # assumes canonical edge (0 <= u < v < num_nodes)
    return np.int64(node1) * np.int64(num_nodes) + np.int64(node2)


@njit(fastmath=True)
def _sample_from_cumulative_weights(
    cumulative_weights: np.ndarray, total_weight: float, rng: np.random.Generator
) -> int:
    threshold = rng.random() * total_weight
    sampled_index = 0
    last_index = cumulative_weights.size - 1
    while sampled_index < last_index and threshold >= cumulative_weights[sampled_index]:
        sampled_index += 1
    return sampled_index


# ---------- Erdős–Rényi random graph -------
@njit(fastmath=True)
def get_er(
    num_nodes: int, mean_degree: float, rng: np.random.Generator
) -> npt.NDArray[np.uint32]:
    """
    Erdős–Rényi random graph with G(n, m) convention
    """
    num_edges = int(0.5 * mean_degree * num_nodes)
    edge_list = np.empty((num_edges, 2), dtype=np.uint32)

    # load factor ~0.4
    table_size = _next_pow2(max(8, int(num_edges * 3)))
    keys = np.full(table_size, -1, dtype=np.int64)

    pos = 0
    while pos < num_edges:
        node1 = int(rng.integers(0, num_nodes))
        node2 = int(rng.integers(0, num_nodes - 1))
        if node2 >= node1:
            node2 += 1
        if node1 == node2:
            continue

        node1, node2 = canonical_edge(node1, node2)
        key = _edge_key(num_nodes, node1, node2)
        if _hash_insert(keys, key):
            edge_list[pos, 0] = node1
            edge_list[pos, 1] = node2
            pos += 1

    return edge_list

# ---------- Watts–Strogatz random graph -------
@njit(fastmath=True)
def get_ws(
    num_nodes: int,
    mean_degree: float,
    rng: np.random.Generator,
    rewire_p_range: tuple[float, float] = (0.02, 0.05),
) -> np.ndarray:
    """
    Generate WS edges by iterating each node's forward neighbors once.
    Ensures no self-loops and no multi-edges (via hash set).
    """
    rewire_p = rng.uniform(*rewire_p_range)

    even_mean_degree = int(round(mean_degree))
    if even_mean_degree % 2 == 1:
        even_mean_degree += 1

    half = even_mean_degree // 2
    num_edges = num_nodes * half
    edge_list = np.empty((num_edges, 2), dtype=np.uint32)

    table_size = _next_pow2(max(8, int(num_edges * 3)))
    keys = np.full(table_size, -1, dtype=np.int64)

    pos = 0
    for source_node in range(num_nodes):
        for neighbor_offset in range(1, half + 1):
            if rng.random() < rewire_p:
                # Rewire: pick a random v not equal u and not already connected.
                while True:
                    target_node = int(rng.integers(0, num_nodes - 1))
                    if target_node >= source_node:
                        target_node += 1
                    edge_node1, edge_node2 = canonical_edge(source_node, target_node)
                    key = _edge_key(num_nodes, edge_node1, edge_node2)
                    if _hash_insert(keys, key):
                        edge_list[pos, 0] = edge_node1
                        edge_list[pos, 1] = edge_node2
                        pos += 1
                        break
            else:
                target_node = (source_node + neighbor_offset) % num_nodes
                edge_node1, edge_node2 = canonical_edge(source_node, target_node)
                key = _edge_key(num_nodes, edge_node1, edge_node2)
                if _hash_insert(keys, key):
                    edge_list[pos, 0] = edge_node1
                    edge_list[pos, 1] = edge_node2
                    pos += 1
                else:
                    # Extremely rare unless mean_degree is close to num_nodes; resample like rewiring.
                    while True:
                        target_node = int(rng.integers(0, num_nodes - 1))
                        if target_node >= source_node:
                            target_node += 1
                        edge_node1, edge_node2 = canonical_edge(
                            source_node, target_node
                        )
                        key2 = _edge_key(num_nodes, edge_node1, edge_node2)
                        if _hash_insert(keys, key2):
                            edge_list[pos, 0] = edge_node1
                            edge_list[pos, 1] = edge_node2
                            pos += 1
                            break

    return edge_list[:pos]


# ---------- Chung-Lu random graph -------
@njit(fastmath=True)
def get_cl(
    num_nodes: int, mean_degree: float, rng: np.random.Generator
) -> npt.NDArray[np.uint32]:
    """
    Chung-Lu random graph (proper independent-edge model).
    Each pair (i, j) is connected independently with probability
    p_ij = min(1, w_i * w_j / sum(w)).
    Reference: Miller & Hagberg, WAW 2011.
    """
    num_edges = int(round(0.5 * mean_degree * num_nodes))

    # power-law exponent gamma ~ Uniform(2, 3)
    exponent = rng.uniform(2.0, 3.0)
    inv_exponent_minus_one = 1.0 / (exponent - 1.0)

    # sample raw expected degrees (Pareto, xmin=1), then scale to sum_w = 2*num_edges
    weights = np.empty(num_nodes, dtype=np.float64)
    max_weight = float(num_nodes - 1)
    for node_idx in range(num_nodes):
        # Randomly assign degree weights from Pareto distribution
        weight = (1.0 - rng.random()) ** (-inv_exponent_minus_one)  # >= 1
        if weight > max_weight:
            weight = max_weight
        weights[node_idx] = weight

    # Normalize weights so that sum(w) = 2 * num_edges_expected
    sum_w = np.sum(weights)
    weights *= 2.0 * num_edges / sum_w
    sum_w = 2.0 * num_edges

    # Sort weights descending; keep mapping to original node indices
    order = np.argsort(-weights)
    sorted_weights = np.empty(num_nodes, dtype=np.float64)
    for sorted_idx in range(num_nodes):
        sorted_weights[sorted_idx] = weights[order[sorted_idx]]

    inverse_weight_sum = 1.0 / sum_w

    # Allocate edge buffer (actual count is random; 2x expected is very safe)
    edge_list = np.empty((num_edges * 2, 2), dtype=np.uint32)
    pos = 0

    for source_node_idx in range(num_nodes - 1):
        target_node_idx = source_node_idx + 1
        pair_scale = sorted_weights[source_node_idx] * inverse_weight_sum
        current_prob = sorted_weights[target_node_idx] * pair_scale
        if current_prob > 1.0:
            current_prob = 1.0
        while target_node_idx < num_nodes and current_prob > 0.0:
            if current_prob < 1.0:
                random_value = rng.random()
                if random_value > 0.0:
                    log_1_minus_p = np.log(1.0 - current_prob)
                    if log_1_minus_p < 0.0:
                        target_node_idx += int(
                            np.floor(np.log(random_value) / log_1_minus_p)
                        )
                else:
                    break  # r == 0 → skip all remaining in this row
            if target_node_idx < num_nodes:
                next_prob = sorted_weights[target_node_idx] * pair_scale
                if next_prob > 1.0:
                    next_prob = 1.0
                if rng.random() < next_prob / current_prob:
                    source_node = int(order[source_node_idx])
                    target_node = int(order[target_node_idx])
                    if source_node > target_node:
                        source_node, target_node = target_node, source_node
                    edge_list[pos, 0] = np.uint32(source_node)
                    edge_list[pos, 1] = np.uint32(target_node)
                    pos += 1
                target_node_idx += 1
                current_prob = next_prob

    return edge_list[:pos]


# ---------- Preferential attachment -------
@njit(fastmath=True)
def _sample_preferential_target(
    endpoint_pool: npt.NDArray[np.uint32], pool_len: int, rng: np.random.Generator
) -> int:
    random_pool_index = rng.integers(0, pool_len)
    return int(endpoint_pool[random_pool_index])


@njit
def mean_degree_to_m(mean_degree: float, num_nodes: int) -> int:
    links_per_new_node_floor = max(1, int(mean_degree / 2))
    mean_degree_floor = (
        links_per_new_node_floor
        * (2 * num_nodes - links_per_new_node_floor - 1)
        / num_nodes
    )
    error_floor = abs(mean_degree - mean_degree_floor)

    links_per_new_node_ceil = min(num_nodes - 1, links_per_new_node_floor + 1)
    mean_degree_ceil = (
        links_per_new_node_ceil
        * (2 * num_nodes - links_per_new_node_ceil - 1)
        / num_nodes
    )
    error_ceil = abs(mean_degree - mean_degree_ceil)

    links_per_new_node = (
        links_per_new_node_floor
        if error_floor < error_ceil
        else links_per_new_node_ceil
    )
    return links_per_new_node


@njit(fastmath=True)
def get_ba(
    num_nodes: int, mean_degree: float, rng: np.random.Generator
) -> npt.NDArray[np.uint32]:
    """
    Barabási–Albert edges.
    Starts with a complete graph on (m+1) nodes, then adds m edges per new node.
    """
    links_per_new_node = mean_degree_to_m(mean_degree, num_nodes)

    # Total edges in NetworkX-style BA:
    # E = (m+1)*m/2 + (num_nodes - (m+1))*m
    num_edges = (links_per_new_node + 1) * links_per_new_node // 2 + (
        num_nodes - (links_per_new_node + 1)
    ) * links_per_new_node
    edge_list = np.empty((num_edges, 2), dtype=np.uint32)

    table_size = _next_pow2(max(8, int(num_edges * 3)))
    keys = np.full(table_size, -1, dtype=np.int64)

    # Endpoint pool size = 2E
    endpoint_pool = np.empty(2 * num_edges, dtype=np.uint32)

    pos, pool_len = 0, 0
    # Initial clique on nodes [0..m]
    for node1 in range(links_per_new_node + 1):
        for node2 in range(node1 + 1, links_per_new_node + 1):
            key = _edge_key(num_nodes, node1, node2)
            _hash_insert(keys, key)
            edge_list[pos, 0] = node1
            edge_list[pos, 1] = node2
            pos += 1
            endpoint_pool[pool_len] = node1
            endpoint_pool[pool_len + 1] = node2
            pool_len += 2

    chosen = np.empty(links_per_new_node, dtype=np.int32)

    for new_node in range(links_per_new_node + 1, num_nodes):
        chosen_len = 0
        while chosen_len < links_per_new_node:
            target = _sample_preferential_target(endpoint_pool, pool_len, rng)
            if target == new_node:
                continue

            duplicated = False
            for chosen_idx in range(chosen_len):
                if chosen[chosen_idx] == target:
                    duplicated = True
                    break
            if duplicated:
                continue

            node1, node2 = canonical_edge(new_node, target)
            key = _edge_key(num_nodes, node1, node2)
            if not _hash_insert(keys, key):
                continue

            edge_list[pos, 0] = node1
            edge_list[pos, 1] = node2
            pos += 1

            endpoint_pool[pool_len] = node1
            endpoint_pool[pool_len + 1] = node2
            pool_len += 2

            chosen[chosen_len] = target
            chosen_len += 1

    return edge_list[:pos]


@njit(fastmath=True)
def get_hk(
    num_nodes: int,
    mean_degree: float,
    rng: np.random.Generator,
    triangle_prob_range: tuple[float, float] = (0.55, 0.8),
) -> npt.NDArray[np.uint32]:
    """
    Holme–Kim (powerlaw + clustering) edge list.
    This is a lightweight implementation using:
    - preferential attachment via endpoint_pool
    - triadic closure by connecting to a uniformly random neighbor
      of the most recently connected target
    """
    triangle_prob = rng.uniform(*triangle_prob_range)

    links_per_new_node = mean_degree_to_m(mean_degree, num_nodes)

    num_edges = (links_per_new_node + 1) * links_per_new_node // 2 + (
        num_nodes - (links_per_new_node + 1)
    ) * links_per_new_node
    edge_list = np.empty((num_edges, 2), dtype=np.uint32)

    table_size = _next_pow2(max(8, int(num_edges * 3)))
    keys = np.full(table_size, -1, dtype=np.int64)

    endpoint_pool = np.empty(2 * num_edges, dtype=np.uint32)
    pos, pool_len = 0, 0

    # Buffer for collecting neighbors during triadic closure
    neighbors = np.empty(num_nodes, dtype=np.uint32)

    # Initial clique
    for node1 in range(links_per_new_node + 1):
        for node2 in range(node1 + 1, links_per_new_node + 1):
            key = _edge_key(num_nodes, node1, node2)
            _hash_insert(keys, key)
            edge_list[pos, 0] = np.uint32(node1)
            edge_list[pos, 1] = np.uint32(node2)
            pos += 1
            endpoint_pool[pool_len] = node1
            endpoint_pool[pool_len + 1] = node2
            pool_len += 2

    chosen = np.empty(links_per_new_node, dtype=np.int32)

    for new_node in range(links_per_new_node + 1, num_nodes):
        chosen_len = 0

        # First edge: preferential attachment
        while True:
            target = _sample_preferential_target(endpoint_pool, pool_len, rng)
            if target == new_node:
                continue
            node1, node2 = canonical_edge(new_node, target)
            key = _edge_key(num_nodes, node1, node2)
            if _hash_insert(keys, key):
                edge_list[pos, 0] = np.uint32(node1)
                edge_list[pos, 1] = np.uint32(node2)
                pos += 1
                endpoint_pool[pool_len] = node1
                endpoint_pool[pool_len + 1] = node2
                pool_len += 2
                chosen[0] = target
                chosen_len = 1
                last_pa_target = target  # track last PA-connected target
                break

        # Remaining m-1 edges: triadic closure or preferential attachment
        while chosen_len < links_per_new_node:
            if rng.random() < triangle_prob:
                # --- TF step: neighbors of LAST PA TARGET (not last connected) ---
                tf_source = np.uint32(last_pa_target)

                # Collect actual neighbors of tf_source
                n_count = 0
                for edge_idx in range(pos):
                    src_node = edge_list[edge_idx, 0]
                    dst_node = edge_list[edge_idx, 1]
                    if src_node == tf_source:
                        neighbors[n_count] = dst_node
                        n_count += 1
                    elif dst_node == tf_source:
                        neighbors[n_count] = src_node
                        n_count += 1

                # Pre-filter: keep only valid candidates
                valid_count = 0
                new_node_u32 = np.uint32(new_node)
                for neighbor_idx in range(n_count):
                    neighbor_node = neighbors[neighbor_idx]
                    if neighbor_node == new_node_u32:
                        continue
                    is_chosen = False
                    for chosen_idx in range(chosen_len):
                        if chosen[chosen_idx] == np.int32(neighbor_node):
                            is_chosen = True
                            break
                    if is_chosen:
                        continue
                    neighbors[valid_count] = neighbor_node  # reuse buffer
                    valid_count += 1

                if valid_count > 0:
                    # Guaranteed success: pick one uniformly at random
                    chosen_neighbor = int(neighbors[rng.integers(0, valid_count)])
                    node1, node2 = canonical_edge(new_node, chosen_neighbor)
                    key = _edge_key(num_nodes, node1, node2)
                    _hash_insert(keys, key)
                    edge_list[pos, 0] = np.uint32(node1)
                    edge_list[pos, 1] = np.uint32(node2)
                    pos += 1
                    endpoint_pool[pool_len] = node1
                    endpoint_pool[pool_len + 1] = node2
                    pool_len += 2
                    chosen[chosen_len] = chosen_neighbor
                    chosen_len += 1
                    continue  # ← last_pa_target NOT updated!

            # PA fallback (also reached when TF fails)
            while True:
                target = _sample_preferential_target(endpoint_pool, pool_len, rng)
                if target == new_node:
                    continue
                duplicated = False
                for chosen_idx in range(chosen_len):
                    if chosen[chosen_idx] == target:
                        duplicated = True
                        break
                if duplicated:
                    continue
                node1, node2 = canonical_edge(new_node, target)
                key = _edge_key(num_nodes, node1, node2)
                if _hash_insert(keys, key):
                    edge_list[pos, 0] = np.uint32(node1)
                    edge_list[pos, 1] = np.uint32(node2)
                    pos += 1
                    endpoint_pool[pool_len] = node1
                    endpoint_pool[pool_len + 1] = node2
                    pool_len += 2
                    chosen[chosen_len] = target
                    chosen_len += 1
                    last_pa_target = target  # ← UPDATE only on PA!
                    break

    return edge_list[:pos]


# ---------- Stochastic block model -------
@njit(fastmath=True)
def get_sbm(
    num_nodes: int,
    mean_degree: float,
    rng: np.random.Generator,
    max_blocks: int = 5,
    community_strength_range: tuple[float, float] = (0.7, 0.85),
) -> npt.NDArray[np.uint32]:
    """
    Sparse SBM-style generator with fixed edge count (G(n, m)-like).
    - Balanced block sizes to reduce degree heterogeneity
    - Community strength controls the within-community edge fraction
    - Pair-weighted sampling keeps degree distribution close to ER-like Poisson
    """
    if num_nodes < 2:
        return np.empty((0, 2), dtype=np.uint32)

    max_blocks = min(max_blocks, num_nodes)
    if max_blocks < 2:
        max_blocks = 2

    num_blocks = int(rng.integers(2, max_blocks + 1))
    community_strength = rng.uniform(*community_strength_range)
    num_edges = int(round(0.5 * mean_degree * num_nodes))

    # Balanced block sizes minimize degree spread across communities.
    base_block_size = num_nodes // num_blocks
    remainder = num_nodes - base_block_size * num_blocks
    block_sizes = np.empty(num_blocks, dtype=np.int64)
    for block_idx in range(num_blocks):
        block_sizes[block_idx] = base_block_size
        if block_idx < remainder:
            block_sizes[block_idx] += 1

    # Prefix sums for contiguous node ranges per block.
    block_starts = np.empty(num_blocks + 1, dtype=np.int64)
    block_starts[0] = 0
    for block_idx in range(num_blocks):
        block_starts[block_idx + 1] = block_starts[block_idx] + block_sizes[block_idx]

    within_pair_counts = np.empty(num_blocks, dtype=np.float64)
    total_within_pair_count = 0.0
    for block_idx in range(num_blocks):
        block_size = float(block_sizes[block_idx])
        pair_count = 0.5 * block_size * (block_size - 1.0)
        within_pair_counts[block_idx] = pair_count
        total_within_pair_count += pair_count

    num_between_block_pairs = num_blocks * (num_blocks - 1) // 2
    between_pair_counts = np.empty(num_between_block_pairs, dtype=np.float64)
    between_pair_block_i = np.empty(num_between_block_pairs, dtype=np.int64)
    between_pair_block_j = np.empty(num_between_block_pairs, dtype=np.int64)
    between_pair_idx = 0
    total_between_pair_count = 0.0
    for block_i in range(num_blocks):
        for block_j in range(block_i + 1, num_blocks):
            pair_count = float(block_sizes[block_i]) * float(block_sizes[block_j])
            between_pair_counts[between_pair_idx] = pair_count
            between_pair_block_i[between_pair_idx] = block_i
            between_pair_block_j[between_pair_idx] = block_j
            total_between_pair_count += pair_count
            between_pair_idx += 1

    target_within_edges = int(round(community_strength * num_edges))
    max_within_edges = int(total_within_pair_count)
    if target_within_edges > max_within_edges:
        target_within_edges = max_within_edges
    if target_within_edges < 0:
        target_within_edges = 0

    target_between_edges = num_edges - target_within_edges
    max_between_edges = int(total_between_pair_count)
    if target_between_edges > max_between_edges:
        target_between_edges = max_between_edges
        target_within_edges = num_edges - target_between_edges

    if total_within_pair_count <= 0.0:
        target_within_edges = 0
        target_between_edges = num_edges
    if total_between_pair_count <= 0.0:
        target_between_edges = 0
        target_within_edges = num_edges

    cumulative_within_pair_counts = np.empty(num_blocks, dtype=np.float64)
    running_within_pair_count = 0.0
    for block_idx in range(num_blocks):
        running_within_pair_count += within_pair_counts[block_idx]
        cumulative_within_pair_counts[block_idx] = running_within_pair_count

    cumulative_between_pair_counts = np.empty(num_between_block_pairs, dtype=np.float64)
    running_between_pair_count = 0.0
    for block_pair_idx in range(num_between_block_pairs):
        running_between_pair_count += between_pair_counts[block_pair_idx]
        cumulative_between_pair_counts[block_pair_idx] = running_between_pair_count

    # Allocate edge list and global hash.
    edge_list = np.empty((num_edges, 2), dtype=np.uint32)
    table_size = _next_pow2(max(8, int(num_edges * 3)))
    keys = np.full(table_size, -1, dtype=np.int64)

    pos = 0
    within_edges_added = 0
    between_edges_added = 0
    trial_count = 0
    max_trials = max(10_000, num_edges * 100)

    # Sample exactly num_edges, while preserving the target within/between split.
    while pos < num_edges:
        trial_count += 1
        if trial_count > max_trials:
            break

        remaining_edges = num_edges - pos
        remaining_within_edges = target_within_edges - within_edges_added
        remaining_between_edges = target_between_edges - between_edges_added

        draw_within = False
        if remaining_within_edges > 0 and remaining_between_edges <= 0:
            draw_within = True
        elif remaining_between_edges > 0 and remaining_within_edges <= 0:
            draw_within = False
        elif remaining_within_edges > 0 and remaining_between_edges > 0:
            draw_within = rng.random() < (
                float(remaining_within_edges) / float(remaining_edges)
            )

        if draw_within:
            if total_within_pair_count <= 0.0:
                continue
            block_idx = _sample_from_cumulative_weights(
                cumulative_within_pair_counts, total_within_pair_count, rng
            )
            start = int(block_starts[block_idx])
            size = int(block_sizes[block_idx])
            if size < 2:
                continue
            node1_local = int(rng.integers(0, size))
            node2_local = int(rng.integers(0, size - 1))
            if node2_local >= node1_local:
                node2_local += 1
            node1 = start + node1_local
            node2 = start + node2_local
        else:
            if total_between_pair_count <= 0.0:
                continue
            sampled_pair_idx = _sample_from_cumulative_weights(
                cumulative_between_pair_counts, total_between_pair_count, rng
            )
            block1 = int(between_pair_block_i[sampled_pair_idx])
            block2 = int(between_pair_block_j[sampled_pair_idx])
            start1 = int(block_starts[block1])
            size1 = int(block_sizes[block1])
            start2 = int(block_starts[block2])
            size2 = int(block_sizes[block2])
            node1 = start1 + int(rng.integers(0, size1))
            node2 = start2 + int(rng.integers(0, size2))

        node1, node2 = canonical_edge(node1, node2)
        key = _edge_key(num_nodes, node1, node2)
        if _hash_insert(keys, key):
            edge_list[pos, 0] = np.uint32(node1)
            edge_list[pos, 1] = np.uint32(node2)
            pos += 1
            if draw_within:
                within_edges_added += 1
            else:
                between_edges_added += 1

    # Dense corner cases can stall rejection sampling; complete with uniform fallback.
    while pos < num_edges:
        node1 = int(rng.integers(0, num_nodes))
        node2 = int(rng.integers(0, num_nodes - 1))
        if node2 >= node1:
            node2 += 1
        node1, node2 = canonical_edge(node1, node2)
        key = _edge_key(num_nodes, node1, node2)
        if _hash_insert(keys, key):
            edge_list[pos, 0] = np.uint32(node1)
            edge_list[pos, 1] = np.uint32(node2)
            pos += 1

    return edge_list[:pos]


# ---------- Random geometric graph -------
@njit(fastmath=True)
def _rgg_num_edges(coords: npt.NDArray[np.float32], radius: float) -> int:
    num_nodes = coords.shape[0]

    inv_radius = 1.0 / radius
    grid_size = int(inv_radius) + 2  # >= 2
    num_cells = grid_size * grid_size

    cell_heads = np.full(num_cells, -1, dtype=np.int32)
    next_node_in_cell = np.full(num_nodes, -1, dtype=np.int32)

    # Insert points into cells
    for node_idx in range(num_nodes):
        cell_x = int(coords[node_idx, 0] * inv_radius)
        if cell_x < 0:
            cell_x = 0
        elif cell_x >= grid_size:
            cell_x = grid_size - 1
        cell_y = int(coords[node_idx, 1] * inv_radius)
        if cell_y < 0:
            cell_y = 0
        elif cell_y >= grid_size:
            cell_y = grid_size - 1
        cell_id = cell_x + grid_size * cell_y
        next_node_in_cell[node_idx] = cell_heads[cell_id]
        cell_heads[cell_id] = node_idx

    radius_square = radius * radius
    num_edges = 0
    for node_idx in range(num_nodes):
        cell_x = int(coords[node_idx, 0] * inv_radius)
        if cell_x < 0:
            cell_x = 0
        elif cell_x >= grid_size:
            cell_x = grid_size - 1
        cell_y = int(coords[node_idx, 1] * inv_radius)
        if cell_y < 0:
            cell_y = 0
        elif cell_y >= grid_size:
            cell_y = grid_size - 1
        for dx in (-1, 0, 1):
            neighbor_cell_x = cell_x + dx
            if neighbor_cell_x < 0 or neighbor_cell_x >= grid_size:
                continue
            for dy in (-1, 0, 1):
                neighbor_cell_y = cell_y + dy
                if neighbor_cell_y < 0 or neighbor_cell_y >= grid_size:
                    continue
                cell_id = neighbor_cell_x + grid_size * neighbor_cell_y
                neighbor_node_idx = cell_heads[cell_id]
                while neighbor_node_idx != -1:
                    if neighbor_node_idx > node_idx:
                        delta_x = coords[node_idx, 0] - coords[neighbor_node_idx, 0]
                        delta_y = coords[node_idx, 1] - coords[neighbor_node_idx, 1]
                        if delta_x * delta_x + delta_y * delta_y <= radius_square:
                            num_edges += 1
                    neighbor_node_idx = next_node_in_cell[neighbor_node_idx]
    return num_edges


@njit(fastmath=True)
def _get_rgg_edge_list(
    coords: npt.NDArray[np.float32], radius: float
) -> npt.NDArray[np.uint32]:
    num_nodes = coords.shape[0]
    num_edges = _rgg_num_edges(coords, radius)
    edge_list = np.empty((num_edges, 2), dtype=np.uint32)

    inv_radius = 1.0 / radius
    grid_size = int(inv_radius) + 2
    num_cells = grid_size * grid_size

    cell_heads = np.full(num_cells, -1, dtype=np.int32)
    next_node_in_cell = np.full(num_nodes, -1, dtype=np.int32)

    for node_idx in range(num_nodes):
        cell_x = int(coords[node_idx, 0] * inv_radius)
        if cell_x < 0:
            cell_x = 0
        elif cell_x >= grid_size:
            cell_x = grid_size - 1
        cell_y = int(coords[node_idx, 1] * inv_radius)
        if cell_y < 0:
            cell_y = 0
        elif cell_y >= grid_size:
            cell_y = grid_size - 1
        cell_id = cell_x + grid_size * cell_y
        next_node_in_cell[node_idx] = cell_heads[cell_id]
        cell_heads[cell_id] = node_idx

    radius_square = radius * radius
    pos = 0
    for node_idx in range(num_nodes):
        cell_x = int(coords[node_idx, 0] * inv_radius)
        if cell_x < 0:
            cell_x = 0
        elif cell_x >= grid_size:
            cell_x = grid_size - 1
        cell_y = int(coords[node_idx, 1] * inv_radius)
        if cell_y < 0:
            cell_y = 0
        elif cell_y >= grid_size:
            cell_y = grid_size - 1

        for dx in (-1, 0, 1):
            neighbor_cell_x = cell_x + dx
            if neighbor_cell_x < 0 or neighbor_cell_x >= grid_size:
                continue
            for dy in (-1, 0, 1):
                neighbor_cell_y = cell_y + dy
                if neighbor_cell_y < 0 or neighbor_cell_y >= grid_size:
                    continue
                cell_id = neighbor_cell_x + grid_size * neighbor_cell_y
                neighbor_node_idx = cell_heads[cell_id]
                while neighbor_node_idx != -1:
                    if neighbor_node_idx > node_idx:
                        delta_x = coords[node_idx, 0] - coords[neighbor_node_idx, 0]
                        delta_y = coords[node_idx, 1] - coords[neighbor_node_idx, 1]
                        if delta_x * delta_x + delta_y * delta_y <= radius_square:
                            edge_list[pos, 0] = np.uint32(node_idx)
                            edge_list[pos, 1] = np.uint32(neighbor_node_idx)
                            pos += 1
                    neighbor_node_idx = next_node_in_cell[neighbor_node_idx]
    return edge_list


@njit(fastmath=True)
def get_rgg(
    num_nodes: int, mean_degree: float, rng: np.random.Generator
) -> npt.NDArray[np.uint32]:
    coords = rng.random((num_nodes, 2)).astype(np.float32)
    max_radius = np.sqrt(2)

    radius1 = np.sqrt(mean_degree / (np.pi * num_nodes))
    if radius1 < 0.0:
        radius1 = 0.0
    elif radius1 > max_radius:
        radius1 = max_radius
    edge_list1 = _get_rgg_edge_list(coords, radius1)
    mean_degree1 = 2.0 * len(edge_list1) / num_nodes

    if mean_degree1 <= 0:
        return edge_list1

    radius2 = radius1 * np.sqrt(mean_degree / mean_degree1)
    if radius2 < 0.0:
        radius2 = 0.0
    elif radius2 > max_radius:
        radius2 = max_radius

    edge_list2 = _get_rgg_edge_list(coords, radius2)
    mean_degree2 = 2.0 * len(edge_list2) / num_nodes

    if abs(mean_degree - mean_degree1) <= abs(mean_degree - mean_degree2):
        return edge_list1
    else:
        return edge_list2


# ---------- Configuration model -------
@njit(fastmath=True)
def _weighted_choice(
    nodes: np.ndarray,
    weights: np.ndarray,
    count: int,
    total: float,
    rng: np.random.Generator,
) -> int:
    if count <= 0 or total <= 0.0:
        return -1

    random_threshold = rng.random() * total
    cumulative_weight = 0.0
    for candidate_idx in range(count):
        cumulative_weight += weights[candidate_idx]
        if random_threshold <= cumulative_weight:
            return int(nodes[candidate_idx])
    return int(nodes[count - 1])  # floating-point safety fallback


@njit(fastmath=True)
def _mark_neighbors(
    node: int,
    first_edge_idx_by_node: np.ndarray,
    adjacent_node: np.ndarray,
    next_edge_idx: np.ndarray,
    marks: np.ndarray,
    stamp: int,
) -> None:
    edge_idx = first_edge_idx_by_node[node]
    while edge_idx != -1:
        marks[adjacent_node[edge_idx]] = stamp
        edge_idx = next_edge_idx[edge_idx]


@njit(fastmath=True)
def configuration_model(
    degrees: npt.NDArray[np.uint32], rng: np.random.Generator, max_trials: int = 100
) -> npt.NDArray[np.uint32]:
    """
    BKS-style configuration model
    """
    num_nodes = len(degrees)
    num_edges = int(degrees.sum() // 2)

    if num_edges == 0:
        return np.empty((0, 2), dtype=np.uint32)

    weights = 1.0 - np.outer(degrees, degrees) / (4.0 * num_edges)
    np.fill_diagonal(weights, 0.0)

    # Reusable buffers (O(N))
    candidate_nodes_buffer = np.empty(num_nodes, dtype=np.int32)
    candidate_weights_buffer = np.empty(num_nodes, dtype=np.float64)
    marks = np.zeros(num_nodes, dtype=np.int32)

    for _ in range(max_trials):
        residual = degrees.copy()
        is_active = residual > 0

        # Edge list output
        edge_list = np.empty((num_edges, 2), dtype=np.uint32)

        # Adjacency linked-list (undirected: 2 entries per edge)
        first_edge_idx_by_node = np.full(num_nodes, -1, dtype=np.int32)
        adjacent_node = np.empty(2 * num_edges, dtype=np.int32)
        next_edge_idx = np.empty(2 * num_edges, dtype=np.int32)
        next_free_edge_idx = 0

        node1_weight_multiplier = weights @ residual.astype(np.float64)

        ok = True
        stamp = 1

        for edge_idx in range(num_edges):
            # ---- choose node1 ----
            num_node1_candidates = 0
            total_node1_weight = 0.0
            for node_idx in range(num_nodes):
                if not is_active[node_idx]:
                    continue
                candidate_weight = (
                    float(residual[node_idx]) * node1_weight_multiplier[node_idx]
                )
                if candidate_weight > 0.0:
                    candidate_nodes_buffer[num_node1_candidates] = node_idx
                    candidate_weights_buffer[num_node1_candidates] = candidate_weight
                    total_node1_weight += candidate_weight
                    num_node1_candidates += 1

            node1 = _weighted_choice(
                candidate_nodes_buffer,
                candidate_weights_buffer,
                num_node1_candidates,
                total_node1_weight,
                rng,
            )
            if node1 < 0:
                ok = False
                break

            # ---- choose node2 (active, not self, not neighbor of node1) ----
            stamp += 1
            _mark_neighbors(
                node1,
                first_edge_idx_by_node,
                adjacent_node,
                next_edge_idx,
                marks,
                stamp,
            )

            num_node2_candidates = 0
            total_node2_weight = 0.0
            for neighbor_idx in range(num_nodes):
                if not is_active[neighbor_idx]:
                    continue
                if neighbor_idx == node1:
                    continue
                if (
                    marks[neighbor_idx] == stamp
                ):  # already neighbor => multi-edge forbidden
                    continue

                candidate_weight = (
                    float(residual[neighbor_idx]) * weights[node1, neighbor_idx]
                )
                if candidate_weight > 0.0:
                    candidate_nodes_buffer[num_node2_candidates] = neighbor_idx
                    candidate_weights_buffer[num_node2_candidates] = candidate_weight
                    total_node2_weight += candidate_weight
                    num_node2_candidates += 1

            node2 = _weighted_choice(
                candidate_nodes_buffer,
                candidate_weights_buffer,
                num_node2_candidates,
                total_node2_weight,
                rng,
            )
            if node2 < 0:
                ok = False
                break

            # ---- add edge ----
            node_low = node1
            node_high = node2
            if node_low > node_high:
                node_low, node_high = node_high, node_low

            edge_list[edge_idx, 0] = np.uint32(node_low)
            edge_list[edge_idx, 1] = np.uint32(node_high)

            adjacent_node[next_free_edge_idx] = node2
            next_edge_idx[next_free_edge_idx] = first_edge_idx_by_node[node1]
            first_edge_idx_by_node[node1] = next_free_edge_idx
            next_free_edge_idx += 1

            adjacent_node[next_free_edge_idx] = node1
            next_edge_idx[next_free_edge_idx] = first_edge_idx_by_node[node2]
            first_edge_idx_by_node[node2] = next_free_edge_idx
            next_free_edge_idx += 1

            # ---- update node1_weight_multiplier ----
            node1_stub = residual[node1]
            node2_stub = residual[node2]

            # subtract for nodes that are NOT neighbors of node1 (excluding node1)
            stamp += 1
            _mark_neighbors(
                node1,
                first_edge_idx_by_node,
                adjacent_node,
                next_edge_idx,
                marks,
                stamp,
            )
            for candidate_node_idx in range(num_nodes):
                if candidate_node_idx == node1:
                    continue
                if marks[candidate_node_idx] != stamp:
                    node1_weight_multiplier[candidate_node_idx] -= weights[
                        candidate_node_idx, node1
                    ]

            # subtract for nodes that are NOT neighbors of node2 (excluding node2)
            stamp += 1
            _mark_neighbors(
                node2,
                first_edge_idx_by_node,
                adjacent_node,
                next_edge_idx,
                marks,
                stamp,
            )
            for candidate_node_idx in range(num_nodes):
                if candidate_node_idx == node2:
                    continue
                if marks[candidate_node_idx] != stamp:
                    node1_weight_multiplier[candidate_node_idx] -= weights[
                        candidate_node_idx, node2
                    ]

            node1_weight_multiplier[node1] -= (node2_stub - 1) * weights[node1, node2]
            node1_weight_multiplier[node2] -= (node1_stub - 1) * weights[node1, node2]

            # clamp non-negative
            for candidate_node_idx in range(num_nodes):
                if node1_weight_multiplier[candidate_node_idx] < 0.0:
                    node1_weight_multiplier[candidate_node_idx] = 0.0

            # residual / active update
            residual[node1] -= 1
            residual[node2] -= 1
            if residual[node1] == 0:
                is_active[node1] = False
            if residual[node2] == 0:
                is_active[node2] = False

        if ok and np.all(residual == 0):
            return edge_list

    return np.empty((0, 2), dtype=np.uint32)


GENERATORS: dict[
    str, Callable[[int, float, np.random.Generator], npt.NDArray[np.uint32]]
] = {
    "WS": get_ws,
    "ER": get_er,
    "SBM": get_sbm,
    "RGG": get_rgg,
    "BA": get_ba,
    "CL": get_cl,
    "HK": get_hk,
}
