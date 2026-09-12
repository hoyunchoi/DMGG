import heapq
import math

import numpy as np
import numpy.typing as npt
from numba import njit

from baselines.ResiNet.observation import ResiNetObs
from DMGG.env import GraphObs
from graph.edge_list import edge_list_to_csr, edge_list_to_degrees

ATTACK_POSITION_DIM = 8
SHORTEST_PATH_DIM = 6
NODE_FEATURE_DIM = ATTACK_POSITION_DIM + SHORTEST_PATH_DIM + 1


def adaptive_degree_removal(
    num_nodes: int, edge_list: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Return ResiNet's adaptive maximum-degree attack order and isolate flag."""
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for node1, node2 in edge_list:
        u, v = int(node1), int(node2)
        adjacency[u].append(v)
        adjacency[v].append(u)

    degrees = np.array([len(neighbors) for neighbors in adjacency], dtype=np.int64)
    active = np.ones(num_nodes, dtype=np.bool_)
    initially_isolated = degrees == 0
    became_isolated = np.zeros(num_nodes, dtype=np.bool_)
    removal_order = np.empty(num_nodes, dtype=np.int64)

    heap = [(-int(degree), node) for node, degree in enumerate(degrees)]
    heapq.heapify(heap)

    for rank in range(num_nodes):
        while True:
            negative_degree, node = heapq.heappop(heap)
            if active[node] and -negative_degree == degrees[node]:
                break

        removal_order[rank] = node
        active[node] = False

        if rank == 0:
            became_isolated[initially_isolated & active] = True

        for neighbor in adjacency[node]:
            if active[neighbor]:
                degrees[neighbor] -= 1
                heapq.heappush(heap, (-int(degrees[neighbor]), neighbor))
                if degrees[neighbor] == 0:
                    became_isolated[neighbor] = True

    return removal_order, became_isolated


def sinusoidal_attack_features(
    removal_order: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Encode each node's rank in the adaptive attack order."""
    num_nodes = len(removal_order)
    ranks = np.empty(num_nodes, dtype=np.int64)
    ranks[removal_order] = np.arange(num_nodes, dtype=np.int64)

    position = ranks.astype(np.float64)[:, None]
    div_term = np.exp(
        np.arange(0, ATTACK_POSITION_DIM, 2, dtype=np.float32)
        * (-math.log(10000.0) / ATTACK_POSITION_DIM)
    )
    features = np.empty((num_nodes, ATTACK_POSITION_DIM), dtype=np.float32)
    features[:, 0::2] = np.sin(position * div_term)
    features[:, 1::2] = np.cos(position * div_term)
    return features


@njit(cache=True)
def _four_frontier_histogram(
    offsets: npt.NDArray[np.uint32], neighbors: npt.NDArray[np.uint32]
) -> npt.NDArray[np.float32]:
    """Count distances 0--4 exactly and put all other nodes in bin 5."""
    num_nodes = len(offsets) - 1
    histogram = np.zeros((num_nodes, SHORTEST_PATH_DIM), dtype=np.float32)
    seen = np.zeros(num_nodes, dtype=np.int64)
    queue = np.empty(num_nodes, dtype=np.uint32)

    for source in range(num_nodes):
        marker = source + 1
        seen[source] = marker
        queue[0] = source
        queue_start = 0
        queue_end = 1
        reached = 1
        histogram[source, 0] = 1.0

        for distance in range(1, 5):
            level_end = queue_end
            while queue_start < level_end:
                node = queue[queue_start]
                queue_start += 1
                for pos in range(offsets[node], offsets[node + 1]):
                    neighbor = neighbors[pos]
                    if seen[neighbor] != marker:
                        seen[neighbor] = marker
                        queue[queue_end] = neighbor
                        queue_end += 1

            count = queue_end - level_end
            histogram[source, distance] = count
            reached += count

        histogram[source, 5] = num_nodes - reached

    return histogram


def four_frontier_histogram(
    num_nodes: int, edge_list: npt.NDArray[np.uint32]
) -> npt.NDArray[np.float32]:
    """Return ResiNet's six-bin shortest-path histogram for every node."""
    degrees = edge_list_to_degrees(num_nodes, edge_list)
    offsets, neighbors = edge_list_to_csr(num_nodes, edge_list, degrees)
    return _four_frontier_histogram(offsets, neighbors)


def get_first_edge_mask(
    num_nodes: int, edge_list: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Mask first edges with no structurally valid oriented second edge."""
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for node1, node2 in edge_list:
        u, v = int(node1), int(node2)
        adjacency[u].add(v)
        adjacency[v].add(u)

    mask = np.ones(len(edge_list), dtype=np.bool_)
    for edge1_idx, (node1, node2) in enumerate(edge_list):
        u, v = int(node1), int(node2)
        for node3, node4 in edge_list:
            x, y = int(node3), int(node4)
            if u == x or u == y or v == x or v == y:
                continue

            mode0_valid = x not in adjacency[u] and y not in adjacency[v]
            mode1_valid = y not in adjacency[u] and x not in adjacency[v]
            if mode0_valid or mode1_valid:
                mask[edge1_idx] = False
                break

    return mask


def _make_fire_view(
    node_features: npt.NDArray[np.float32],
    edge_list: npt.NDArray[np.int64],
    node_to_action: npt.NDArray[np.int64],
    edge_to_action: npt.NDArray[np.int64],
    glob_attr: npt.NDArray[np.float32],
) -> tuple[GraphObs, npt.NDArray[np.int64], npt.NDArray[np.int64], bool]:
    num_nodes = len(node_features)
    degrees = edge_list_to_degrees(num_nodes, edge_list.astype(np.uint32, copy=False))
    kept_nodes = np.flatnonzero(degrees > 0).astype(np.int64)

    old_to_new = np.full(num_nodes, -1, dtype=np.int64)
    old_to_new[kept_nodes] = np.arange(len(kept_nodes), dtype=np.int64)
    local_edges = old_to_new[edge_list]

    view = GraphObs(
        node_attr=node_features[kept_nodes].copy(),
        edge_list=local_edges.astype(np.int64, copy=True),
        glob_attr=glob_attr.copy(),
    )
    return (
        view,
        node_to_action[kept_nodes].copy(),
        edge_to_action.copy(),
        len(local_edges) > 0,
    )


def build_resinet_obs(action: GraphObs) -> ResiNetObs:
    """Build K=1 ResiNet filtration features from a copied action graph."""
    num_nodes = len(action.node_attr)
    edge_list = action.edge_list

    removal_order, became_isolated = adaptive_degree_removal(num_nodes, edge_list)
    attack_features = sinusoidal_attack_features(removal_order)
    raw_sp_features = four_frontier_histogram(
        num_nodes, edge_list.astype(np.uint32, copy=False)
    )
    attack_indicator = (~became_isolated).astype(np.float32)[:, None]
    raw_features = np.concatenate(
        [attack_features, raw_sp_features, attack_indicator], axis=1
    )

    raw_view = _make_fire_view(
        raw_features,
        edge_list,
        np.arange(num_nodes, dtype=np.int64),
        np.arange(len(edge_list), dtype=np.int64),
        action.glob_attr,
    )

    removed_node = removal_order[0]
    remaining_nodes = np.delete(np.arange(num_nodes, dtype=np.int64), removed_node)
    action_to_filtered = np.full(num_nodes, -1, dtype=np.int64)
    action_to_filtered[remaining_nodes] = np.arange(
        len(remaining_nodes), dtype=np.int64
    )

    kept_edge_mask = np.all(edge_list != removed_node, axis=1)
    filtered_edge_to_action = np.flatnonzero(kept_edge_mask).astype(np.int64)
    filtered_edges = action_to_filtered[edge_list[kept_edge_mask]]
    filtered_sp_features = four_frontier_histogram(
        len(remaining_nodes), filtered_edges.astype(np.uint32, copy=False)
    )
    filtered_features = np.concatenate(
        [
            attack_features[remaining_nodes],
            filtered_sp_features,
            attack_indicator[remaining_nodes],
        ],
        axis=1,
    )
    filtered_view = _make_fire_view(
        filtered_features,
        filtered_edges,
        remaining_nodes,
        filtered_edge_to_action,
        action.glob_attr,
    )

    return ResiNetObs(
        action=action,
        fire_views=(raw_view[0], filtered_view[0]),
        fire_node_to_action=(raw_view[1], filtered_view[1]),
        fire_edge_to_action=(raw_view[2], filtered_view[2]),
        view_valid=np.array([raw_view[3], filtered_view[3]], dtype=np.bool_),
        first_edge_mask=get_first_edge_mask(num_nodes, edge_list),
    )
