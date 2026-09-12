from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from collections import defaultdict
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.small_system.common import K_to_rho, SmallData
from graph.edge_list import edge_list_to_degrees
from graph.generator import get_er


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, required=True, help="number of nodes")
    parser.add_argument("--mean_degree", type=float, required=True)
    parser.add_argument("--graph_seed", type=int, required=True)
    return parser.parse_args()


@lru_cache(maxsize=200_000)
def _is_graphical(degrees: tuple[int, ...]) -> bool:
    """Check the Erdős-Gallai inequalities for a residual degree sequence."""
    sorted_degrees = np.sort(np.asarray(degrees))[::-1]

    return all(
        sum(sorted_degrees[:size])
        <= size * (size - 1)
        + sum(min(degree, size) for degree in sorted_degrees[size:])
        for size in range(1, len(sorted_degrees) + 1)
    )


def iter_graphs(
    degrees: npt.NDArray[np.uint32],
) -> Iterator[tuple[npt.NDArray[np.uint32], int]]:
    """
    For given degree sequence, yeild every graphs with their K value, where
        K = sum_(i,j in E) d_i d_j,
    which can directly converted to the assortativity by 'K_to_rho'

    Nodes are processed in the fixed order 0, 1, ..., N - 1.
    At the step for node i, all edges whose smaller endpoint is less than i have already been fixed.
    The function chooses every possible set of later neighbors needed to satisfy node i and then continues a depth-first search using an explicit stack.
    Therefore each labeled simple graph is reached exactly once.
    A branch is discarded when its residual degree sequence is not graphical.

    Yields:
        edge_list: [num_edges, 2]
        K_value
    """
    # Complement graph: graph composed of empty edges
    num_nodes, num_edges = len(degrees), degrees.sum() // 2
    complement_degrees = num_nodes - 1 - degrees
    complement_num_edges = num_nodes * (num_nodes - 1) // 2 - num_edges

    # Enumerating the complement graph is cheaper when it contains fewer edges
    if complement_num_edges < num_edges:
        use_complement = True
        residual_degrees = complement_degrees
    else:
        use_complement = False
        residual_degrees = degrees

    # sum of degree products over every possible node pair.
    # sum_{i < j} d_i d_j = [(sum_i d_i)^2 - sum_i d_i^2] / 2
    total_K = int(degrees.sum()**2 - (degrees**2).sum()) // 2

    pending_states: list[  # residual degrees, accumulated_K, edges
        tuple[npt.NDArray[np.uint32], int, tuple[tuple[int, int], ...]]
    ] = [(residual_degrees, 0, ())]

    while pending_states:
        residual_degrees, accumulated_K, edges = pending_states.pop()
        node = num_nodes - len(residual_degrees)

        # residual[j] is the number of edges still required by node + j.
        # A non-graphical residual cannot lead to a valid completion.
        if not _is_graphical(tuple(residual_degrees.tolist())):
            continue

        if node == num_nodes:
            # Graph is completed
            if use_complement:
                # Convert the complement graph to the original graph.
                graph_edges = [
                    (node1, node2)
                    for node1, node2 in itertools.combinations(range(num_nodes), 2)
                    if (node1, node2) not in set(edges)
                ]
                K_value = total_K - accumulated_K
            else:
                graph_edges = edges
                K_value = accumulated_K

            yield np.array(graph_edges, dtype=np.uint32), K_value
            continue

        # All earlier nodes are already fixed.
        # Connect this node only to later nodes, choosing every possible neighbor set of the required size.
        required, tail = int(residual_degrees[0]), residual_degrees[1:]
        available = tuple(index for index, degree in enumerate(tail) if degree)

        # The stack is LIFO, so push the combinations in reverse to preserve
        # the graph order produced by the previous recursive implementation.
        for selected in reversed(tuple(itertools.combinations(available, required))):
            next_residual = tail.copy()
            added_edges = []
            added_k = 0
            for index in selected:
                next_residual[index] -= 1
                neighbor = node + index + 1
                added_edges.append((node, neighbor))
                added_k += int(degrees[node] * degrees[neighbor])

            pending_states.append(
                (next_residual, accumulated_K + added_k, edges + tuple(added_edges))
            )


def main() -> None:
    args = get_args()
    num_nodes: int = args.num_nodes
    mean_degree: float = args.mean_degree
    graph_seed: int = args.graph_seed
    small_data = SmallData(num_nodes, mean_degree, graph_seed)

    # Generate ER graph and extract degrees
    rng = np.random.default_rng(graph_seed)
    edge_list = get_er(num_nodes, mean_degree, rng)
    degrees = edge_list_to_degrees(num_nodes, edge_list)

    # Iterate over available graphs and their K values
    edge_lists_per_K: dict[int, list[npt.NDArray[np.uint32]]] = defaultdict(list)
    for edge_list, K_value in iter_graphs(degrees):
        edge_lists_per_K[K_value].append(edge_list)
    edge_lists_per_K = dict(sorted(edge_lists_per_K.items()))
    total_num_graphs = sum(len(edge_lists) for edge_lists in edge_lists_per_K.values())

    # Print the results
    print(f"ER graph with N={num_nodes}, k={mean_degree}, seed={graph_seed}")
    print(f"Degree sequence: {degrees}")
    print(f"total available graphs={total_num_graphs:,}")
    for K_value, edge_lists in edge_lists_per_K.items():
        assortativity = K_to_rho(K_value, degrees)
        print(f"K={K_value}: rho={assortativity:.4f}, count={len(edge_lists)}")

    # Save the result
    with open(small_data.file_path("graphs"), "wb") as f:
        pickle.dump(edge_lists_per_K, f)


if __name__ == "__main__":
    main()
