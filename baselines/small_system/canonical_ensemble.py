"""Compute canonical ensembles from an exact small-graph dataset."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.small_system.common import K_to_rho, SmallData
from graph.edge_list import edge_list_to_degrees, get_mean_adjacency_matrix
from graph.entropy import get_dyad_independent_entropy
from graph.generator import get_er


class CanonicalEnsemble(TypedDict):
    """Exact canonical ensemble statistics for one target assortativity."""

    assortativity: float  # rho
    conjugate_parameter: float  # lambda_K
    K_distribution: npt.NDArray[np.float64]  # P(K)
    log_partition: float  # log Z
    shannon_entropy: float  # S
    dyad_independent_entropy: float  # S_DI


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, required=True, help="number of nodes")
    parser.add_argument("--mean_degree", type=float, required=True)
    parser.add_argument("--graph_seed", type=int, required=True)
    return parser.parse_args()


def get_canonical_statistics(
    lambda_K: float,
    K_values: npt.NDArray[np.int64],
    num_graphs: npt.NDArray[np.uint32],
    only_K: bool = False,
) -> tuple[float, npt.NDArray[np.float64], float, float]:
    """
    Return the statistics of canonical ensemble defined by lambda_K.

    Returns
        expected_K: <K>
        K_distribution: P(K)
        log_partition: log Z; zero when only_K is True
        shannon_entropy: S; zero when only_K is True
    """
    K_min = K_values.min()

    # K' = K - K_min to prevent overflow without changing the probabilities
    shifted_K = K_values - K_min

    # log w(K) = log Omega(K) + lambda_K K'
    log_weights = np.log(num_graphs) + lambda_K * shifted_K

    # log Z' = log sum_K exp[log w(K)]
    shifted_log_partition = cast(float, logsumexp(log_weights))

    # P(K) = P(K') = Omega(K) exp(lambda_K K') / Z' = w(K) / Z' = exp[log w(K) - log Z']
    K_distribution = cast(
        npt.NDArray[np.float64], np.exp(log_weights - shifted_log_partition)
    )

    # <K'> = sum_K P(K) K'
    expected_shifted_K = K_distribution @ shifted_K

    # <K> = K_min + sum_K P(K) K'
    expected_K = float(K_min + expected_shifted_K)

    if only_K:
        return expected_K, K_distribution, 0.0, 0.0

    # log Z = log Z' + lambda_K K_min
    log_partition = float(shifted_log_partition + lambda_K * K_min)

    # S = log Z - lambda_K <K> = log Z' - lambda_K <K'>
    shannon_entropy = float(log_partition - lambda_K * expected_K)

    return expected_K, K_distribution, log_partition, shannon_entropy


def get_canonical_mean_adjacency_matrix(
    K_distribution: npt.NDArray[np.float64],
    mean_adjacency_matrices: dict[int, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    r"""
    Return the mean adjacency matrix of canonical ensemble defined by lambda_K

    Args
        K_distribution: [num_K, ] P(K)
        mean_adjacency_matrices: {K_value: mean_adjacency_matrix}
                                 \bar{A}(K) = (1 / Omega(K)) sum_{G : K(G) = K} A(G)

    Return:
        mean_adjacency_matrix: [num_nodes, num_nodes]

        <A> = sum_K P(K) \bar{A}(K)
    """
    adj = np.stack(
        list(mean_adjacency_matrices.values())
    )  # [num_K, num_nodes, num_nodes]
    return np.sum(K_distribution[:, None, None] * adj, axis=0, dtype=np.float64)


def compute_conjugate_parameter(
    target_K: int, K_values: npt.NDArray[np.int64], num_graphs: npt.NDArray[np.uint32]
) -> float:
    """
    Get conjugate parameter lambda satisfying <K> = target_K
    """

    # f'(lambda_K) = Var(K) >= 0 : f is monotonically increasing
    def f(lambda_K: float) -> float:
        expected_K, *_ = get_canonical_statistics(lambda_K, K_values, num_graphs, only_K=True)
        return expected_K - target_K

    # Find range of lambda_K s.t. f(lower_lambda_k) <= 0 <= f(upper_lambda_k)
    lower_lambda_K, upper_lambda_K = -1.0, 1.0
    while f(lower_lambda_K) > 0:
        lower_lambda_K *= 2.0
    while f(upper_lambda_K) < 0:
        upper_lambda_K *= 2.0

    # Find a unique lambda_K satisfying <K> = target_K
    return cast(float, brentq(f, lower_lambda_K, upper_lambda_K))


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
    print(f"ER graph with N={num_nodes}, k={mean_degree}, seed={graph_seed}")
    print(f"Degree sequence: {degrees}")

    # Load all graphs that can be generated by this degree sequence
    with open(small_data.file_path("graphs"), "rb") as f:
        edge_lists_per_K: dict[int, list[npt.NDArray[np.uint32]]] = pickle.load(f)
    K_values = np.array([K_value for K_value in edge_lists_per_K], dtype=np.int64)
    num_graphs = np.array(
        [len(edge_lists) for edge_lists in edge_lists_per_K.values()], dtype=np.uint32
    )
    mean_adjacency_matrices = {
        K_value: get_mean_adjacency_matrix(num_nodes, np.stack(edge_lists))
        for K_value, edge_lists in edge_lists_per_K.items()
    }

    canonical_ensembles: dict[int, CanonicalEnsemble] = {}
    for K_value, edge_lists in edge_lists_per_K.items():
        # Target assortativity as microcanonical assortativity
        target_rho = K_to_rho(K_value, degrees)

        if K_value == K_values[0]:
            # Target is minimum feasible rho
            lambda_K = -np.inf
            K_distribution = np.zeros(len(K_values), dtype=np.float64)
            K_distribution[0] = 1.0
            log_partition = -np.inf
            shannon_entropy = np.log(len(edge_lists))
            dyad_independent_entropy = get_dyad_independent_entropy(
                mean_adjacency_matrices[K_value]
            )

        elif K_value == K_values[-1]:
            # Target is maximum feasible rho
            lambda_K = np.inf
            K_distribution = np.zeros(len(K_values), dtype=np.float64)
            K_distribution[-1] = 1.0
            log_partition = np.inf
            shannon_entropy = np.log(len(edge_lists))
            dyad_independent_entropy = get_dyad_independent_entropy(
                mean_adjacency_matrices[K_value]
            )

        else:
            lambda_K = compute_conjugate_parameter(K_value, K_values, num_graphs)
            _, K_distribution, log_partition, shannon_entropy = (
                get_canonical_statistics(lambda_K, K_values, num_graphs)
            )
            mean_adjacency_matrix = get_canonical_mean_adjacency_matrix(
                K_distribution, mean_adjacency_matrices
            )
            dyad_independent_entropy = get_dyad_independent_entropy(
                mean_adjacency_matrix
            )

        # Report result
        print(
            f"K={K_value:,} (rho={target_rho:.4f}), "
            f"lambda_K={lambda_K:.12g}, log Z={log_partition:.12g}, "
            f"S={shannon_entropy:.12g}, S_DI={dyad_independent_entropy:.12g}"
        )

        # Store result
        canonical_ensembles[K_value] = CanonicalEnsemble(
            assortativity=target_rho,
            conjugate_parameter=lambda_K,
            K_distribution=K_distribution,
            log_partition=log_partition,
            shannon_entropy=shannon_entropy,
            dyad_independent_entropy=dyad_independent_entropy,
        )

    with open(small_data.file_path("canonical"), "wb") as f:
        pickle.dump(canonical_ensembles, f)


if __name__ == "__main__":
    main()
