import argparse
import sys
import time
import typing
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ERGM.data import ERGMData, save
from ERGM.joint_degree_flux import get_joint_degree_flux
from ERGM.type_aliases import RESET_METHOD, STORE
from graph.assortativity import (
    get_assortativity,
    get_degree_variance,
    get_joint_degree_matrix,
    get_marginal_prob,
)
from graph.edge_list import (
    canonical_edge,
    edge_list_to_csr,
    edge_list_to_degrees,
    edge_list_to_graph,
)
from graph.generator import GENERATORS, configuration_model
from graph.properties import get_average_clustering_coefficient
from graph.type_aliases import EDGE, NODE
from path import DATA_DIR


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Initial graph configuration
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--num_edges", type=int, default=300)
    parser.add_argument("--graph_type", default="ER", choices=list(GENERATORS.keys()))
    parser.add_argument("--graph_seed", type=int, default=0)
    parser.add_argument(
        "--reset_method",
        default="soft",
        choices=typing.get_args(RESET_METHOD),
        help="Reset strategy for initial graph. Hard: create new initial graph for every sampling. Soft: keep degree sequence (use configuration model). None: reuse initial graph for every sampling.",
    )

    # Sampling configuration
    parser.add_argument("--seeds", type=int, nargs=2, default=(0, 1000))
    parser.add_argument("--max_steps", type=int, default=200_000)
    parser.add_argument("--assortativity_weight", type=float, default=0.0)

    # Logging configuration
    parser.add_argument(
        "--store", nargs="+", default=["metadata"], choices=typing.get_args(STORE)
    )
    parser.add_argument("--log_interval", type=int, default=5000)

    return parser.parse_args()


def rewire(
    old_edges: tuple[EDGE, EDGE],
    new_edges: tuple[EDGE, EDGE],
    edge_indices: tuple[int, int],
    edge_list: npt.NDArray[np.uint32],
    adjacency_list: list[set[NODE]],
    degrees: npt.NDArray[np.uint32],
    joint_degree_matrix: npt.NDArray[np.uint32] | None = None,
) -> None:
    """
    Rewire the edges and change graph properties accordingly.

    Args:
        old_edges: old_edge1, old_edge2. will be replaced by new_edges
        new_edges: new_edge1, new_edge2. will replace old_edges
        edge_indices: index of old_edge1, old_edge2 in edge_list
        edge_list: [E, 2] edge list, with sorted nodes (node1, node2) while node1 < node2
        adjacency_list: [N, ] adjacency list, where adjacency_list[node] is set of neighbors of node
        degrees: [N, ] degrees of each node
        joint_degree_matrix: [D, D]
    """
    old_edge1, old_edge2 = old_edges
    new_edge1, new_edge2 = new_edges
    edge1_idx, edge2_idx = edge_indices

    (node1, node2), (node3, node4) = old_edge1, old_edge2
    (new_node1, new_node2), (new_node3, new_node4) = new_edge1, new_edge2

    degree1, degree2 = degrees[node1], degrees[node2]
    degree3, degree4 = degrees[node3], degrees[node4]
    new_degree1, new_degree2 = degrees[new_node1], degrees[new_node2]
    new_degree3, new_degree4 = degrees[new_node3], degrees[new_node4]

    # ---------------- Replace edge list  ----------------
    edge_list[edge1_idx] = np.array(new_edge1, dtype=np.uint32)
    edge_list[edge2_idx] = np.array(new_edge2, dtype=np.uint32)

    # ---------------- Adjacency list ----------------
    adjacency_list[node1].remove(node2)
    adjacency_list[node2].remove(node1)
    adjacency_list[node3].remove(node4)
    adjacency_list[node4].remove(node3)
    adjacency_list[new_node1].add(new_node2)
    adjacency_list[new_node2].add(new_node1)
    adjacency_list[new_node3].add(new_node4)
    adjacency_list[new_node4].add(new_node3)

    # ---------------- Joint degree matrix ----------------
    if joint_degree_matrix is not None:
        joint_degree_matrix[degree1 - 1, degree2 - 1] -= 1
        joint_degree_matrix[degree2 - 1, degree1 - 1] -= 1
        joint_degree_matrix[degree3 - 1, degree4 - 1] -= 1
        joint_degree_matrix[degree4 - 1, degree3 - 1] -= 1
        joint_degree_matrix[new_degree1 - 1, new_degree2 - 1] += 1
        joint_degree_matrix[new_degree2 - 1, new_degree1 - 1] += 1
        joint_degree_matrix[new_degree3 - 1, new_degree4 - 1] += 1
        joint_degree_matrix[new_degree4 - 1, new_degree3 - 1] += 1


class ERGMSampler:
    """
    Exponential Random Graph Model (ERGM) sampler using Metropolis-Hastings algorithm

    Starting from ER random graph, rewire edges to sample graphs following exp(-J * assortativity(G)) distribution
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        graph_type: str,
        graph_seed: int,
        reset_method: RESET_METHOD,
    ) -> None:
        """
        Args:
            num_nodes: Number of nodes
            num_edges: Number of edges
            graph_type: Topological type of initial graph
            graph_rng: Random number generator for initial graphs
        """
        self.num_nodes = num_nodes
        self.graph_type = graph_type
        self.graph_seed = graph_seed
        self.reset_method: RESET_METHOD = reset_method

        self.graph_rng = np.random.default_rng(graph_seed)
        self.mean_degree = num_edges / num_nodes * 2.0

        # Initialize graph
        self.init_edge_list = GENERATORS[graph_type](
            num_nodes, self.mean_degree, self.graph_rng
        )
        self.num_edges = len(self.init_edge_list)
        self.init_degrees = edge_list_to_degrees(num_nodes, self.init_edge_list)
        self.init_graph = edge_list_to_graph(num_nodes, self.init_edge_list)

    def reset(self) -> None:
        if self.reset_method == "none":
            self.edge_list = self.init_edge_list.copy()
            self.degrees = self.init_degrees.copy()
            graph = self.init_graph.copy()

        elif self.reset_method == "soft":
            self.edge_list = configuration_model(self.init_degrees, self.graph_rng)
            self.degrees = self.init_degrees.copy()
            graph = edge_list_to_graph(self.num_nodes, self.edge_list)

        else:
            self.edge_list = GENERATORS[self.graph_type](
                self.num_nodes, self.mean_degree, self.graph_rng
            )
            self.degrees = edge_list_to_degrees(self.num_nodes, self.edge_list)
            graph = edge_list_to_graph(self.num_nodes, self.edge_list)

        # Current information
        self.adjacency = [set(graph.neighbors(node)) for node in range(self.num_nodes)]
        self.joint_degree_matrix = get_joint_degree_matrix(self.edge_list, self.degrees)
        self.marginal_prob = get_marginal_prob(self.joint_degree_matrix, self.num_edges)
        self.degree_variance = get_degree_variance(
            self.degrees.max(), self.marginal_prob
        )
        self.assortativity = get_assortativity(
            self.joint_degree_matrix, self.num_edges, self.marginal_prob, self.degree_variance
        )

    def sample_rewire(
        self, rng: np.random.Generator
    ) -> tuple[tuple[EDGE, EDGE], tuple[EDGE, EDGE], tuple[int, int]]:
        """
        Sample a edge rewiring
        Returns:
            (old_edge1, old_edge2): old edges (node1, node2), (node3, node4)
            (new_edge1, new_edge2): new edges (node1, node3), (node2, node4)
            (edge1_idx, edge2_idx): indices of old and new edges
        """

        while True:
            # Randomly select two edges and mode to rewire
            edge1_idx, edge2_idx = rng.choice(self.num_edges, size=2, replace=False)
            edge1: EDGE = tuple(self.edge_list[edge1_idx].tolist())
            edge2: EDGE = tuple(self.edge_list[edge2_idx].tolist())

            # Check if (node3, node4) contains node1 or node2
            # which leads to self-loop or pointless rewire
            if set(edge1) & set(edge2):
                continue

            (node1, node2), (node3, node4) = edge1, edge2

            # If True, the mode is forbidden
            mask_mode0, mask_mode1 = False, False

            if (node3 in self.adjacency[node1]) or (node4 in self.adjacency[node2]):
                # mode0 leads to multi-edge
                mask_mode0 = True
            if (node4 in self.adjacency[node1]) or (node3 in self.adjacency[node2]):
                # mode1 leads to multi-edge
                mask_mode1 = True

            # Both modes are forbidden: try new set of edges
            if mask_mode0 and mask_mode1:
                continue

            # Choose mode based on masks
            if mask_mode0:
                mode = 1
            elif mask_mode1:
                mode = 0
            else:
                mode = rng.choice(2)

            # New edge pairs
            if mode == 0:
                new_edge1 = canonical_edge(node1, node3)
                new_edge2 = canonical_edge(node2, node4)
            else:
                new_edge1 = canonical_edge(node1, node4)
                new_edge2 = canonical_edge(node2, node3)

            return (edge1, edge2), (new_edge1, new_edge2), (edge1_idx, edge2_idx)

    def __call__(
        self,
        seed: int,
        max_steps: int,
        assortativity_weight: float,
        log_interval: int | None,
        store: list[STORE],
    ) -> ERGMData:
        """
        Args:
            rng: RNG for sampling rewiring and acceptance probability
            max_steps: Maximum number of steps to sample
            assortativity_weight: Weight for assortativity
            log_interval: Interval to log progress
            store: Which data to store
        Return:
            edge_list: sampled graph
            assortativity_history: [T+1, ] history of assortativity including initial value
            accepted_history: [T, ] history of acceptance
            joint_degree_matrices: [T+1, K, K] or empty, history of joint degree matrix
            joint_degree_fluxes: [T, K, K] or empty, history of joint degree flux
        """
        rng = np.random.default_rng(seed)

        start = time.perf_counter()

        # Set log interval to max steps if not provided (not logging)
        if log_interval is None:
            log_interval = max_steps

        # History
        assortativity_history = np.zeros(max_steps + 1, dtype=np.float32)
        assortativity_history[0] = self.assortativity
        accepted_history = np.zeros(max_steps, dtype=np.bool_)

        if "joint_degree" in store:
            max_degree = self.degrees.max()
            joint_degree_matrices = np.zeros(
                (max_steps + 1, max_degree, max_degree), dtype=np.uint32
            )
            joint_degree_fluxes = np.zeros(
                (max_steps, max_degree, max_degree), dtype=np.float32
            )
        else:
            joint_degree_matrices = np.array([], dtype=np.uint32)
            joint_degree_fluxes = np.array([], dtype=np.float32)

        # Start Markov Chain
        num_accepted = 0
        for step in range(max_steps):

            # Joint degree flux computation
            if "joint_degree" in store:
                joint_degree_matrices[step] = self.joint_degree_matrix.copy()

                # Compute joint degree flux
                adjacency_matrix = np.zeros(
                    (self.num_nodes, self.num_nodes), dtype=np.bool_
                )
                for i, neighbors in enumerate(self.adjacency):
                    for n in neighbors:
                        adjacency_matrix[i, n] = True
                joint_degree_fluxes[step] = get_joint_degree_flux(
                    self.num_edges,
                    self.edge_list,
                    self.degrees,
                    adjacency_matrix,
                    assortativity_weight,
                )

            # Sample random rewiring
            old_edges, new_edges, edge_indices = self.sample_rewire(rng)
            old_node1, old_node2 = old_edges[0]
            old_node3, old_node4 = old_edges[1]
            new_node1, new_node2 = new_edges[0]
            new_node3, new_node4 = new_edges[1]

            old_degree1 = int(self.degrees[old_node1])
            old_degree2 = int(self.degrees[old_node2])
            old_degree3 = int(self.degrees[old_node3])
            old_degree4 = int(self.degrees[old_node4])
            new_degree1 = int(self.degrees[new_node1])
            new_degree2 = int(self.degrees[new_node2])
            new_degree3 = int(self.degrees[new_node3])
            new_degree4 = int(self.degrees[new_node4])

            # New graph after rewiring
            rewire(
                old_edges,
                new_edges,
                edge_indices,
                self.edge_list,
                self.adjacency,
                self.degrees,
                self.joint_degree_matrix,
            )

            # Accept the proposal with Boltzmann probability
            delta_s = 2.0 * (
                (old_degree1 * old_degree2 + old_degree3 * old_degree4)
                - (new_degree1 * new_degree2 + new_degree3 * new_degree4)
            )
            delta_energy = assortativity_weight * delta_s
            acceptance_prob = min(
                1.0, np.exp(-np.float128(delta_energy))
            )  # reduce overflow
            accepted = rng.random() < acceptance_prob

            if accepted:
                self.assortativity = get_assortativity(
                    self.joint_degree_matrix,
                    self.num_edges,
                    self.marginal_prob,
                    self.degree_variance,
                )
                num_accepted += 1
            else:
                # Undo the rewiring: rewire new_edges -> old_edges
                rewire(
                    new_edges,
                    old_edges,
                    edge_indices,
                    self.edge_list,
                    self.adjacency,
                    self.degrees,
                    self.joint_degree_matrix,
                )

            # Store history
            assortativity_history[step + 1] = self.assortativity
            accepted_history[step] = accepted

            # Logging
            if (step + 1) % log_interval == 0:
                print(
                    (
                        f"Step {step+ 1}: assortativity={self.assortativity:.4f}"
                        f", acceptance_rate={(num_accepted / (step + 1)):.4f}"
                    ),
                    flush=True,
                )

        # Final (max_steps + 1) joint degree matrix
        if "joint_degree" in store:
            joint_degree_matrices[-1] = self.joint_degree_matrix.copy()

        end = time.perf_counter()

        # Get clustering coefficient of the sampled graph
        degree = edge_list_to_degrees(self.num_nodes, self.edge_list)
        offset, neighbor = edge_list_to_csr(self.num_nodes, self.edge_list, degree)
        clustering_coefficient = get_average_clustering_coefficient(
            degree, offset, neighbor
        )

        return ERGMData(
            graph_type=self.graph_type,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            graph_seed=self.graph_seed,
            reset_method=self.reset_method,
            seed=seed,
            max_steps=max_steps,
            assortativity_weight=assortativity_weight,
            edge_list=self.edge_list,
            assortativity_history=assortativity_history,
            accepted_history=accepted_history,
            joint_degree_matrix_history=joint_degree_matrices,
            joint_degree_flux_history=joint_degree_fluxes,
            clustering_coefficient=clustering_coefficient,
            runtime=end - start,
        )


def main():
    args = get_args()

    if "joint_degree" in args.store:
        print(
            "WARNING: computing flux. This will significantly slow down the sampling.",
            flush=True,
        )

    sampler = ERGMSampler(
        args.num_nodes,
        args.num_edges,
        args.graph_type,
        args.graph_seed,
        args.reset_method,
    )

    data_list: list[ERGMData] = []
    for seed in range(args.seeds[0], args.seeds[1]):
        print(f"{seed=} started", flush=True)

        # Reset the sampler for new sampling
        sampler.reset()

        # Sampling
        data = sampler(
            seed,
            args.max_steps,
            args.assortativity_weight,
            args.log_interval,
            args.store,
        )

        data_list.append(data)

    # Store data
    data_dir = DATA_DIR / "ERGM"
    data_dir.mkdir(exist_ok=True)
    file_name = f"{args.graph_type}_N{args.num_nodes}_w{args.assortativity_weight}_{datetime.now().strftime('%m%d_%H%M%S_%f')}.h5"

    save(data_dir / file_name, data_list, store=args.store)


if __name__ == "__main__":
    main()
