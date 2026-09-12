import argparse
import sys
import time
import typing
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from path import DATA_DIR

from ERGM.data import ERGMData, save
from ERGM.joint_degree_flux import get_joint_degree_flux
from ERGM.rewire import rewire
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
from graph.type_aliases import EDGE


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


class ERGMSampler:
    """
    Metropolis-Hastings sampler for fixed-degree simple graphs.

    The target distribution is P_lambda(G) proportional to exp(lambda * K(G)),
    where K(G) = sum_{(u, v) in E(G)} d_u * d_v.
    For a fixed degree sequence, assortativity is an affine increasing function of K(G).
    Therefore, lambda > 0 favors assortative graphs, whereas lambda < 0 favors disassortative graphs.

    Each proposal is a symmetric attempted degree-preserving rewiring.
    Invalid proposals leave the graph unchanged.
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
            self.joint_degree_matrix,
            self.num_edges,
            self.marginal_prob,
            self.degree_variance,
        )

    def sample_rewire(
        self, rng: np.random.Generator
    ) -> tuple[tuple[EDGE, EDGE], tuple[EDGE, EDGE], tuple[int, int]] | None:
        """
        Sample one random edge-rewiring attempt.
        Returns:
            (old_edge1, old_edge2): old edges (node1, node2), (node3, node4)
            (new_edge1, new_edge2): new edges (node1, node3), (node2, node4)
            (edge1_idx, edge2_idx): indices of old edges
            None: the sampled action is invalid
        """

        # Randomly select two edges and a rewiring mode exactly once
        edge1_idx, edge2_idx = rng.choice(self.num_edges, size=2, replace=False)
        mode = rng.choice(2)
        edge1: EDGE = tuple(self.edge_list[edge1_idx].tolist())
        edge2: EDGE = tuple(self.edge_list[edge2_idx].tolist())

        # Shared endpoints lead to a self-loop or a pointless rewire
        if set(edge1) & set(edge2):
            return None

        (node1, node2), (node3, node4) = edge1, edge2

        if mode == 0:
            new_edge1 = canonical_edge(node1, node3)
            new_edge2 = canonical_edge(node2, node4)
        else:
            new_edge1 = canonical_edge(node1, node4)
            new_edge2 = canonical_edge(node2, node3)

        # Reject the sampled action if it would create a multi-edge
        if (new_edge1[1] in self.adjacency[new_edge1[0]]) or (
            new_edge2[1] in self.adjacency[new_edge2[0]]
        ):
            return None

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
            seed: Seed for sampling rewiring and acceptance probability
            max_steps: Maximum number of steps to sample
            assortativity_weight: Conjugate parameter lambda multiplying K(G)
            log_interval: Interval to log progress
            store: Which data to store
        Return:
            edge_list: sampled graph
            assortativity_history: [T+1, ] history of assortativity including initial value
            accepted_history: [T, ] history of acceptance
            joint_degree_matrices: [T+1, D, D] or empty, history of joint degree matrix
            joint_degree_fluxes: [T, D, D] or empty, history of joint degree flux
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
            proposal = self.sample_rewire(rng)

            if proposal is None:
                accepted = False
            else:
                old_edges, new_edges, edge_indices = proposal

                old_degree1, old_degree2 = self.degrees[list(old_edges[0])]
                old_degree3, old_degree4 = self.degrees[list(old_edges[1])]
                new_degree1, new_degree2 = self.degrees[list(new_edges[0])]
                new_degree3, new_degree4 = self.degrees[list(new_edges[1])]

                # Accept the proposal with Metropolis-Hastings probability
                # Delta K = K(new) - K(old)
                delta_K = int(
                    new_degree1 * new_degree2 + new_degree3 * new_degree4
                ) - int(old_degree1 * old_degree2 + old_degree3 * old_degree4)
                log_acceptance_ratio = assortativity_weight * delta_K
                acceptance_prob = np.exp(min(0.0, log_acceptance_ratio))
                accepted = rng.random() < acceptance_prob

                if accepted:
                    rewire(
                        edge_indices,
                        new_edges,
                        self.edge_list,
                        self.adjacency,
                        self.degrees,
                        self.joint_degree_matrix,
                    )
                    self.assortativity = get_assortativity(
                        self.joint_degree_matrix,
                        self.num_edges,
                        self.marginal_prob,
                        self.degree_variance,
                    )
                    num_accepted += 1

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
