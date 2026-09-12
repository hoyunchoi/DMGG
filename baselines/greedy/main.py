import argparse
import sys
import time
import typing
from bisect import bisect_left
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path import DATA_DIR

from ERGM.rewire import rewire
from ERGM.type_aliases import RESET_METHOD, STORE
from graph.assortativity import (
    get_assortativity,
    get_degree_variance,
    get_joint_degree_matrix,
    get_marginal_prob,
)
from graph.edge_list import canonical_edge, edge_list_to_degrees, edge_list_to_graph
from graph.generator import GENERATORS, configuration_model
from graph.type_aliases import EDGE
from greedy.data import GreedyData, save

DEGREE_CLASS = tuple[int, int]
PROPOSAL = tuple[tuple[int, int], tuple[EDGE, EDGE]]


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
        help=(
            "Reset strategy for initial graph. Hard: create a new initial graph "
            "for every run. Soft: keep the degree sequence and use the "
            "configuration model. None: reuse the initial graph."
        ),
    )

    # Greedy search configuration
    parser.add_argument("--seeds", type=int, nargs=2, default=(0, 1000))
    parser.add_argument("--max_steps", type=int, default=10_000_000)
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=0.001)

    # Logging configuration
    parser.add_argument(
        "--store", nargs="+", default=["metadata"], choices=typing.get_args(STORE)
    )
    parser.add_argument("--log_interval", type=int, default=5000)

    return parser.parse_args()


@dataclass(frozen=True, slots=True)
class RewireAction:
    edge_class1: DEGREE_CLASS
    edge_class2: DEGREE_CLASS
    mode: int
    delta_K: int


class GreedySampler:
    """
    Degree-preserving greedy rewiring in degree-class space.

    Degree-class actions are sorted by Delta K once per degree sequence.
    At every step, actions are checked in order of their distance from the current target residual.
    A valid rewiring is sampled uniformly among the tied actions, and only strict improvements are accepted.
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        graph_type: str,
        graph_seed: int,
        reset_method: RESET_METHOD,
        tolerance: float,
    ) -> None:
        self.num_nodes = num_nodes
        self.graph_type = graph_type
        self.graph_seed = graph_seed
        self.reset_method: RESET_METHOD = reset_method

        self.graph_rng = np.random.default_rng(graph_seed)
        self.mean_degree = num_edges / num_nodes * 2.0

        # tolerance of the stopping rule
        self.tolerance = tolerance

        # Initialize graph
        self.init_edge_list = GENERATORS[graph_type](
            num_nodes, self.mean_degree, self.graph_rng
        )
        self.num_edges = len(self.init_edge_list)
        self.init_degrees = edge_list_to_degrees(num_nodes, self.init_edge_list)
        self.init_graph = edge_list_to_graph(num_nodes, self.init_edge_list)

    def reset(self) -> None:
        """Reset the graph and rebuild occupied degree-class buckets."""
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

        # ------------- Current information -------------
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
        self.physical_tolerance = np.maximum(
            self.tolerance, 0.99 / (self.num_edges * self.degree_variance)
        )

        self.edges_by_class: dict[DEGREE_CLASS, set[int]] = {}
        for edge_idx, edge in enumerate(self.edge_list):
            edge_class = self._get_edge_class((int(edge[0]), int(edge[1])))
            self.edges_by_class.setdefault(edge_class, set()).add(edge_idx)

        self._initialize_sorted_class_actions()

    def _get_edge_class(self, edge: EDGE) -> DEGREE_CLASS:
        degree1 = int(self.degrees[edge[0]])
        degree2 = int(self.degrees[edge[1]])
        return min(degree1, degree2), max(degree1, degree2)

    def _orient_edge_by_degree(self, edge: EDGE) -> EDGE:
        """
        Orient an edge such that the first node has a lower or equal degree than the second node.
        """
        node1, node2 = edge
        degree1 = int(self.degrees[node1])
        degree2 = int(self.degrees[node2])
        if degree1 < degree2:
            return node1, node2
        if degree2 < degree1:
            return node2, node1
        return min(node1, node2), max(node1, node2)

    def _initialize_sorted_class_actions(self) -> None:
        """Build degree-class actions once for the current degree sequence."""
        degree_counts = np.bincount(self.degrees)
        degree_signature = tuple(int(count) for count in degree_counts)
        if getattr(self, "degree_signature", None) == degree_signature:
            return

        degree_values = [
            degree
            for degree, count in enumerate(degree_counts)
            if degree > 0 and count > 0
        ]
        edge_classes = [
            (degree1, degree2)
            for degree1_idx, degree1 in enumerate(degree_values)
            for degree2 in degree_values[degree1_idx:]
            if degree1 != degree2 or degree_counts[degree1] >= 2
        ]

        actions_by_delta_K: dict[int, list[RewireAction]] = {}
        for class1_idx, edge_class1 in enumerate(edge_classes):
            for edge_class2 in edge_classes[class1_idx:]:
                degree1, degree2 = edge_class1
                degree3, degree4 = edge_class2
                old_K = degree1 * degree2 + degree3 * degree4
                delta_Ks = (
                    degree1 * degree3 + degree2 * degree4 - old_K,
                    degree1 * degree4 + degree2 * degree3 - old_K,
                )

                for mode, delta_K in enumerate(delta_Ks):
                    if delta_K == 0:
                        continue
                    action = RewireAction(
                        edge_class1=edge_class1,
                        edge_class2=edge_class2,
                        mode=mode,
                        delta_K=delta_K,
                    )
                    actions_by_delta_K.setdefault(delta_K, []).append(action)

        self.degree_signature = degree_signature
        self.actions_by_delta_K = actions_by_delta_K
        self.sorted_delta_K = sorted(actions_by_delta_K)

    def _iter_improving_delta_K(
        self, target: float
    ) -> Generator[list[int], None, None]:
        """Yield equal-loss Delta K groups that strictly reduce the target gap."""
        residual_K = float(
            (target - self.assortativity) * self.num_edges * self.degree_variance
        )
        old_loss = abs(residual_K)
        right = bisect_left(self.sorted_delta_K, residual_K)
        left = right - 1

        while left >= 0 or right < len(self.sorted_delta_K):
            left_loss = (
                abs(residual_K - self.sorted_delta_K[left]) if left >= 0 else np.inf
            )
            right_loss = (
                abs(residual_K - self.sorted_delta_K[right])
                if right < len(self.sorted_delta_K)
                else np.inf
            )
            loss = min(left_loss, right_loss)
            delta_Ks: list[int] = []

            if left_loss == loss:
                delta_Ks.append(self.sorted_delta_K[left])
                left -= 1
            if right_loss == loss:
                delta_Ks.append(self.sorted_delta_K[right])
                right += 1

            if loss >= old_loss:
                return
            yield delta_Ks

    def _iter_edge_indices(
        self, edge_class1: DEGREE_CLASS, edge_class2: DEGREE_CLASS
    ) -> Generator[tuple[int, int], None, None]:
        indices1 = self.edges_by_class.get(edge_class1)
        indices2 = self.edges_by_class.get(edge_class2)
        if indices1 is None or indices2 is None:
            return

        if edge_class1 == edge_class2:
            yield from combinations(indices1, 2)
        else:
            for edge1_idx in indices1:
                for edge2_idx in indices2:
                    yield edge1_idx, edge2_idx

    def _iter_valid_proposals(
        self, action: RewireAction
    ) -> Generator[PROPOSAL, None, None]:
        for edge1_idx, edge2_idx in self._iter_edge_indices(
            action.edge_class1, action.edge_class2
        ):
            edge1: EDGE = tuple(self.edge_list[edge1_idx].tolist())
            edge2: EDGE = tuple(self.edge_list[edge2_idx].tolist())
            if set(edge1) & set(edge2):
                continue

            node1, node2 = self._orient_edge_by_degree(edge1)
            node3, node4 = self._orient_edge_by_degree(edge2)
            if action.mode == 0:
                new_edge1 = canonical_edge(node1, node3)
                new_edge2 = canonical_edge(node2, node4)
            else:
                new_edge1 = canonical_edge(node1, node4)
                new_edge2 = canonical_edge(node2, node3)

            if (new_edge1[1] in self.adjacency[new_edge1[0]]) or (
                new_edge2[1] in self.adjacency[new_edge2[0]]
            ):
                continue

            old_K = int(
                self.degrees[edge1[0]] * self.degrees[edge1[1]]
                + self.degrees[edge2[0]] * self.degrees[edge2[1]]
            )
            new_K = int(
                self.degrees[new_edge1[0]] * self.degrees[new_edge1[1]]
                + self.degrees[new_edge2[0]] * self.degrees[new_edge2[1]]
            )
            delta_K = new_K - old_K
            if delta_K != action.delta_K:
                raise RuntimeError("Degree-class delta K does not match actual edges")

            yield (edge1_idx, edge2_idx), (new_edge1, new_edge2)

    def _iter_tied_proposals(
        self, delta_Ks: list[int]
    ) -> Generator[PROPOSAL, None, None]:
        """Yield all valid proposals with the given tied Delta K values."""
        for delta_K in delta_Ks:
            for action in self.actions_by_delta_K[delta_K]:
                yield from self._iter_valid_proposals(action)

    def _sample_tied_proposal(
        self, rng: np.random.Generator, delta_Ks: list[int]
    ) -> PROPOSAL | None:
        """Uniformly sample one proposal without storing the complete tie group."""
        selected: PROPOSAL | None = None
        for num_valid, proposal in enumerate(
            self._iter_tied_proposals(delta_Ks), start=1
        ):
            if int(rng.integers(num_valid)) == 0:
                selected = proposal
        return selected

    @staticmethod
    def _accumulate_action_flux(
        joint_degree_flux: npt.NDArray[np.float32],
        action: RewireAction,
        weight: np.float32,
    ) -> None:
        """Add a weighted class-level rewiring to a joint degree flux matrix."""
        degree1, degree2 = action.edge_class1
        degree3, degree4 = action.edge_class2
        old_edges = ((degree1, degree2), (degree3, degree4))
        if action.mode == 0:
            new_edges = ((degree1, degree3), (degree2, degree4))
        else:
            new_edges = ((degree1, degree4), (degree2, degree3))

        for old_degree1, old_degree2 in old_edges:
            joint_degree_flux[old_degree1 - 1, old_degree2 - 1] -= weight
            joint_degree_flux[old_degree2 - 1, old_degree1 - 1] -= weight
        for new_degree1, new_degree2 in new_edges:
            joint_degree_flux[new_degree1 - 1, new_degree2 - 1] += weight
            joint_degree_flux[new_degree2 - 1, new_degree1 - 1] += weight

    def get_joint_degree_flux(self, target: float) -> npt.NDArray[np.float32]:
        """Return the expected one-step joint degree matrix change."""
        max_degree = int(self.degrees.max())

        for delta_Ks in self._iter_improving_delta_K(target):
            joint_degree_flux = np.zeros((max_degree, max_degree), dtype=np.float32)
            num_valid = 0
            for delta_K in delta_Ks:
                for action in self.actions_by_delta_K[delta_K]:
                    action_count = sum(1 for _ in self._iter_valid_proposals(action))
                    if action_count == 0:
                        continue

                    self._accumulate_action_flux(
                        joint_degree_flux, action, np.float32(action_count)
                    )
                    num_valid += action_count

            if num_valid > 0:
                return joint_degree_flux / np.float32(num_valid)

        raise RuntimeError("No valid improving rewiring exists")

    def sample_rewire(self, rng: np.random.Generator, target: float) -> PROPOSAL:
        """ Return a random valid rewiring among the tied actions. """
        # Delta_K are yielded by increasing post-rewiring loss.
        # A group can contain two Delta K values equidistant from the current gap,
        # including one that approaches the target and one that overshoots it.
        for delta_Ks in self._iter_improving_delta_K(target):
            proposal = self._sample_tied_proposal(rng, delta_Ks)
            if proposal is not None:
                return proposal

        raise RuntimeError("No valid improving rewiring exists")

    def _apply_rewire(
        self, edge_indices: tuple[int, int], new_edges: tuple[EDGE, EDGE]
    ) -> None:
        old_edge_classes = [
            self._get_edge_class(tuple(self.edge_list[edge_idx].tolist()))
            for edge_idx in edge_indices
        ]
        rewire(
            edge_indices,
            new_edges,
            self.edge_list,
            self.adjacency,
            self.degrees,
            self.joint_degree_matrix,
        )

        # Update the class membership of the rewired edges
        for edge_idx, old_class, new_edge in zip(
            edge_indices, old_edge_classes, new_edges
        ):
            # Remove old edges
            self.edges_by_class[old_class].remove(edge_idx)

            # Add new edges
            new_class = self._get_edge_class(new_edge)
            self.edges_by_class.setdefault(new_class, set()).add(edge_idx)

            # Remove empty classes
            if not self.edges_by_class[old_class]:
                del self.edges_by_class[old_class]

    def __call__(
        self,
        seed: int,
        max_steps: int,
        target: float,
        log_interval: int | None,
        store: list[STORE],
    ) -> GreedyData:
        """Run a first-hitting degree-class greedy search."""
        rng = np.random.default_rng(seed)
        start = time.perf_counter()

        if log_interval is None:
            log_interval = max_steps

        assortativity_history = [self.assortativity]
        joint_degree_matrix_history: list[npt.NDArray[np.uint32]] = []
        joint_degree_flux_history: list[npt.NDArray[np.float32]] = []

        is_success = False
        num_steps = 0
        for _ in range(max_steps):

            if np.abs(target - self.assortativity) < self.physical_tolerance:
                is_success = True
                break

            if "joint_degree" in store:
                joint_degree_matrix_history.append(self.joint_degree_matrix.copy())
                joint_degree_flux_history.append(self.get_joint_degree_flux(target))

            edge_indices, new_edges = self.sample_rewire(rng, target)
            self._apply_rewire(edge_indices, new_edges)
            self.assortativity = get_assortativity(
                self.joint_degree_matrix,
                self.num_edges,
                self.marginal_prob,
                self.degree_variance,
            )

            assortativity_history.append(self.assortativity)
            num_steps += 1

            if num_steps % log_interval == 0:
                print(
                    (f"Step {num_steps}: assortativity={self.assortativity:.4f}"),
                    flush=True,
                )

        if "joint_degree" in store:
            joint_degree_matrix_history.append(self.joint_degree_matrix.copy())

        end = time.perf_counter()

        return GreedyData(
            graph_type=self.graph_type,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            graph_seed=self.graph_seed,
            reset_method=self.reset_method,
            seed=seed,
            max_steps=max_steps,
            target=target,
            tolerance=self.tolerance,
            physical_tolerance=float(self.physical_tolerance),
            is_success=is_success,
            num_steps=num_steps,
            edge_list=self.edge_list.copy(),
            assortativity_history=np.asarray(assortativity_history),
            joint_degree_matrix_history=np.asarray(joint_degree_matrix_history),
            joint_degree_flux_history=np.asarray(joint_degree_flux_history),
            runtime=end - start,
        )


def main() -> None:
    args = get_args()
    sampler = GreedySampler(
        args.num_nodes,
        args.num_edges,
        args.graph_type,
        args.graph_seed,
        args.reset_method,
        args.tolerance,
    )

    data_list: list[GreedyData] = []
    for seed in range(args.seeds[0], args.seeds[1]):
        print(f"{seed=} started", flush=True)
        sampler.reset()
        data = sampler(seed, args.max_steps, args.target, args.log_interval, args.store)
        data_list.append(data)

        assortativity = data.assortativity_history[-1]
        print(
            (
                f"{seed=} finished: success={data.is_success}, "
                f"numsteps={data.num_steps}, rho={assortativity:.4f}, "
                f"gap={abs(assortativity - data.target):.4g}, "
                f"runtime={data.runtime:.3f}s"
            ),
            flush=True,
        )

    data_dir = DATA_DIR / "greedy"
    data_dir.mkdir(exist_ok=True)
    filename = (
        f"{args.graph_type}_N{args.num_nodes}_rho{args.target:4f}_"
        f"{datetime.now().strftime('%m%d_%H%M%S_%f')}.h5"
    )
    save(data_dir / filename, data_list, args.store)


if __name__ == "__main__":
    main()
