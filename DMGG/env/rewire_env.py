"""
Notation
N: number of nodes
E: number of edges
D: maximum degree
"""

import copy
import time
from dataclasses import dataclass
from typing import Any, Mapping, Self

import networkx as nx
import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces

from DMGG.env.graph_space import GraphObs, GraphSpace
from DMGG.type_aliases import RESET_METHOD
from graph.assortativity import (
    get_assortativity,
    get_degree_variance,
    get_marginal_prob,
    get_maximum_assortativity,
    get_minimum_assortativity,
    get_joint_degree_matrix,
)
from graph.edge_list import (
    canonical_edge,
    edge_list_to_degrees,
    edge_list_to_graph,
    graph_to_edge_list,
)
from graph.generator import GENERATORS, configuration_model
from graph.type_aliases import EDGE


@dataclass(slots=True)
class GraphProperty:
    num_nodes: int
    num_edges: int
    graph_type: str

    # Undirected edge list (node1, node2) where node1 < node2
    edge_list: npt.NDArray[np.uint32]  # [E, 2]

    # degrees of each node
    degrees: npt.NDArray[np.uint32]
    max_degree: np.uint32
    normalized_degrees: npt.NDArray[np.float32]

    # number of edges between degree i and degree j
    joint_degree_matrix: npt.NDArray[np.uint32]  # [D, D]

    # marginal probability of each degree
    marginal_prob: npt.NDArray[np.float32]  # [D, ]

    # variance of degrees
    degree_variance: np.float32

    @classmethod
    def from_edge_list(
        cls, num_nodes: int, edge_list: npt.NDArray[np.uint32], graph_type: str
    ) -> Self:
        num_edges = len(edge_list)

        degrees = edge_list_to_degrees(num_nodes, edge_list)
        max_degree = degrees.max()
        normalized_degrees = np.divide(degrees, max_degree, dtype=np.float32)

        # For assortativity computation
        joint_degree_matrix = get_joint_degree_matrix(edge_list, degrees)
        marginal_prob = get_marginal_prob(joint_degree_matrix, num_edges)
        degree_variance = get_degree_variance(max_degree, marginal_prob)

        return cls(
            num_nodes,
            num_edges,
            graph_type,
            edge_list,
            degrees,
            max_degree,
            normalized_degrees,
            joint_degree_matrix,
            marginal_prob,
            degree_variance,
        )

    @classmethod
    def from_graph(cls, graph: nx.Graph, graph_type: str) -> Self:
        num_nodes = graph.number_of_nodes()
        edge_list = graph_to_edge_list(graph)

        return cls.from_edge_list(num_nodes, edge_list, graph_type)

    @property
    def graph(self) -> nx.Graph:
        return edge_list_to_graph(self.num_nodes, self.edge_list)

    def rewire(self, edge1_idx: int, edge2_idx: int, mode: int) -> None:
        """
        Rewire the edge

        Args:
            edge1_idx: index of old edge (node1, node2)
            edge2_idx: index of old edge (node3, node4)
            mode: if 0, new edge pair is (node1, node3), (node2, node4).
                  if 1, new edge pair is (node1, node4), (node2, node3).
        """
        # Old edges and nodes
        edge1: EDGE = tuple(self.edge_list[edge1_idx].tolist())
        edge2: EDGE = tuple(self.edge_list[edge2_idx].tolist())
        (node1, node2), (node3, node4) = edge1, edge2

        # New edges and nodes
        if mode == 0:
            new_edge1 = canonical_edge(node1, node3)
            new_edge2 = canonical_edge(node2, node4)
        else:
            new_edge1 = canonical_edge(node1, node4)
            new_edge2 = canonical_edge(node2, node3)
        (new_node1, new_node2), (new_node3, new_node4) = new_edge1, new_edge2

        # ---------------- Edge list ----------------
        self.edge_list[edge1_idx] = np.array(new_edge1, dtype=np.uint32)
        self.edge_list[edge2_idx] = np.array(new_edge2, dtype=np.uint32)

        # ---------------- Joint degree matrix ----------------
        degree1, degree2 = self.degrees[node1], self.degrees[node2]
        degree3, degree4 = self.degrees[node3], self.degrees[node4]
        new_degree1, new_degree2 = self.degrees[new_node1], self.degrees[new_node2]
        new_degree3, new_degree4 = self.degrees[new_node3], self.degrees[new_node4]

        self.joint_degree_matrix[degree1 - 1, degree2 - 1] -= 1
        self.joint_degree_matrix[degree2 - 1, degree1 - 1] -= 1
        self.joint_degree_matrix[degree3 - 1, degree4 - 1] -= 1
        self.joint_degree_matrix[degree4 - 1, degree3 - 1] -= 1
        self.joint_degree_matrix[new_degree1 - 1, new_degree2 - 1] += 1
        self.joint_degree_matrix[new_degree2 - 1, new_degree1 - 1] += 1
        self.joint_degree_matrix[new_degree3 - 1, new_degree4 - 1] += 1
        self.joint_degree_matrix[new_degree4 - 1, new_degree3 - 1] += 1


class RewireEnv(Env):
    """
    Gymnasium-style environment for edge rewiring to reach a target assortativity (rho).

    Observation:
        - node_attr: [N, 1], normalized degrees of each node
        - edge_list: [E, 2], undirected edge list (node1, node2) where node1 < node2
        - glob_attr: [2, ], [rho^* - rho, max_degree]

    Action:
        - edge1_idx: index at edge_list (node1, node2)
        - edge2_idx: index at edge_list (node3, node4)
        - mode: if 0, new edges are (node1, node3), (node2, node4).
                if 1, new edges are (node1, node4), (node2, node3).

    Reward:
        rho_gap = |rho^* - rho|
        Physical tolerance: max(tolerance, 2.0 / (E * D_var))
        - Potential-based reward: Phi(rho_gap) - Phi(rho_gap_last),
          where Phi(rho_gap) = |rho^*| + alpha / rho_gap + alpha.
        - Effective step penalty: min(rho_gap / (10.0 * physical_tolerance), 1) * step_penalty
        - Success bonus: additional reward if rho_gap < physical_tolerance
        - Total reward: potential - effective_step_penalty + success_bonus

    Termination or truncation:
        - |rho_gap| < tolerance => terminated
        - max_rewire_steps <= rewire_step => truncated
    """

    metadata = {"render_modes": []}

    _graph_property: GraphProperty

    obs: GraphObs
    rewires: int
    history: list[np.float32]
    physical_tolerance: np.float32

    target_rho: np.float32
    feasible_rho_range: tuple[float, float]

    def __init__(
        self,
        # ------ Graph -------
        graph_types: list[str],
        num_nodes_range: tuple[int, int],
        mean_degree_range: tuple[float, float],
        # ------ Rewards -------
        target_rho_range: tuple[float, float],
        zeta: float,
        success_bonus: float,
        step_penalty: float,
        # ------ Truncation and termination -------
        tolerance: float,
        max_rewires: int,
    ) -> None:
        """
        Args:
            graph_types: list of graph types to sample from
            num_nodes_min: minimum number of nodes
            num_nodes_max: maximum number of nodes
            mean_degree_min: minimum mean degree
            mean_degree_max: maximum mean degree

            target_rho_range: rho^* will be uniformly sampled from here within the feasible range for the given degree sequence
            zeta: parameter for potential function
            success_bonus: additional reward for convergence
            step_penalty: penalty for each step for faster termination

            tolerance: tolerance for hard-constraint
            max_rewires: maximum number of rewiring steps before truncation
        """
        # ------------- Initial graph -------------
        self.graph_types = graph_types
        self.num_nodes_range = num_nodes_range
        self.mean_degree_range = mean_degree_range

        # ------------- Spaces -------------
        self.observation_space = GraphSpace(
            node_space=spaces.Box(low=0.0, high=1.0, shape=(1,)),
            glob_space=spaces.Box(low=-2.0, high=np.inf, shape=(2,)),
        )

        num_edges_max = int(0.5 * num_nodes_range[1] * mean_degree_range[1])
        self.action_space = spaces.MultiDiscrete([num_edges_max, num_edges_max, 2])

        # ------------ Reward: penalty and potential ------------
        self.target_rho_range = target_rho_range
        self.step_penalty = np.float32(step_penalty)
        self.zeta = np.float32(zeta)
        self.success_reward = np.float32(success_bonus)

        # ------------ Truncation and termination ------------
        self.tolerance = np.float32(tolerance)
        self.max_rewires = max_rewires

    def get_obs(self, rho: np.float32) -> GraphObs:
        return GraphObs(
            node_attr=self.normalized_degrees[:, None],
            edge_list=self.edge_list.astype(np.int64),
            glob_attr=np.array(
                [self.target_rho - rho, self.max_degree],
                dtype=np.float32,
            ),
        )

    def sample_target_rho(self) -> np.float32 | None:
        """
        Target rho is sampled from target range, as long as the range is feasible
        Return None if the target range is infeasible
        """
        # Feasible range with small margin
        feasible_rho_min = self.feasible_rho_range[0] + 0.01
        feasible_rho_max = self.feasible_rho_range[1] - 0.01
        target_rho_min, target_rho_max = self.target_rho_range

        # Target is should be sampled from intersection
        # of the range of feasible rho and the range of target rho
        low = max(feasible_rho_min, target_rho_min)
        high = min(feasible_rho_max, target_rho_max)

        # If the intersection is empty, return None
        if low > high:
            return None

        return np.float32(sample_u_shape(low, high, self.np_random))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GraphObs, dict[str, Any]]:
        super().reset(seed=seed)
        # ------------- Create random initial graph -------------
        self._graph_property, self.feasible_rho_range = self.create_new_graph()

        # Compute assortativity of initial graph
        rho = get_assortativity(
            self.joint_degree_matrix,
            self.num_edges,
            self.marginal_prob,
            self.degree_variance,
        )

        # Physically valid tolerance for small graphs
        # Single rewire change assortativity by at least 2 / (E * D_var)
        self.physical_tolerance = np.maximum(
            self.tolerance,
            2.0 / (self.num_edges * self.degree_variance),
            dtype=np.float32,
        )

        # ------------- Set random target assortativity -------------
        target_rho = self.sample_target_rho()
        assert target_rho is not None
        self.target_rho = target_rho

        # ------------- Initialization -------------
        self.obs = self.get_obs(rho)
        self.rewires = 0
        self.history = [rho]

        return self.obs, {}

    def step(
        self, action: npt.NDArray[np.int64]
    ) -> tuple[GraphObs, float, bool, bool, dict[str, Any]]:
        """
        Args:
            action: [edge1_idx, edge2_idx, mode]
        Return:
            [observation, reward, terminated, truncated, info]
        """
        self.rewires += 1
        edge1_idx, edge2_idx, mode = action

        # Apply rewiring to the graph
        self._graph_property.rewire(int(edge1_idx), int(edge2_idx), int(mode))

        # Compute assortativity
        rho = get_assortativity(
            self.joint_degree_matrix,
            self.num_edges,
            self.marginal_prob,
            self.degree_variance,
        )

        # Compute reward
        rho_gap = np.float32(np.abs(self.target_rho - rho))
        potential = self.get_potential(self.target_rho, rho_gap)

        last_rho_gap = np.float32(np.abs(self.target_rho - self.history[-1]))
        last_potential = self.get_potential(self.target_rho, last_rho_gap)

        # Base reward from potential function
        reward = potential - last_potential

        # Step penalty: when rho_gap is within 10 * physical_tolerance,
        # reduce the penalty, proportional to rho_gap
        step_penalty = (
            np.minimum(rho_gap / (10.0 * self.physical_tolerance), 1.0)
            * self.step_penalty
        )
        reward += step_penalty

        # Check termination or truncation
        terminated = bool(rho_gap < self.physical_tolerance)
        if terminated:
            # Add bonus reward for convergence to target assortativity
            reward += self.success_reward
            truncated = False
        else:
            # Truncate if max rewire steps is reached
            truncated = bool(self.rewires >= self.max_rewires)

        # Update observation and history
        self.obs = self.get_obs(rho)
        self.history.append(rho)

        return self.obs, float(reward), terminated, truncated, {}

    def create_new_graph(self) -> tuple[GraphProperty, tuple[float, float]]:
        # Randomly sample graph type, number of nodes, and mean degree
        while True:
            graph_type = self.np_random.choice(self.graph_types)
            num_nodes = int(
                self.np_random.integers(*self.num_nodes_range, endpoint=True)
            )
            mean_degree = self.np_random.uniform(*self.mean_degree_range)

            edge_list = GENERATORS[graph_type](num_nodes, mean_degree, self.np_random)
            degrees = edge_list_to_degrees(num_nodes, edge_list)

            # Configuration model to remove initial topology-dependence
            edge_list = configuration_model(degrees, self.np_random)
            if len(edge_list) == 0:
                continue

            feasible_rho_min = get_minimum_assortativity(
                degrees, edge_list, self.np_random
            )
            feasible_rho_max = get_maximum_assortativity(
                degrees, edge_list, self.np_random
            )

            if (
                (feasible_rho_min is not None)
                and (feasible_rho_max is not None)
                and (feasible_rho_min + 0.02 < feasible_rho_max)
            ):
                break

        graph_property = GraphProperty.from_edge_list(num_nodes, edge_list, graph_type)

        return graph_property, (float(feasible_rho_min), float(feasible_rho_max))

    def get_potential(self, target: np.float32, gap: np.float32) -> np.float32:
        return np.divide(
            abs(target) + self.zeta,
            gap + self.zeta,
            dtype=np.float32,
        )

    def get_rng_state(self) -> Mapping[str, Any]:
        return self.np_random.bit_generator.state

    def set_rng_state(self, state: dict[str, Any]) -> None:
        self.np_random.bit_generator.state = state

    @property
    def num_nodes(self) -> int:
        return self._graph_property.num_nodes

    @property
    def num_edges(self) -> int:
        return self._graph_property.num_edges

    @property
    def mean_degree(self) -> float:
        return float(2 * self.num_edges / self.num_nodes)

    @property
    def graph_type(self) -> str:
        return self._graph_property.graph_type

    @property
    def graph(self) -> nx.Graph:
        return self._graph_property.graph

    @property
    def edge_list(self) -> npt.NDArray[np.uint32]:
        return self._graph_property.edge_list

    @property
    def degrees(self) -> npt.NDArray[np.uint32]:
        return self._graph_property.degrees

    @property
    def max_degree(self) -> np.uint32:
        return self._graph_property.max_degree

    @property
    def normalized_degrees(self) -> npt.NDArray[np.float32]:
        return self._graph_property.normalized_degrees

    @property
    def joint_degree_matrix(self) -> npt.NDArray[np.uint32]:
        return self._graph_property.joint_degree_matrix

    @property
    def marginal_prob(self) -> npt.NDArray[np.float32]:
        return self._graph_property.marginal_prob

    @property
    def degree_variance(self) -> np.float32:
        return self._graph_property.degree_variance


class EvalRewireEnv(RewireEnv):
    """
    Evaluation variant of RewireEnv
    """

    # Initial graph (if provided)
    _init_graph_property: GraphProperty | None
    _init_feasible_rho_range: tuple[float, float] | None

    _infeasible_target: bool

    # Runtime
    start_time: float
    end_time: float

    def __init__(
        self,
        reset_method: RESET_METHOD = "soft",
        init_graph: nx.Graph | None = None,
        init_graph_type: str | None = None,
        init_feasible_rho_range: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._reset_method = reset_method

        # Do the hard reset: create new initial graph for every sampling
        if self._reset_method == "hard":
            self._init_graph_property = None
            self._init_feasible_rho_range = None

        # Do the soft or none reset
        # soft reset: keep degree sequence but use new configuration model
        # none reset: reuse initial graph for every sampling
        else:
            assert init_graph is not None
            assert init_graph_type is not None
            assert init_feasible_rho_range is not None

            self._init_graph_property = GraphProperty.from_graph(
                init_graph, init_graph_type
            )
            self._init_feasible_rho_range = init_feasible_rho_range

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GraphObs, dict[str, Any]]:
        Env.reset(self, seed=seed)

        # Do the hard reset: create new initial graph for every sampling
        if self._reset_method == "hard":
            self._graph_property, self.feasible_rho_range = self.create_new_graph()
        else:
            assert self._init_graph_property is not None
            assert self._init_feasible_rho_range is not None

            # Feasible rho range is fixed when degree sequence is fixed
            self.feasible_rho_range = self._init_feasible_rho_range

            # If do soft reset, use configuration model
            if self._reset_method == "soft":
                while True:
                    edge_list = configuration_model(
                        self._init_graph_property.degrees, self.np_random
                    )
                    if len(edge_list) > 0:
                        break

                self._graph_property = GraphProperty.from_edge_list(
                    self._init_graph_property.num_nodes,
                    edge_list,
                    self._init_graph_property.graph_type,
                )
            # If not reset, reuse initial graph
            else:
                self._graph_property = copy.deepcopy(self._init_graph_property)

        # Compute assortativity of initial graph
        rho = get_assortativity(
            self.joint_degree_matrix,
            self.num_edges,
            self.marginal_prob,
            self.degree_variance,
        )

        # Physically valid tolerance for small graphs
        # Single rewire change assortativity by at least 2 / (E * D_var)
        self.physical_tolerance = np.maximum(
            self.tolerance,
            2.0 / (self.num_edges * self.degree_variance),
            dtype=np.float32,
        )

        # ------------- Set random target assortativity -------------
        target_rho = self.sample_target_rho()
        if target_rho is None:
            self._infeasible_target = True
            self.target_rho = np.float32(0.0)  # Will not be used anyway
        else:
            self._infeasible_target = False
            self.target_rho = target_rho

        # ------------- Initialization -------------
        self.obs = self.get_obs(rho)
        self.rewires = 0
        self.history = [rho]

        # New episode starts from here
        self.start_time = time.perf_counter()

        return self.obs, {}

    def get_info(self) -> dict[str, Any]:
        return {
            "graph_type": self.graph_type,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "mean_degree": self.mean_degree,
            "edge_list": self.edge_list,
            "target_rho": self.target_rho,
            "feasible_rho_range": self.feasible_rho_range,
            "history": self.history,
            "runtime": self.end_time - self.start_time,
            "joint_degree_matrix": self.joint_degree_matrix,
        }

    def step(
        self, action: npt.NDArray[np.int64]
    ) -> tuple[GraphObs, float, bool, bool, dict[str, Any]]:
        # When target range is infeasible, return 0 reward and truncate episode
        if self._infeasible_target:
            self.end_time = time.perf_counter()

            return self.obs, 0.0, False, True, self.get_info()

        # Discard reward as we only evaluate the policy
        self.obs, _, terminated, truncated, _ = super().step(action)

        info = {}
        # Add additional information if episode is finished
        if terminated or truncated:
            self.end_time = time.perf_counter()

            info = self.get_info()

        return self.obs, 0.0, terminated, truncated, info


def sample_u_shape(
    low: int | float, high: int | float, rng: np.random.Generator, strength: float = 0.5
) -> int | float:
    """
    Sample random number, where probability of sampling is high in both ends of the range (U-shape)
    Args:
        low: lower bound (inclusive)
        high: upper bound (inclusive)
        strength: strength of bias. 0.0 is uniform, larger the value is, stronger the bias.
    """
    alpha = 1.0 / (1.0 + strength)
    biased = rng.beta(alpha, alpha)
    if isinstance(low, int):
        return int(low + (high - low + 1) * biased)
    else:
        return low + (high - low) * biased
