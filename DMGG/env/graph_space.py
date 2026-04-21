from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.space import Space

from DMGG.env.graph_obs import GraphObs


class GraphSpace(Space[GraphObs]):
    """
    Space for GraphObs.

    node_attr: [N, ...] sampled from `node_space`
    edge_attr: [E, ...] sampled from `edge_space` (optional)
    edge_links: [E, 2] integer node indices
    glob_attr: [...] sampled from `glob_space` (optional)
    """

    def __init__(
        self,
        node_space: spaces.Space,
        glob_space: spaces.Space,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self.node_space = node_space
        self.glob_space = glob_space
        super().__init__(shape=None, dtype=None, seed=seed)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def seed(self, seed: int | None = None) -> tuple[Any, ...]: # type:ignore[return-value]
        seeds: list[Any] = [super().seed(seed), self.node_space.seed(seed)]
        if self.glob_space is not None:
            seeds.append(self.glob_space.seed(seed))
        return tuple(seeds)

    def __repr__(self) -> str:
        return (
            "GraphSpace("
            f"node_space={self.node_space}, "
            f"glob_space={self.glob_space})"
        )

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, GraphSpace)
            and self.node_space == other.node_space
            and self.glob_space == other.glob_space
        )
