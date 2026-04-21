from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class GraphObs(NamedTuple):
    """
    Variation of gymnasium.spaces.graph.GraphInstance

    node_attr: [N, NF]
    edge_list: [E, 2], undirected edge list (node1, node2) where node1 < node2
    glob_attr: [GF,]
    """

    node_attr: npt.NDArray[np.float32]
    edge_list: npt.NDArray[np.int64]
    glob_attr: npt.NDArray[np.float32]


GraphVecObs = list[GraphObs]
