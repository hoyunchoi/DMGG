from dataclasses import dataclass, field
from typing import Self, cast

import numpy as np
import torch

from DMGG.env.graph_obs import GraphObs, GraphVecObs


@dataclass(slots=True)
class PygObs:
    """
    Observation in PyG format

    WARNING
    default edge representations are uni-directional.
    i.e. if (u, v) is in edge_list, then (v, u) is not in edge_list.
    To get the PyG version of edges, use bi_edge_list
    As bi_edge_list is a tensor of shape [2BE, 2], do edge_index = bi_edge_list.T
    """

    node_attr: torch.Tensor  # [BN, NF]
    edge_list: torch.LongTensor  # [BE, 2]
    glob_attr: torch.Tensor  # [B, GF]
    node_batch: torch.LongTensor  # [BN, ]
    node_ptr: torch.LongTensor  # [B+1, ]
    edge_batch: torch.LongTensor  # [BE, ]
    edge_ptr: torch.LongTensor  # [B+1, ]

    bi_edge_list: torch.LongTensor = field(init=False)  # [2BE, 2]
    bi_edge_batch: torch.LongTensor = field(init=False)  # [2BE, ]

    def __post_init__(self) -> None:
        """Initialize bi-directional edge index, attribute, batch, and pointer"""

        # [(u1, v1), (u1, v2), ...] -> [(u1, v1), (v1, u1), (u1, v2), (v2, u1), ...]
        # Same as follows
        # self.bi_edge_list = torch.empty(2*self.total_num_edges, 2)
        # self.bi_edge_list[::2] = self.edge_list
        # self.bi_edge_list[1::2] = self.edge_list.flip(1)
        self.bi_edge_list = cast(
            torch.LongTensor,
            torch.stack([self.edge_list, self.edge_list.flip(1)], dim=1).reshape(-1, 2),
        )

        # [E1, E2, ...] -> [E1, E1, E2, E2, ...]
        self.bi_edge_batch = cast(
            torch.LongTensor, self.edge_batch.repeat_interleave(2, dim=0)
        )

    @classmethod
    def from_graph_vec_obs(
        cls, graph_vec_obs: GraphVecObs, device: torch.device
    ) -> Self:
        num_nodes = np.array([len(go.node_attr) for go in graph_vec_obs], dtype=np.uint32)
        num_edges = np.array([len(go.edge_list) for go in graph_vec_obs], dtype=np.uint32)
        num_graphs = len(graph_vec_obs)

        node_batch = np.repeat(np.arange(num_graphs, dtype=np.int64), num_nodes)
        node_ptr = np.empty(num_graphs + 1, dtype=np.int64)
        node_ptr[0] = 0
        node_ptr[1:] = num_nodes.cumsum(0)
        edge_batch = np.repeat(np.arange(num_graphs, dtype=np.int64), num_edges)
        edge_ptr = np.empty(num_graphs + 1, dtype=np.int64)
        edge_ptr[0] = 0
        edge_ptr[1:] = num_edges.cumsum(0)

        # B * [N, NF] -> [BN, NF]
        node_attr = np.concatenate([go.node_attr for go in graph_vec_obs], axis=0)

        # Add offset for graph index and B * [E, 2] -> [BE, 2]
        edge_list = np.concatenate(
            [np + el for np, el in zip(node_ptr[:-1], [go.edge_list for go in graph_vec_obs])], axis=0
        )

        # B * [GF,] -> [B, GF]
        glob_attr = np.stack([go.glob_attr for go in graph_vec_obs], axis=0)

        return cls(
            node_attr=torch.as_tensor(node_attr, device=device),
            edge_list=cast(torch.LongTensor, torch.as_tensor(edge_list, device=device)),
            glob_attr=torch.as_tensor(glob_attr, device=device),
            node_batch=cast(
                torch.LongTensor, torch.as_tensor(node_batch, device=device)
            ),
            node_ptr=cast(torch.LongTensor, torch.as_tensor(node_ptr, device=device)),
            edge_batch=cast(
                torch.LongTensor, torch.as_tensor(edge_batch, device=device)
            ),
            edge_ptr=cast(torch.LongTensor, torch.as_tensor(edge_ptr, device=device)),
        )

    @classmethod
    def from_graph_obs(
        cls, graph_obs: GraphObs, device: torch.device
    ) -> Self:
        num_nodes = len(graph_obs.node_attr)
        num_edges = len(graph_obs.edge_list)

        node_batch = np.zeros(num_nodes, dtype=np.int64)
        node_ptr = np.array([0, num_nodes], dtype=np.int64)
        edge_batch = np.zeros(num_edges, dtype=np.int64)
        edge_ptr = np.array([0, num_edges], dtype=np.int64)

        return cls(
            node_attr=torch.as_tensor(graph_obs.node_attr, device=device),
            edge_list=cast(
                torch.LongTensor,
                torch.as_tensor(graph_obs.edge_list, device=device),
            ),
            glob_attr=torch.as_tensor(graph_obs.glob_attr[None, ...], device=device),
            node_batch=cast(
                torch.LongTensor, torch.as_tensor(node_batch, device=device)
            ),
            node_ptr=cast(torch.LongTensor, torch.as_tensor(node_ptr, device=device)),
            edge_batch=cast(
                torch.LongTensor, torch.as_tensor(edge_batch, device=device)
            ),
            edge_ptr=cast(torch.LongTensor, torch.as_tensor(edge_ptr, device=device)),
        )

    def __getitem__(self, index: int) -> Self:
        node_start, node_end = int(self.node_ptr[index]), int(self.node_ptr[index + 1])
        edge_start, edge_end = int(self.edge_ptr[index]), int(self.edge_ptr[index + 1])
        num_nodes = node_end - node_start
        num_edges = edge_end - edge_start

        # Subtract offset for graph index at edge list
        edge_list = self.edge_list[edge_start:edge_end] - self.node_ptr[index]

        # Batch and ptr
        node_batch = torch.zeros(num_nodes, dtype=torch.int64, device=self.device)
        edge_batch = torch.zeros(num_edges, dtype=torch.int64, device=self.device)
        node_ptr = torch.tensor([0, num_nodes], dtype=torch.int64, device=self.device)
        edge_ptr = torch.tensor([0, num_edges], dtype=torch.int64, device=self.device)

        return self.__class__(
            node_attr=self.node_attr[node_start:node_end],  # [N, NF]
            edge_list=cast(torch.LongTensor, edge_list),  # [E, 2]
            glob_attr=self.glob_attr[index].unsqueeze(0),  # [1, GF]
            node_batch=cast(torch.LongTensor, node_batch),  # [N, ]
            node_ptr=cast(torch.LongTensor, node_ptr),  # [2, ]
            edge_batch=cast(torch.LongTensor, edge_batch),  # [E, ]
            edge_ptr=cast(torch.LongTensor, edge_ptr),  # [2, ]
        )

    @property
    def total_num_nodes(self) -> int:
        return len(self.node_attr)

    @property
    def total_num_edges(self) -> int:
        return len(self.edge_list)

    @property
    def num_graphs(self) -> int:
        return len(self.glob_attr)

    @property
    def device(self) -> torch.device:
        return self.node_attr.device
