from dataclasses import dataclass
from typing import NamedTuple, Self, cast

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops

from DMGG.env import GraphObs, PygObs


class ResiNetObs(NamedTuple):
    """One action graph and its raw/one-node-removed FireGNN views."""

    action: GraphObs
    fire_views: tuple[GraphObs, GraphObs]
    fire_node_to_action: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]
    fire_edge_to_action: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]
    view_valid: npt.NDArray[np.bool_]
    first_edge_mask: npt.NDArray[np.bool_]


ResiNetVecObs = list[ResiNetObs]


@dataclass(slots=True)
class ResiNetPygObs:
    """Batched action graphs and their two FireGNN views.

    ``fire_edge_to_action`` follows the interleaved order of
    ``fire.bi_edge_list``. Each entry points to the graph-contiguous directed
    action-edge layout ``[canonical edges, reversed edges]``.
    """

    action: PygObs
    fire: PygObs
    fire_node_to_action: torch.LongTensor
    fire_edge_to_action: torch.LongTensor
    fire_edge_index: torch.LongTensor
    action_to_original_fire_edge: torch.LongTensor
    view_to_graph: torch.LongTensor
    view_valid: torch.BoolTensor
    first_edge_mask: torch.BoolTensor
    directed_edge_ptr: torch.LongTensor
    directed_edge_batch: torch.LongTensor
    canonical_edge: torch.LongTensor
    reversed_edge: torch.LongTensor

    @classmethod
    def from_graph_vec_obs(
        cls, graph_vec_obs: ResiNetVecObs, device: torch.device
    ) -> Self:
        action = PygObs.from_graph_vec_obs(
            [obs.action for obs in graph_vec_obs], device
        )
        fire_views = [view for obs in graph_vec_obs for view in obs.fire_views]
        fire = PygObs.from_graph_vec_obs(fire_views, device)

        node_maps: list[npt.NDArray[np.int64]] = []
        directed_edge_maps: list[npt.NDArray[np.int64]] = []
        view_valid: list[npt.NDArray[np.bool_]] = []
        node_offset = 0
        edge_offset = 0

        for obs in graph_vec_obs:
            num_edges = len(obs.action.edge_list)
            directed_offset = 2 * edge_offset

            for view_idx in range(2):
                node_maps.append(obs.fire_node_to_action[view_idx] + node_offset)

                edge_map = obs.fire_edge_to_action[view_idx]
                canonical = directed_offset + edge_map
                reversed_ = directed_offset + num_edges + edge_map
                directed_edge_maps.append(
                    np.stack([canonical, reversed_], axis=1).reshape(-1)
                )

            view_valid.append(obs.view_valid)
            node_offset += len(obs.action.node_attr)
            edge_offset += num_edges

        fire_node_to_action = np.concatenate(node_maps)
        fire_edge_to_action = np.concatenate(directed_edge_maps)
        view_to_graph = np.repeat(np.arange(len(graph_vec_obs), dtype=np.int64), 2)
        first_edge_mask = np.concatenate([obs.first_edge_mask for obs in graph_vec_obs])

        fire_edge_to_action_tensor = cast(
            torch.LongTensor,
            torch.as_tensor(fire_edge_to_action, device=device),
        )
        fire_edge_index = fire.bi_edge_list.T.contiguous()
        fire_edge_index, _ = remove_self_loops(fire_edge_index)
        fire_edge_index, _ = add_self_loops(
            fire_edge_index, num_nodes=fire.total_num_nodes
        )

        edge_count = action.edge_ptr[1:] - action.edge_ptr[:-1]
        directed_edge_ptr = torch.zeros(
            action.num_graphs + 1, dtype=torch.int64, device=device
        )
        directed_edge_ptr[1:] = (2 * edge_count).cumsum(dim=0)
        directed_edge_batch = torch.arange(
            action.num_graphs, device=device
        ).repeat_interleave(2 * edge_count)

        global_edge = torch.arange(action.total_num_edges, device=device)
        local_edge = global_edge - action.edge_ptr[action.edge_batch]
        canonical_edge = directed_edge_ptr[action.edge_batch] + local_edge
        reversed_edge = canonical_edge + edge_count[action.edge_batch]

        original_view = torch.arange(fire.num_graphs, device=device).remainder(2) == 0
        original_fire_edge = torch.where(original_view[fire.bi_edge_batch])[0]
        action_to_original_fire_edge = torch.empty(
            len(directed_edge_batch), dtype=torch.int64, device=device
        )
        action_to_original_fire_edge[fire_edge_to_action_tensor[original_fire_edge]] = (
            original_fire_edge
        )

        return cls(
            action=action,
            fire=fire,
            fire_node_to_action=cast(
                torch.LongTensor,
                torch.as_tensor(fire_node_to_action, device=device),
            ),
            fire_edge_to_action=fire_edge_to_action_tensor,
            fire_edge_index=cast(torch.LongTensor, fire_edge_index),
            action_to_original_fire_edge=cast(
                torch.LongTensor, action_to_original_fire_edge
            ),
            view_to_graph=cast(
                torch.LongTensor, torch.as_tensor(view_to_graph, device=device)
            ),
            view_valid=cast(
                torch.BoolTensor,
                torch.as_tensor(np.concatenate(view_valid), device=device),
            ),
            first_edge_mask=cast(
                torch.BoolTensor,
                torch.as_tensor(first_edge_mask, device=device),
            ),
            directed_edge_ptr=cast(torch.LongTensor, directed_edge_ptr),
            directed_edge_batch=cast(torch.LongTensor, directed_edge_batch),
            canonical_edge=cast(torch.LongTensor, canonical_edge),
            reversed_edge=cast(torch.LongTensor, reversed_edge),
        )

    @classmethod
    def from_graph_obs(cls, graph_obs: ResiNetObs, device: torch.device) -> Self:
        return cls.from_graph_vec_obs([graph_obs], device)

    @property
    def num_graphs(self) -> int:
        return self.action.num_graphs

    @property
    def device(self) -> torch.device:
        return self.action.device
