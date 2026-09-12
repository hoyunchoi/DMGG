from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv

from DMGG.env import PygObs
from DMGG.scatter import scatter_sum


class GraphNorm(nn.Module):
    """GraphNorm used by the official ResiNet implementation."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self, x: torch.Tensor, batch: torch.LongTensor, num_graphs: int
    ) -> torch.Tensor:
        count = torch.bincount(batch, minlength=num_graphs).to(x.dtype)
        mean = scatter_sum(x, batch, dim=0, dim_size=num_graphs)
        mean = mean / count.clamp_min(1).unsqueeze(-1)

        centered = x - mean[batch] * self.mean_scale
        variance = scatter_sum(centered.square(), batch, dim=0, dim_size=num_graphs)
        variance = variance / count.clamp_min(1).unsqueeze(-1)

        return self.weight * centered / torch.sqrt(variance[batch] + 1e-6) + self.bias


class GINNodeMLP(nn.Module):
    """The two-linear-layer ApplyNodeFunc used inside each official GINConv."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.inner_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.inner_batch_norm(self.linear1(x)))
        x = self.linear2(x)
        return F.selu(self.output_batch_norm(x))


class BiGRUJumpingKnowledge(nn.Module):
    """Attention-weighted layer aggregation from ResiNet's custom JK module."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(2 * hidden_dim, 1)

    def forward(self, hidden: list[torch.Tensor]) -> torch.Tensor:
        layer_sequence = torch.stack(hidden, dim=1)
        gru_output, _ = self.gru(layer_sequence)
        weight = torch.softmax(self.attention(gru_output).squeeze(-1), dim=1)
        return (layer_sequence * weight.unsqueeze(-1)).sum(dim=1)


class ResiNetGIN(nn.Module):
    """PyG port of the five-layer GIN backbone in the official repository."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 64, num_layers: int = 5):
        super().__init__()
        if num_layers < 2:
            raise ValueError(
                "ResiNet GIN needs at least two layers for jumping knowledge"
            )

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(
                GINConv(
                    GINNodeMLP(layer_input_dim, hidden_dim),
                    eps=0.0,
                    train_eps=True,
                )
            )
            self.graph_norms.append(
                GraphNorm(layer_input_dim if layer == 0 else hidden_dim)
            )

        self.jumping_knowledge = BiGRUJumpingKnowledge(hidden_dim)

    def forward(self, graph: PygObs, edge_index: torch.LongTensor) -> torch.Tensor:
        # This explicit self-loop is intentional. The official DGL graph receives
        # one self-loop before GINConv, in addition to GINConv's trainable center term.
        hidden = self.layers[0](graph.node_attr.float(), edge_index)
        layer_history: list[torch.Tensor] = []
        for layer in range(1, self.num_layers):
            hidden = self.layers[layer](hidden, edge_index)
            hidden = F.selu(hidden)
            hidden = self.graph_norms[layer](hidden, graph.node_batch, graph.num_graphs)
            hidden = F.normalize(hidden, p=2, dim=-1)
            layer_history.append(hidden)

        # As in the official code, layer 0 is deliberately absent from JK.
        hidden = self.jumping_knowledge(layer_history)
        return F.normalize(hidden, p=2, dim=-1)


@dataclass
class FireGNNOutput:
    node_embedding: torch.Tensor
    directed_edge_embedding: torch.Tensor | None
    view_embedding: torch.Tensor
    graph_embedding: torch.Tensor


class FireGNN(nn.Module):
    """Encode all filtration views and retain their aligned multi-view outputs."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 5,
        encode_edges: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encode_edges = encode_edges
        self.gin = ResiNetGIN(input_dim, hidden_dim, num_layers)
        self.edge_merger = (
            nn.Linear(6 * hidden_dim, hidden_dim) if encode_edges else None
        )

    def forward(
        self,
        fire: PygObs,
        edge_index: torch.LongTensor,
        view_to_graph: torch.LongTensor,
        view_valid: torch.BoolTensor,
        num_graphs: int,
    ) -> FireGNNOutput:
        node_embedding = self.gin(fire, edge_index)

        view_embedding = scatter_sum(
            node_embedding,
            fire.node_batch,
            dim=0,
            dim_size=fire.num_graphs,
        )
        valid_weight = view_valid.to(view_embedding.dtype).unsqueeze(-1)
        graph_embedding = scatter_sum(
            view_embedding * valid_weight,
            view_to_graph,
            dim=0,
            dim_size=num_graphs,
        )
        view_count = scatter_sum(
            view_valid.to(view_embedding.dtype),
            view_to_graph,
            dim=0,
            dim_size=num_graphs,
        )
        graph_embedding = graph_embedding / (view_count.unsqueeze(-1) + 1e-5)

        directed_edge_embedding: torch.Tensor | None = None
        if self.edge_merger is not None:
            source, target = fire.bi_edge_list.T
            source_embedding = node_embedding[source]
            target_embedding = node_embedding[target]
            edge_sum = source_embedding + target_embedding
            directed_edge_embedding = self.edge_merger(
                torch.cat(
                    [
                        torch.minimum(source_embedding, target_embedding),
                        torch.maximum(source_embedding, target_embedding),
                        0.5 * edge_sum,
                        edge_sum,
                        source_embedding - target_embedding,
                        source_embedding * target_embedding,
                    ],
                    dim=-1,
                )
            )

        return FireGNNOutput(
            node_embedding=node_embedding,
            directed_edge_embedding=directed_edge_embedding,
            view_embedding=view_embedding,
            graph_embedding=graph_embedding,
        )


class FiltrationEdgeAggregator(nn.Module):
    """Official sigmoid-gated, unnormalized sum over filtration views."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.edge_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        directed_edge_embedding: torch.Tensor,
        view_embedding: torch.Tensor,
        edge_to_view: torch.LongTensor,
        edge_to_action: torch.LongTensor,
        num_action_directed_edges: int,
    ) -> torch.Tensor:
        contribution = self.gate(view_embedding[edge_to_view]) * self.edge_projection(
            directed_edge_embedding
        )
        return scatter_sum(
            contribution,
            edge_to_action,
            dim=0,
            dim_size=num_action_directed_edges,
        )


class DualFiltrationEdgeAggregator(nn.Module):
    """Evaluate the two independent filtration aggregators in one edge pass."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.first = FiltrationEdgeAggregator(hidden_dim)
        self.second = FiltrationEdgeAggregator(hidden_dim)

    def forward(
        self,
        directed_edge_embedding: torch.Tensor,
        view_embedding: torch.Tensor,
        edge_to_view: torch.LongTensor,
        edge_to_action: torch.LongTensor,
        num_action_directed_edges: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_input = view_embedding[edge_to_view]
        first_gate = cast(nn.Linear, self.first.gate[0])
        second_gate = cast(nn.Linear, self.second.gate[0])
        gates = torch.sigmoid(
            F.linear(
                gate_input,
                torch.cat([first_gate.weight, second_gate.weight], dim=0),
                torch.cat([first_gate.bias, second_gate.bias], dim=0),
            )
        ).unsqueeze(-1)

        projections = F.linear(
            directed_edge_embedding,
            torch.cat(
                [
                    self.first.edge_projection.weight,
                    self.second.edge_projection.weight,
                ],
                dim=0,
            ),
            torch.cat(
                [
                    self.first.edge_projection.bias,
                    self.second.edge_projection.bias,
                ],
                dim=0,
            ),
        ).unflatten(-1, (2, self.hidden_dim))
        aggregated = scatter_sum(
            gates * projections,
            edge_to_action,
            dim=0,
            dim_size=num_action_directed_edges,
        )
        return aggregated[:, 0], aggregated[:, 1]
