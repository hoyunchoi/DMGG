"""
N: number of nodes
E: number of edges

B: number of graphs (batches)
BN: number of nodes in batched graph = N_1 + N_2 + ... + N_B
BE: number of edges in batched graph = E_1 + E_2 + ... + E_B
2BE: number of bidirectional edges in batched graph = 2*BE

NF: number of node features
EF: number of edge features
GF: number of global features

L: number of layers
H: hidden dimension
"""

import torch
import torch_geometric.nn as gnn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from DMGG.env.graph_space import GraphSpace
from DMGG.env.pyg_obs import PygObs

from DMGG.net.film import FiLM


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: GraphSpace, num_layers: int, hidden_dim: int):
        super().__init__(observation_space, features_dim=1)  # dummy
        self.hidden_dim = hidden_dim

        # ------- Encoders -------
        # Input global attribute will be 0 or 1 (sign of gap)
        self.glob_encoder = nn.Embedding(2, hidden_dim)

        # Node attribute and global embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # ------- GNN Layers with FiLM -------
        # FiLM conditioned by global embedding
        self.film = FiLM(
            hidden_dim, hidden_dim, out_dim=hidden_dim, num_layers=num_layers
        )

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = gnn.GINConv(
                nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ),
                train_eps=True,
            )
            self.gnn_layers.append(conv)

        # ------- Readouts -------
        self.node_readout = nn.Sequential(
            nn.Linear((num_layers + 1) * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.edge_readout = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, obs: PygObs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ------- Input processing -------
        node_attr = obs.node_attr
        edge_index = obs.edge_list.T  # [2, BE]
        bi_edge_index = obs.bi_edge_list.T  # [2, 2BE]
        node_batch = obs.node_batch

        # --------- Encodings ---------
        # Change global attribute to signed gap
        # [-1, 1] -> [0, 1]
        gap = obs.glob_attr[:, 0]  # [B, ]
        signed_gap = (0.5 * (torch.sign(gap) + 1.0)).to(torch.int64)

        # Global embedding: [B, ] -> [B, H]
        glob_emb = self.glob_encoder(signed_gap)

        # Node encoding: [BN, 1 + H] -> [BN, H]
        node_emb = self.node_encoder(
            torch.cat([node_attr, glob_emb[node_batch]], dim=-1)
        )

        # ------- FiLM parameters from global attribute -------
        gamma, beta = self.film(glob_emb)  # [B, L=2, H]

        # ------- GNN with FiLM -------
        node_embs: list[torch.Tensor] = [node_emb]  # (L+1) * [BN, H]
        for i, conv in enumerate(self.gnn_layers):
            # GIN convolution
            node_emb = conv(node_emb, bi_edge_index)  # [BN, H]

            # FiLM
            node_emb = gamma[node_batch, i] * node_emb + beta[node_batch, i]

            node_embs.append(node_emb)

        # ------- Readouts -------
        # (L+1) * [BN, H] -> [BN, (L+1) * H] -> [BN, H]
        node_emb = self.node_readout(torch.cat(node_embs, dim=-1))

        # Edge embedding from node: [BE, H]
        row, col = edge_index
        node1_emb = node_emb[row]  # [BE, H]
        node2_emb = node_emb[col]  # [BE, H]

        # Edge embedding from node embeddings: 3 * [BE, H] -> [BE, 3H]
        edge_emb = torch.cat(
            [
                node1_emb + node2_emb,
                torch.abs(node1_emb - node2_emb),
                node1_emb * node2_emb,
            ],
            dim=-1,
        )
        # [BE, 3H] -> [BE, H]
        edge_emb = self.edge_readout(edge_emb)

        return node_emb, edge_emb, glob_emb
