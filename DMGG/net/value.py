"""
N: number of nodes
E: number of edges

B: number of graphs (batches)
BN: number of nodes in batched graph = N_1 + N_2 + ... + N_B
BE: number of edges in batched graph = E_1 + E_2 + ... + E_B

NF: number of node features
EF: number of edge features
GF: number of global features

L: number of layers
H: hidden dimension
"""


import torch
from torch import nn

from DMGG.net.film import FiLM
from DMGG.scatter import scatter_mean


class ValueNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.film = FiLM(1, hidden_dim, hidden_dim, num_layers=4)

        # Global feature pooled from node and edge processing
        self.node_linear1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.node_linear2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.edge_linear1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_linear2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        self.readout = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        glob_attr: torch.Tensor,
        node_batch: torch.LongTensor,
        edge_batch: torch.LongTensor,
    ):
        """
        Args:
            node_emb: [BN, H]
            edge_emb: [BE, H]
            glob_attr: [B, 2]
            node_batch: [BN, ]
            edge_batch: [BE, ]

        Return:
            value: [B, ]
        """
        # ------- Input processing -------
        num_batch = len(glob_attr)
        gap = glob_attr[:, 0].unsqueeze(-1)  # [B, 1]

        # FiLM, conditioned by glob attr
        gamma, beta = self.film(gap) # [B, L=4, H]

        # Global pooling from node and edge embeddings
        node_mean_pool = scatter_mean(  # [B, H]
            node_emb, node_batch, dim=0, dim_size=num_batch
        )
        edge_mean_pool = scatter_mean(  # [B, H]
            edge_emb, edge_batch, dim=0, dim_size=num_batch
        )

        # ------------ Node processing ------------
        glob_node_emb1 = self.node_linear1(node_mean_pool)  # [B, H]
        glob_node_emb1 = gamma[:, 0] * glob_node_emb1 + beta[:, 0]  # [B, H]

        glob_node_emb2 = self.node_linear2(glob_node_emb1)  # [B, H]
        glob_node_emb2 = gamma[:, 1] * glob_node_emb2 + beta[:, 1]  # [B, H]

        glob_node_emb = 0.5 * (glob_node_emb1 + glob_node_emb2)

        # ------------ Edge processing ------------
        glob_edge_emb1 = self.edge_linear1(edge_mean_pool)  # [B, H]
        glob_edge_emb1 = gamma[:, 2] * glob_edge_emb1 + beta[:, 2]  # [B, H]

        glob_edge_emb2 = self.edge_linear2(glob_edge_emb1)  # [B, H]
        glob_edge_emb2 = gamma[:, 3] * glob_edge_emb2 + beta[:, 3]  # [B, H]

        glob_edge_emb = 0.5 * (glob_edge_emb1 + glob_edge_emb2)

        # ------------ Attention-based processing ------------
        # Use node embeddings as query, edge embeddings as key and value
        # [B, 1, H] -> [B, 1, H]
        attn_out, _ = self.attention(
            node_mean_pool.unsqueeze(1),
            edge_mean_pool.unsqueeze(1),
            edge_mean_pool.unsqueeze(1),
            need_weights=False,
        )
        # [B, 1, H] -> [B, H]
        attn_out = attn_out.squeeze(1)

        # ------------ Final readout ------------
        # [B, 3H] -> [B, 1]
        value = self.readout(
            torch.cat([glob_node_emb, glob_edge_emb, attn_out], dim=-1)
        )

        return value.squeeze(-1)  # [B, ]
