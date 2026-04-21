import torch
from torch import nn

from DMGG.net.film import FiLM


class EdgeScorer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edge_emb: torch.Tensor,
        glob_emb: torch.Tensor,
        edge_batch: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            edge_emb: [BE, H]
            glob_emb: [B, H]
            edge_batch: [BE, ]
        Return:
            edge_score: [BE, 1]
        """
        # [BE, H] + [BE, H] -> [BE, 2H]
        edge_score = torch.cat([edge_emb, glob_emb[edge_batch]], dim=-1)

        # [BE, 2H] -> [BE, 1]
        return self.net(edge_score)


class ConditionalEdgeScorer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Scale attention score by sqrt(H) for stability
        self.attention_scale = 1.0 / (hidden_dim**0.5)

        # Query and key projectors
        self.query_projector = nn.Linear(2 * hidden_dim, hidden_dim)
        self.key_projector = nn.Linear(hidden_dim, hidden_dim)

        # Bias scorer: Does Edge2 look good independently?
        self.bias_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edge1_idx: torch.LongTensor,
        edge_emb: torch.Tensor,
        glob_emb: torch.Tensor,
        edge_batch: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            edge1_idx: [B, ]
            edge_emb: [BE, H]
            glob_emb: [B, H]
            edge_batch: [BE, ]
        Return:
            edge2_score: [BE, 1]
        """
        # Query: [B, H], [B, H] -> [B, 2H] -> [B, H]
        edge1_emb = edge_emb[edge1_idx]  # [B, H]
        query = self.query_projector(torch.cat([edge1_emb, glob_emb], dim=-1))

        # Key: [BE, H] -> [BE, H]
        key = self.key_projector(edge_emb)

        # Attention Score: [BE, H] * [BE, H] -> [BE, 1]
        attention_score = (query[edge_batch] * key).sum(dim=-1, keepdim=True)
        attention_score = self.attention_scale * attention_score

        # Bias Score: [BE, H] -> [BE, 1]
        bias_score = self.bias_scorer(edge_emb)

        return attention_score + bias_score


class ModeScorer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.film = FiLM(hidden_dim, hidden_dim, out_dim=hidden_dim, num_layers=1)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, edge1_emb: torch.Tensor, edge2_emb: torch.Tensor, glob_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edge1_emb: [B, H]
            edge2_emb: [B, H]
            glob_emb: [B, H]
        Return:
            mode_score: [B, 1]
        """
        # Feature Engineering: [B, H]
        edge_emb_sum = edge1_emb + edge2_emb
        edge_emb_diff = torch.abs(edge1_emb - edge2_emb)
        edge_emb_prod = edge1_emb * edge2_emb  # Added: Interaction

        # [B, 3H] -> [B, H]
        mode_score = self.scorer(
            torch.cat([edge_emb_sum, edge_emb_diff, edge_emb_prod], dim=-1)
        )

        # FiLM condition by glob_emb
        gamma, beta = self.film(glob_emb)  # [B, L=1, H]
        mode_score = gamma[:, 0] * mode_score + beta[:, 0]

        # [B, H] -> [B, 1] with squeezed values of [0, 1]
        return self.out(mode_score)
