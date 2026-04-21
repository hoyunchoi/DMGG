import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * num_layers * out_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arg:
            x: [*, in_dim]

        Returns:
            gamma: [*, L, out_dim]
            beta: [*, L, out_dim]
        """
        prefix_shape = x.shape[:-1]

        # [*, in_dim] -> [*, 2 * L * out_dim]
        out = self.mlp(x)

        # [*, 2 * L * out_dim] -> [*, L, out_dim, 2]
        out = out.view(*prefix_shape, self.num_layers, self.out_dim, 2)

        # [*, L, out_dim], [*, L, out_dim]
        gamma, beta = out[..., 0], out[..., 1]
        return gamma, beta