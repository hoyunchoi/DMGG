from typing import NamedTuple, cast

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.preprocessing import get_action_dim

from baselines.ResiNet.observation import ResiNetObs, ResiNetPygObs
from DMGG.env import GraphSpace
from DMGG.RL.graph_rollout_buffer import GraphRolloutBuffer


class ResiNetRolloutBufferSamples(NamedTuple):
    observations: ResiNetPygObs
    actions: torch.LongTensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ResiNetRolloutBuffer(GraphRolloutBuffer):
    """Graph rollout buffer that batches TC-ResiNet filtration observations."""

    graphs: list[list[ResiNetObs]]  # type: ignore[reportIncompatibleVariableOverride]
    observations: list[ResiNetObs]  # type: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        buffer_size: int,
        observation_space: GraphSpace,
        action_space: gym.spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        batch_size: int = 1,
    ) -> None:
        # GraphRolloutBuffer always requests pinned host memory, which raises
        # when PyTorch was built with CUDA support but no CUDA driver is
        # available. Keep its storage layout while making CPU smoke tests and
        # inference work as well.
        self.buffer_size = buffer_size
        self.action_dim = get_action_dim(action_space)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.pos = 0
        self.full = False
        self.generator_ready = False

        pin_memory = self.device.type == "cuda" and torch.cuda.is_available()
        self._pin_returns = torch.empty((batch_size,), pin_memory=pin_memory)
        self._pin_advantages = torch.empty((batch_size,), pin_memory=pin_memory)
        self.reset()

    def _get_samples(  # type: ignore[override]
        self, batch_inds: np.ndarray
    ) -> ResiNetRolloutBufferSamples:
        self._pin_advantages.copy_(
            torch.from_numpy(self.advantages[batch_inds].flatten())
        )
        self._pin_returns.copy_(torch.from_numpy(self.returns[batch_inds].flatten()))

        graph_vec_obs = [self.observations[idx] for idx in batch_inds]
        observations = ResiNetPygObs.from_graph_vec_obs(
            graph_vec_obs, device=self.device
        )

        return ResiNetRolloutBufferSamples(
            observations=observations,
            actions=cast(torch.LongTensor, self.actions[batch_inds]),
            old_values=self.values[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=self._pin_advantages.to(device=self.device, non_blocking=True),
            returns=self._pin_returns.to(device=self.device, non_blocking=True),
        )
