from typing import Generator, NamedTuple, cast, overload

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.preprocessing import get_action_dim

from DMGG.env import GraphObs, GraphSpace, GraphVecObs, PygObs


class GraphRolloutBufferSamples(NamedTuple):
    observations: PygObs
    actions: torch.LongTensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class GraphRolloutBuffer(DictRolloutBuffer):
    graphs: list[list[GraphObs]]  # buffer * num_env
    observations: GraphVecObs
    actions: torch.LongTensor
    rewards: npt.NDArray[np.float32]
    returns: npt.NDArray[np.float32]
    episode_starts: npt.NDArray[np.bool_]
    values: torch.Tensor
    log_probs: torch.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: GraphSpace,
        action_space: gym.spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        batch_size: int = 1,
    ):
        self.buffer_size = buffer_size

        self.action_dim = get_action_dim(action_space)

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.device = torch.device(device)

        self.pos = 0
        self.full = False
        self.generator_ready = False

        # Pinned tensors for efficient GPU transfer
        self._pin_returns = torch.empty((batch_size,), pin_memory=True)
        self._pin_advantages = torch.empty((batch_size,), pin_memory=True)

        self.reset()

    def reset(self) -> None:
        self.graphs = [  # type:ignore[assignment]
            [None for _ in range(self.n_envs)] for _ in range(self.buffer_size)
        ]
        self.actions = cast(
            torch.LongTensor,
            torch.zeros(
                (self.buffer_size, self.n_envs, self.action_dim),
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.values = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.log_probs = torch.zeros(
            (self.buffer_size, self.n_envs), device=self.device
        )
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.generator_ready = False

        # Reset in BaseBuffer
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: np.ndarray
    ) -> None:
        # Convert to numpy
        values = self.values.clone().cpu().numpy()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values.clone().cpu().numpy().flatten()
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(
                    np.float32
                )
                next_values = values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + values

    def add(  # type: ignore[override]
        self,
        obs: GraphVecObs,
        action: torch.LongTensor,
        reward: npt.NDArray[np.float32],
        episode_start: npt.NDArray[np.bool_],
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        obs: [n_envs, obs_shape] for each key
        action: [n_envs, action_dim]
        reward: [n_envs, ]
        episode_start: [n_envs, ]
        value: [n_envs, ]
        log_prob: [n_envs, ]
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.graphs[self.pos] = obs

        self.actions[self.pos] = action.reshape((self.n_envs, self.action_dim))
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos].copy_(value.flatten())
        self.log_probs[self.pos].copy_(log_prob)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get( # type: ignore[override]
        self, batch_size: int | None = None
    ) -> Generator[GraphRolloutBufferSamples, None, None]:
        """Get samples from buffer using pre-allocated pinned tensors"""
        assert self.full, "Rollout buffer must be full before sampling"

        # Prepare the data
        if not self.generator_ready:
            self.observations = self.swap_and_flatten(self.graphs) # type: ignore[assignment]
            del self.graphs

            self.actions = cast(torch.LongTensor, self.swap_and_flatten(self.actions))
            self.values = self.swap_and_flatten(self.values)
            self.log_probs = self.swap_and_flatten(self.log_probs)
            self.advantages = self.swap_and_flatten(self.advantages)
            self.returns = self.swap_and_flatten(self.returns)
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        indices = np.random.permutation(self.buffer_size * self.n_envs)
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> GraphRolloutBufferSamples: # type: ignore[override]
        """Get samples using pinned tensors with non-blocking transfer"""

        # Overwrite batch data to pinned tensors
        self._pin_advantages.copy_(
            torch.from_numpy(self.advantages[batch_inds].flatten())
        )
        self._pin_returns.copy_(torch.from_numpy(self.returns[batch_inds].flatten()))

        graph_vec_obs = [self.observations[idx] for idx in batch_inds]
        observations = PygObs.from_graph_vec_obs(graph_vec_obs, device=self.device)

        # Get data from pinned tensors and transfer to GPU
        return GraphRolloutBufferSamples(
            observations=observations,
            actions= cast(torch.LongTensor, self.actions[batch_inds]),
            old_values=self.values[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=self._pin_advantages.to(device=self.device, non_blocking=True),
            returns=self._pin_returns.to(device=self.device, non_blocking=True),
        )

    @staticmethod
    @overload
    def swap_and_flatten(arr: list[list[GraphObs]]) -> GraphVecObs: ...

    @staticmethod
    @overload
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray: ...

    @staticmethod
    @overload
    def swap_and_flatten(arr: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    def swap_and_flatten(
        arr: list[list[GraphObs]] | np.ndarray | torch.Tensor,
    ) -> GraphVecObs | np.ndarray | torch.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        """
        if isinstance(arr, list):
            transposed = list(zip(*arr))
            flattend: GraphVecObs = []
            for x in transposed:
                flattend.extend(x)
            return flattend

        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
