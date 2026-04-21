from typing import Self, cast

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from DMGG.env import GraphVecEnv, GraphVecObs
from DMGG.env.graph_vec_env import GraphVecEnvStepReturn


class GraphVecNormalize(VecNormalize):
    """
    VecNormalize variant for graph observations.

    SB3 VecNormalize assumes observations are np.ndarray or dict[str, np.ndarray],
    but GraphVecEnv returns GraphVecObs (list-like). We only keep reward
    normalization and bypass observation normalization/assertions.
    """

    venv: GraphVecEnv

    def reset(self) -> GraphVecObs: # type: ignore[override]
        obs = self.venv.reset()
        self.old_obs = obs  # type: ignore[assignment]
        self.returns = np.zeros(self.num_envs)

        return obs

    def step_wait(self) -> GraphVecEnvStepReturn: # type: ignore[override]
        obs, rewards, dones, infos = self.venv.step_wait()
        self.old_obs = obs  # type: ignore[assignment]
        self.old_reward = rewards

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    @classmethod
    def load(cls, load_path: str, venv: VecEnv) -> Self: # type: ignore[override]
        return cast(Self, super().load(load_path, venv))
