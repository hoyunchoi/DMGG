from typing import Any, Callable, Iterable, Protocol, Sequence, cast, runtime_checkable

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from DMGG.env.graph_obs import GraphObs, GraphVecObs
from DMGG.env.graph_space import GraphSpace
from DMGG.env.rewire_env import RewireEnv

GraphVecEnvStepReturn = tuple[GraphVecObs, np.ndarray, np.ndarray, list[dict]]


@runtime_checkable
class GraphVecEnv(Protocol):
    num_envs: int
    observation_space: GraphSpace
    action_space: spaces.Space

    def step(self, actions: np.ndarray) -> GraphVecEnvStepReturn: ...

    def step_wait(self) -> GraphVecEnvStepReturn: ...

    def reset(self) -> GraphVecObs: ...

    def seed(self, seed: int | None = None) -> Sequence[int | None]: ...

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: int | Iterable[int] | None = None,
        **method_kwargs
    ) -> list[Any]: ...

    def close(self) -> None: ...

    def get_attr(self, attr_name: str) -> list[Any]: ...


class GraphDummyVecEnv(DummyVecEnv):
    """DummyVecEnv for GraphEnv"""

    observation_space: GraphSpace

    def __init__(self, env_fns: list[Callable[[], RewireEnv]]):
        self.envs: list[RewireEnv] = [_patch_env(fn()) for fn in env_fns]  # type: ignore[assignment]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

        # Graph observations have variable numbers of nodes/edges per env.
        # Keep per-env observations as-is and stack them with `_stack_obs`.
        self.buf_obs: GraphVecObs = [None for _ in range(self.num_envs)]  # type: ignore[assignment]

        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata

    def _save_obs(self, env_idx: int, obs: GraphObs) -> None:  # type: ignore[override]
        self.buf_obs[env_idx] = obs

    def _obs_from_buf(self) -> GraphVecObs:  # type: ignore[override]
        return _stack_obs(self.buf_obs)

    # -------------- Type overrides --------------
    def step(self, actions: np.ndarray) -> GraphVecEnvStepReturn:  # type: ignore[override]
        return cast(GraphVecEnvStepReturn, super().step(actions))

    def step_wait(self) -> GraphVecEnvStepReturn:  # type: ignore[override]
        return cast(GraphVecEnvStepReturn, super().step_wait())

    def reset(self) -> GraphVecObs:  # type: ignore[override]
        return cast(GraphVecObs, super().reset())


class GraphSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv for GraphEnv"""

    observation_space: GraphSpace


    def step_wait(self) -> GraphVecEnvStepReturn:  # type: ignore[override]
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]

        return _stack_obs(obs), np.stack(rews), np.stack(dones), list(infos)

    def reset(self) -> GraphVecObs:  # type: ignore[override]
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs)

    # -------------- Type overrides --------------
    def __init__(self, env_fns: list[Callable[[], RewireEnv]], start_method: str | None = None) -> None:
        super().__init__(env_fns, start_method) # type: ignore[arg-type]

    def step(self, actions: np.ndarray) -> GraphVecEnvStepReturn:  # type: ignore[override]
        step_return = super().step(actions)
        return cast(GraphVecEnvStepReturn, step_return)


def _stack_obs(obs_list: list[GraphObs] | tuple[GraphObs]) -> GraphVecObs:
    return list(obs_list)
