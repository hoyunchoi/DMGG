from typing import Any, cast

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from baselines.ResiNet.observation import ResiNetObs
from baselines.ResiNet.preprocessing import build_resinet_obs
from DMGG.env import EvalRewireEnv, GraphObs, RewireEnv


def _configure_resinet_action_space(env: RewireEnv, filtration_order: int) -> None:
    if filtration_order != 1:
        raise ValueError("TC-ResiNet-sign currently requires filtration_order=1")
    base_action_space = cast(spaces.MultiDiscrete, env.action_space)
    num_edges_max = int(base_action_space.nvec[0])
    env.action_space = spaces.MultiDiscrete(
        np.array([num_edges_max, 2 * num_edges_max], dtype=np.int64)
    )


def _build_resinet_obs(base_obs: GraphObs) -> ResiNetObs:
    action_obs = GraphObs(
        node_attr=base_obs.node_attr.copy(),
        edge_list=base_obs.edge_list.copy(),
        glob_attr=base_obs.glob_attr.copy(),
    )
    return build_resinet_obs(action_obs)


def _to_rewire_action(
    action: npt.NDArray[np.int64], num_edges: int
) -> npt.NDArray[np.int64]:
    first_edge, directed_second_edge = action
    return np.array(
        [
            first_edge,
            directed_second_edge % num_edges,
            directed_second_edge // num_edges,
        ],
        dtype=np.int64,
    )


class ResiNetRewireEnv(RewireEnv):
    """DMGG rewiring environment with TC-ResiNet-sign observations/actions."""

    obs: ResiNetObs  # type: ignore

    def __init__(self, filtration_order: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _configure_resinet_action_space(self, filtration_order)

    def get_obs(self, rho: np.float32) -> ResiNetObs:  # type: ignore[override]
        return _build_resinet_obs(super().get_obs(rho))

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ResiNetObs, dict[str, Any]]:
        return cast(
            tuple[ResiNetObs, dict[str, Any]],
            super().reset(seed=seed, options=options),
        )

    def step(  # type: ignore[override]
        self, action: npt.NDArray[np.int64]
    ) -> tuple[ResiNetObs, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(
            _to_rewire_action(action, self.num_edges)
        )
        return cast(ResiNetObs, obs), reward, terminated, truncated, info


class ResiNetEvalRewireEnv(EvalRewireEnv):
    """EvalRewireEnv with TC-ResiNet-sign observations and actions."""

    obs: ResiNetObs  # type: ignore

    def __init__(self, filtration_order: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _configure_resinet_action_space(self, filtration_order)

    def get_obs(self, rho: np.float32) -> ResiNetObs:  # type: ignore[override]
        return _build_resinet_obs(super().get_obs(rho))

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ResiNetObs, dict[str, Any]]:
        return cast(
            tuple[ResiNetObs, dict[str, Any]],
            super().reset(seed=seed, options=options),
        )

    def step(  # type: ignore[override]
        self, action: npt.NDArray[np.int64]
    ) -> tuple[ResiNetObs, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(
            _to_rewire_action(action, self.num_edges)
        )
        return cast(ResiNetObs, obs), reward, terminated, truncated, info
