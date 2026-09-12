from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import (
    explained_variance,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.ppo import PPO
from torch.optim import Optimizer

from baselines.ResiNet.observation import ResiNetPygObs, ResiNetVecObs
from baselines.ResiNet.rollout_buffer import ResiNetRolloutBuffer
from DMGG.env import GraphVecEnv, RewireEnv
from DMGG.RL.graph_ppo import GraphPPO
from DMGG.RL.graph_rollout_buffer import GraphRolloutBuffer


def normalize_advantages(
    advantages: torch.Tensor, enabled: bool = True
) -> torch.Tensor:
    """Apply the same per-minibatch advantage normalization as SB3 PPO."""
    if enabled and len(advantages) > 1:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def dual_clip_surrogate(
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    clip_range: float,
    dual_clip_param: float = 10.0,
) -> torch.Tensor:
    """Per-sample dual-clipped PPO surrogate used by ResiNet."""
    surrogate_unclipped = advantages * ratio
    surrogate_clipped = advantages * torch.clamp(
        ratio, 1.0 - clip_range, 1.0 + clip_range
    )
    ppo_surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)
    negative_advantage_surrogate = torch.maximum(
        ppo_surrogate, dual_clip_param * advantages
    )
    return torch.where(advantages < 0.0, negative_advantage_surrogate, ppo_surrogate)


def dual_clip_policy_loss(
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    clip_range: float,
    dual_clip_param: float = 10.0,
) -> torch.Tensor:
    return -dual_clip_surrogate(advantages, ratio, clip_range, dual_clip_param).mean()


def conservative_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range_vf: float | None,
) -> torch.Tensor:
    """Original PPO conservative value loss: max of clipped/unclipped errors."""
    value_loss_unclipped = torch.square(values - returns)
    if clip_range_vf is None:
        return value_loss_unclipped.mean()

    clipped_values = old_values + torch.clamp(
        values - old_values, -clip_range_vf, clip_range_vf
    )
    value_loss_clipped = torch.square(clipped_values - returns)
    return torch.maximum(value_loss_unclipped, value_loss_clipped).mean()


def _optimizer_parameters(optimizer: Optimizer) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


class ResiNetPPO(GraphPPO):
    """SB3-compatible PPO with ResiNet's dual clip and split optimizers."""

    _last_episode_starts: npt.NDArray[np.bool_]
    _last_obs: ResiNetVecObs  # type: ignore[reportIncompatibleVariableOverride]
    policy: ResiNetPolicy  # type: ignore[reportIncompatibleVariableOverride]
    rollout_buffer: ResiNetRolloutBuffer  # type: ignore[reportIncompatibleVariableOverride]
    clip_range: float
    clip_range_vf: float | None

    def __init__(
        self,
        *args: Any,
        critic_learning_rate: float = 7e-4,
        ent_coef: float = 0.01,
        dual_clip_param: float = 10.0,
        **kwargs: Any,
    ) -> None:
        if dual_clip_param <= 1.0:
            raise ValueError("dual_clip_param must be greater than 1")

        self.critic_learning_rate = critic_learning_rate
        self.dual_clip_param = dual_clip_param

        policy_kwargs = dict(kwargs.get("policy_kwargs") or {})
        policy_kwargs.setdefault("critic_learning_rate", self.critic_learning_rate)
        kwargs["policy_kwargs"] = policy_kwargs

        super().__init__(
            *args,
            ent_coef=ent_coef,
            **kwargs,
        )

    def _setup_model(self) -> None:
        OnPolicyAlgorithm._setup_model(self)
        if self.clip_range_vf is not None:
            assert self.clip_range_vf > 0, (
                "`clip_range_vf` must be positive, "
                "pass `None` to deactivate vf clipping"
            )

    def set_env(self, env: RewireEnv | GraphVecEnv, force_reset: bool = True) -> None:
        old_num_envs = self.n_envs if hasattr(self, "n_envs") else None

        wrapped_env = cast(
            GraphVecEnv,
            self._wrap_env(env, self.verbose, True),  # type: ignore[arg-type]
        )
        self.env = wrapped_env
        self.n_envs = wrapped_env.num_envs
        self.observation_space = wrapped_env.observation_space
        self.action_space = wrapped_env.action_space
        self._check_buffer_size()
        self._vec_normalize_env = unwrap_vec_normalize(wrapped_env)  # type: ignore[arg-type]

        if old_num_envs is not None and old_num_envs != self.n_envs:
            if self.verbose >= 1:
                print(
                    "Recreating rollout buffer: "
                    f"{old_num_envs} envs -> {self.n_envs} envs"
                )
            buffer_factory = cast(type[ResiNetRolloutBuffer], self.rollout_buffer_class)
            self.rollout_buffer = buffer_factory(  # type: ignore[reportIncompatibleVariableOverride]
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                **self.rollout_buffer_kwargs,
            )

        if force_reset:
            self._last_obs = None  # type: ignore[assignment]

    def collect_rollouts(
        self,
        env: GraphVecEnv,
        callback: BaseCallback,
        rollout_buffer: GraphRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        resinet_rollout_buffer = cast(ResiNetRolloutBuffer, rollout_buffer)
        self.policy.set_training_mode(False)

        n_steps = 0
        resinet_rollout_buffer.reset()
        callback.on_rollout_start()
        new_obs = self._last_obs
        dones = np.zeros(env.num_envs, dtype=np.bool_)

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                tensor_obs = ResiNetPygObs.from_graph_vec_obs(
                    self._last_obs, device=self.device
                )
                actions, values, log_probs = self.policy(tensor_obs)
            values = values.to(torch.float32)
            log_probs = log_probs.to(torch.float32)

            graph_obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            new_obs = cast(ResiNetVecObs, graph_obs)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            for index, done in enumerate(dones):
                if (
                    done
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = ResiNetPygObs.from_graph_obs(
                        infos[index]["terminal_observation"], device=self.device
                    )
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)
                    rewards[index] += self.gamma * terminal_value.item()

            resinet_rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[reportIncompatibleVariableOverride]
            self._last_episode_starts = dones

        with torch.no_grad():
            last_obs = ResiNetPygObs.from_graph_vec_obs(new_obs, device=self.device)
            values = self.policy.predict_values(last_obs)
        values = values.to(torch.float32)

        resinet_rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)

        actor_optimizer = self.policy.actor_optimizer
        critic_optimizer = self.policy.critic_optimizer
        actor_learning_rate = float(self.learning_rate)  # type: ignore[arg-type]
        critic_learning_rate = self.critic_learning_rate
        entropy_coefficient = self.ent_coef
        update_learning_rate(actor_optimizer, actor_learning_rate)
        update_learning_rate(critic_optimizer, critic_learning_rate)

        clip_range = self.clip_range
        clip_range_vf = self.clip_range_vf

        actor_parameters = _optimizer_parameters(actor_optimizer)
        critic_parameters = _optimizer_parameters(critic_optimizer)

        entropy_losses: list[torch.Tensor] = []
        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        clip_fractions: list[torch.Tensor] = []
        dual_clip_fractions: list[torch.Tensor] = []
        approximate_kls: list[torch.Tensor] = []
        actor_grad_norms: list[torch.Tensor] = []
        critic_grad_norms: list[torch.Tensor] = []
        last_loss = torch.zeros((), device=self.device)

        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions
                )
                if entropy is None:
                    raise RuntimeError(
                        "ResiNetPolicy must return conditional autoregressive entropy"
                    )

                values = values.flatten().to(torch.float32)
                log_prob = log_prob.flatten().to(torch.float32)
                entropy = entropy.flatten().to(torch.float32)
                advantages = normalize_advantages(
                    rollout_data.advantages, self.normalize_advantage
                )

                log_ratio = log_prob - rollout_data.old_log_prob
                ratio = torch.exp(log_ratio)
                surrogate = dual_clip_surrogate(
                    advantages,
                    ratio,
                    clip_range,
                    self.dual_clip_param,
                )
                policy_loss = -surrogate.mean()
                value_loss = conservative_value_loss(
                    values,
                    rollout_data.old_values,
                    rollout_data.returns,
                    clip_range_vf,
                )
                entropy_loss = -entropy.mean()

                actor_loss = policy_loss + entropy_coefficient * entropy_loss
                critic_loss = self.vf_coef * value_loss
                loss = actor_loss + critic_loss
                last_loss = loss.detach()

                with torch.no_grad():
                    standard_surrogate = torch.minimum(
                        advantages * ratio,
                        advantages
                        * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range),
                    )
                    dual_clip_active = (advantages < 0.0) & (
                        standard_surrogate < self.dual_clip_param * advantages
                    )
                    approximate_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean()

                    policy_losses.append(policy_loss.detach())
                    value_losses.append(value_loss.detach())
                    entropy_losses.append(entropy_loss.detach())
                    clip_fractions.append(
                        (torch.abs(ratio - 1.0) > clip_range).float().mean()
                    )
                    dual_clip_fractions.append(dual_clip_active.float().mean())
                    approximate_kls.append(approximate_kl)

                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(actor_optimizer)
                self.grad_scaler.unscale_(critic_optimizer)
                actor_grad_norms.append(
                    torch.as_tensor(
                        torch.nn.utils.clip_grad_norm_(
                            actor_parameters, self.max_grad_norm
                        )
                    ).detach()
                )
                critic_grad_norms.append(
                    torch.as_tensor(
                        torch.nn.utils.clip_grad_norm_(
                            critic_parameters, self.max_grad_norm
                        )
                    ).detach()
                )
                self.grad_scaler.step(actor_optimizer)
                self.grad_scaler.step(critic_optimizer)
                self.grad_scaler.update()

            self._n_updates += 1

        explained_var = explained_variance(
            self.rollout_buffer.values.clone().flatten().cpu().numpy(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/learning_rate", actor_learning_rate)
        self.logger.record("train/critic_learning_rate", critic_learning_rate)
        self.logger.record("train/ent_coef", entropy_coefficient)
        self.logger.record(
            "train/entropy_loss", torch.stack(entropy_losses).mean().item()
        )
        self.logger.record(
            "train/policy_gradient_loss",
            torch.stack(policy_losses).mean().item(),
        )
        self.logger.record("train/value_loss", torch.stack(value_losses).mean().item())
        self.logger.record(
            "train/approx_kl", torch.stack(approximate_kls).mean().item()
        )
        self.logger.record(
            "train/clip_fraction", torch.stack(clip_fractions).mean().item()
        )
        self.logger.record(
            "train/dual_clip_fraction",
            torch.stack(dual_clip_fractions).mean().item(),
        )
        self.logger.record(
            "train/actor_grad_norm",
            torch.stack(actor_grad_norms).mean().item(),
        )
        self.logger.record(
            "train/critic_grad_norm",
            torch.stack(critic_grad_norms).mean().item(),
        )
        self.logger.record("train/loss", last_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return [
            "policy",
            "policy.actor_optimizer",
            "policy.critic_optimizer",
        ], []

    @classmethod
    def load(
        cls,
        result_dir: Path,
        env: RewireEnv | GraphVecEnv,
        device: torch.device | str = "cuda",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        compile: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        model = cast(
            Self,
            PPO.load.__func__(  # type: ignore[attr-defined]
                cls,
                result_dir / "model.zip",
                env=env,
                device=device,
                custom_objects=custom_objects,
                print_system_info=print_system_info,
                force_reset=force_reset,
                **kwargs,
            ),
        )

        if not hasattr(model, "_graph_ppo_state"):
            raise ValueError(f"{result_dir} is not a GraphPPO-compatible checkpoint")
        graph_ppo_state = model._graph_ppo_state
        delattr(model, "_graph_ppo_state")

        model._amp_dtype = graph_ppo_state["amp_dtype"]
        model._init_amp()
        model.grad_scaler.load_state_dict(
            graph_ppo_state["grad_scaler_state_dict"]  # type: ignore
        )

        saved_compile = bool(graph_ppo_state["compile"])
        model._is_compiled = saved_compile if compile is None else bool(compile)
        if compile is not None and bool(compile) != saved_compile:
            print(
                "Compile setting is overridden by user-provided value: "
                f"{bool(compile)} (from {saved_compile})"
            )
        model._init_compile()
        return model
