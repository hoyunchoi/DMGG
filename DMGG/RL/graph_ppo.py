from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.ppo.ppo import PPO
from torch import nn
from torch.amp.grad_scaler import GradScaler

from DMGG.env import GraphVecEnv, GraphVecObs, PygObs, RewireEnv
from DMGG.net import AutocastWrapper
from DMGG.RL.graph_rollout_buffer import GraphRolloutBuffer
from DMGG.RL.rewire_policy import RewirePolicy


class GraphPPO(PPO):
    _last_episode_starts: npt.NDArray[np.bool_]
    _last_obs: GraphVecObs
    env: GraphVecEnv
    policy: RewirePolicy  # type: ignore[override]
    rollout_buffer: GraphRolloutBuffer

    def __init__(
        self,
        *args,
        compile: bool = False,
        amp_dtype: str = "bf16",
        _init_setup_model: bool = True,
        **kwargs,
    ):
        super().__init__(*args, _init_setup_model=_init_setup_model, **kwargs)

        self._amp_dtype = amp_dtype
        self._is_compiled = compile

        if not _init_setup_model:
            # AMP and compile setup will be done in load()
            return

        self._check_buffer_size()
        self._init_amp()
        self._init_compile()

    def _check_buffer_size(self) -> None:
        """Check that the rollout buffer size is a multiple of the mini-batch size"""
        assert self.env is not None
        buffer_size = self.env.num_envs * self.n_steps
        assert (
            buffer_size % self.batch_size == 0
        ), "Rollout buffer size must be a multiple of the mini-batch size"

    def _init_amp(self) -> None:

        for target_name in self.policy.amp_config:
            target = getattr(self.policy, target_name)
            target = AutocastWrapper(target, dtype=self._amp_dtype, device=self.device)
            setattr(self.policy, target_name, target)

        self.grad_scaler = GradScaler(enabled=(self._amp_dtype == "fp16"))

    def _init_compile(self) -> None:
        if not self._is_compiled:
            return

        import torch._inductor.config

        # CUDA Graphs disable - GNN's dynamic shape and conflict
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.triton.cudagraph_trees = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        for target_name, config in (
            self.policy.compiled_nns | self.policy.compiled_fns
        ).items():
            target = getattr(self.policy, target_name)
            target = torch.compile(target, **config)
            setattr(self.policy, target_name, target)

    def set_env(  # type: ignore[override]
        self, env: RewireEnv | GraphVecEnv, force_reset: bool = True
    ) -> None:
        n_envs = self.n_envs if hasattr(self, "n_envs") else None

        env = cast(GraphVecEnv, self._wrap_env(env, self.verbose, True))  # type: ignore[arg-type]

        self.env = env  # type: ignore[assignment]
        self.n_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._check_buffer_size()

        self._vec_normalize_env = unwrap_vec_normalize(env)  # type:ignore[arg-type]

        if n_envs is not None and n_envs != self.n_envs:
            # If number of environments is different from the loaded environment, recreate the rollout buffer according to the new environment
            if self.verbose >= 1:
                print(f"Recreating rollout buffer: {n_envs} envs -> {self.n_envs} envs")

            self.rollout_buffer = GraphRolloutBuffer(  # type: ignore[assignment]
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                **self.rollout_buffer_kwargs,
            )

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None  # type:ignore[assignment]

    def collect_rollouts(  # type: ignore[override]
        self,
        env: GraphVecEnv,
        callback: BaseCallback,
        rollout_buffer: GraphRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                actions, values, log_probs = self.policy(
                    PygObs.from_graph_vec_obs(self._last_obs, device=self.device)
                )
            values = values.to(torch.float32)
            log_probs = log_probs.to(torch.float32)

            new_obs, rewards, dones, infos = env.step(actions.cpu().numpy())

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = PygObs.from_graph_obs(
                        infos[idx]["terminal_observation"], device=self.device
                    )

                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)
                    rewards[idx] += self.gamma * terminal_value.item()

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(
                PygObs.from_graph_vec_obs(new_obs, device=self.device)  # type: ignore[unbound-variable]
            )
        values = values.to(torch.float32)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)  # type: ignore[unbound-variable]

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        The following follows the structure of SB3 PPO.train(),
        but wraps only the loss calculation/backward part with AMP.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(
            self._current_progress_remaining
        )  # type:ignore[operator]

        # Optional: clip range for the value function
        if self.clip_range_vf is None:
            clip_range_vf = None
        else:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )  # type:ignore[operator]

        entropy_losses: list[torch.Tensor] = []
        pg_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        clip_fractions: list[torch.Tensor] = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs: list[torch.Tensor] = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                log_ratio = log_prob - rollout_data.old_log_prob
                ratio = torch.exp(log_ratio)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value function loss (clipped)
                if clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -(-log_prob).mean()
                else:
                    entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with torch.no_grad():
                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (log_ratio.exp() - 1 - log_ratio).mean()

                    # Logging
                    pg_losses.append(policy_loss)
                    clip_fractions.append(
                        torch.mean((torch.abs(ratio - 1) > clip_range).float())
                    )
                    value_losses.append(value_loss)
                    entropy_losses.append(entropy_loss)
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad(set_to_none=True)

                # AMP: scaled backward
                self.grad_scaler.scale(loss).backward()

                # Unscale before grad clipping
                self.grad_scaler.unscale_(self.policy.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                #! Optimizer step (scaled)
                self.grad_scaler.step(self.policy.optimizer)
                self.grad_scaler.update()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.clone().flatten().cpu().numpy(),
            self.rollout_buffer.returns.flatten(),
        )

        # Logs
        self.logger.record(
            "train/entropy_loss", torch.stack(entropy_losses).mean().item()
        )
        self.logger.record(
            "train/policy_gradient_loss", torch.stack(pg_losses).mean().item()
        )
        self.logger.record("train/value_loss", torch.stack(value_losses).mean().item())
        self.logger.record("train/approx_kl", torch.stack(approx_kl_divs).mean().item())  # type: ignore[unbound-variable]
        self.logger.record(
            "train/clip_fraction", torch.stack(clip_fractions).mean().item()
        )
        self.logger.record("train/loss", loss.item())  # type: ignore[unbound-variable]
        self.logger.record("train/explained_variance", explained_var)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _excluded_save_params(self) -> list[str]:
        """Add custom parameters to the excluded parameters, which prevent saving in pickle format"""
        return super()._excluded_save_params() + ["grad_scaler"]

    def save(self, result_dir: Path) -> None:  # type: ignore[override]
        # Save original policy if compiled
        compiled: dict[str, nn.Module] = {}
        if self._is_compiled:
            for target_name in self.policy.compiled_nns:
                target = getattr(self.policy, target_name)
                compiled[target_name] = target
                if hasattr(target, "_orig_mod"):
                    setattr(self.policy, target_name, target._orig_mod)

        # Temporarily store GraphPPO-specific state (AMP, compile)
        # This will be pickled along with the model
        self._graph_ppo_state = {
            "amp_dtype": self._amp_dtype,
            "grad_scaler_state_dict": self.grad_scaler.state_dict(),
            "compile": self._is_compiled,
        }

        super().save(result_dir / "model.zip")

        # Restore original model
        delattr(self, "_graph_ppo_state")
        if self._is_compiled:
            for target_name, compiled_target in compiled.items():
                setattr(self.policy, target_name, compiled_target)
        del compiled

    @classmethod
    def load(  # type: ignore[override]
        cls,
        result_dir: Path,
        env: RewireEnv | GraphVecEnv,
        device: torch.device | str = "cuda",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        compile: bool | None = None,
        **kwargs,
    ) -> Self:
        # Load model without environment as the action space could be different
        model = super().load(
            result_dir / "model.zip",
            None,
            device,
            custom_objects,
            print_system_info,
            force_reset,
            **kwargs,
        )

        # Set environment after loading model
        model.set_env(env, force_reset=True)

        assert hasattr(
            model, "_graph_ppo_state"
        ), f"Load model from {result_dir} does not have _graph_ppo_state"
        graph_ppo_state = model._graph_ppo_state
        delattr(model, "_graph_ppo_state")

        # Load AMP settings
        model._amp_dtype = graph_ppo_state["amp_dtype"]
        model._init_amp()

        # Load compile settings
        if compile is None:
            model._is_compiled = graph_ppo_state["compile"]
        else:
            compile = bool(compile)
            model._is_compiled = compile
            if compile != bool(graph_ppo_state["compile"]):
                print(
                    f"Compile setting is overridden by user-provided value: {compile} (from {graph_ppo_state['compile']})"
                )
        model._init_compile()

        return model
