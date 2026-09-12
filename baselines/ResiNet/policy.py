from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.optim import Optimizer

from baselines.ResiNet.fire_gnn import (
    DualFiltrationEdgeAggregator,
    FireGNN,
    FireGNNOutput,
)
from baselines.ResiNet.observation import ResiNetObs, ResiNetPygObs, ResiNetVecObs
from DMGG.RL.rewire_policy import RewirePolicy
from DMGG.scatter import ScatterCategorical, scatter_argmax, scatter_sum


class _UnusedFeaturesExtractor(BaseFeaturesExtractor):
    """Satisfy ActorCriticPolicy's constructor; graph extraction is custom."""

    def __init__(self, observation_space: spaces.Space):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ResiNetPolicy directly consumes ResiNetPygObs")


class _UnusedMlpExtractor(nn.Module):
    latent_dim_pi = 1
    latent_dim_vf = 1

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return features, features

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features


class ConditionedPointer(nn.Module):
    """Additive pointer score with a state-dependent scoring vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        context: torch.Tensor,
        condition: torch.Tensor,
        candidate: torch.Tensor,
        candidate_batch: torch.LongTensor,
    ) -> torch.Tensor:
        energy = torch.tanh(
            self.attention(torch.cat([context[candidate_batch], candidate], dim=-1))
        )
        return (condition[candidate_batch] * energy).sum(dim=-1)


@dataclass
class DirectedEdgeLayout:
    ptr: torch.LongTensor
    batch: torch.LongTensor
    canonical: torch.LongTensor
    reversed: torch.LongTensor

    @classmethod
    def from_action_obs(cls, obs: ResiNetPygObs) -> "DirectedEdgeLayout":
        return cls(
            ptr=obs.directed_edge_ptr,
            batch=obs.directed_edge_batch,
            canonical=obs.canonical_edge,
            reversed=obs.reversed_edge,
        )


class ResiNetPolicy(ActorCriticPolicy):
    """Target-conditioned ResiNet policy with separate actor and critic encoders."""

    actor_optimizer: Optimizer
    critic_optimizer: Optimizer

    # PyG message passing and scatter aggregation keep their dynamic eager
    # implementations. The fixed-shape policy/value heads use GraphPPO's
    # existing optional torch.compile path.
    compiled_nns: dict[str, dict[str, Any]] = {
        "actor_fire": {"dynamic": True, "fullgraph": False},
        "critic_fire": {"dynamic": True, "fullgraph": False},
        "first_pointer": {"dynamic": True, "fullgraph": False},
        "second_pointer": {"dynamic": True, "fullgraph": False},
        "value_net": {"dynamic": True, "fullgraph": False},
        "edge_aggregator": {"dynamic": True, "fullgraph": False},
    }

    compiled_fns: dict[str, dict[str, Any]] = {}
    amp_config: list[str] = [
        "actor_fire",
        "critic_fire",
        "edge_aggregator",
        "first_pointer",
        "second_pointer",
    ]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        hidden_dim: int = 64,
        num_layers: int = 5,
        node_dim: int = 15,
        filtration_order: int = 1,
        critic_learning_rate: float | None = None,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.filtration_order = filtration_order
        self.critic_learning_rate = critic_learning_rate

        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.setdefault("net_arch", [])
        kwargs.setdefault("ortho_init", False)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=_UnusedFeaturesExtractor,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _UnusedMlpExtractor()

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        hidden_dim = self.hidden_dim

        # Actor: FireGNN and both autoregressive pointer heads.
        self.actor_fire = FireGNN(
            self.node_dim, hidden_dim, self.num_layers, encode_edges=True
        )
        self.edge_aggregator = DualFiltrationEdgeAggregator(hidden_dim)
        self.sign_embedding = nn.Embedding(2, hidden_dim)
        self.actor_context = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.first_pointer = ConditionedPointer(hidden_dim)
        self.first_edge_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.second_context = nn.Linear(2 * hidden_dim, hidden_dim)
        self.second_pointer = ConditionedPointer(hidden_dim)

        # Critic: a fully separate FireGNN and target-magnitude branch.
        self.critic_fire = FireGNN(
            self.node_dim, hidden_dim, self.num_layers, encode_edges=False
        )
        self.critic_delta = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        actor_modules = [
            self.actor_fire,
            self.edge_aggregator,
            self.sign_embedding,
            self.actor_context,
            self.first_pointer,
            self.first_edge_transform,
            self.second_context,
            self.second_pointer,
        ]
        critic_modules = [self.critic_fire, self.critic_delta, self.value_net]
        actor_parameters = [
            parameter for module in actor_modules for parameter in module.parameters()
        ]
        critic_parameters = [
            parameter for module in critic_modules for parameter in module.parameters()
        ]
        assert {id(parameter) for parameter in actor_parameters}.isdisjoint(
            id(parameter) for parameter in critic_parameters
        )

        actor_learning_rate = lr_schedule(1.0)
        critic_learning_rate = (
            actor_learning_rate
            if self.critic_learning_rate is None
            else self.critic_learning_rate
        )
        optimizer_factory = cast(Callable[..., Optimizer], self.optimizer_class)
        self.actor_optimizer = optimizer_factory(
            actor_parameters, lr=actor_learning_rate, **self.optimizer_kwargs
        )
        self.critic_optimizer = optimizer_factory(
            critic_parameters, lr=critic_learning_rate, **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            node_dim=self.node_dim,
            filtration_order=self.filtration_order,
            critic_learning_rate=self.critic_learning_rate,
        )
        return data

    @staticmethod
    def _categorical(
        logits: torch.Tensor,
        batch: torch.LongTensor,
        ptr: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> ScatterCategorical:
        return ScatterCategorical(
            logits=logits.masked_fill(mask, -torch.inf),
            batch=batch,
            ptr=ptr,
        )

    @staticmethod
    def _select(
        distribution: ScatterCategorical,
        deterministic: bool,
    ) -> torch.LongTensor:
        if deterministic:
            return cast(
                torch.LongTensor,
                scatter_argmax(
                    distribution.logits,
                    distribution.batch,
                    dim=0,
                    dim_size=distribution.num_batch,
                ),
            )
        return distribution.sample()

    @staticmethod
    def _pack_mode_masks(
        mode0_mask: torch.BoolTensor,
        mode1_mask: torch.BoolTensor,
        layout: DirectedEdgeLayout,
    ) -> torch.BoolTensor:
        directed_mask = torch.empty(
            len(layout.batch), dtype=torch.bool, device=mode0_mask.device
        )
        directed_mask[layout.canonical] = mode0_mask
        directed_mask[layout.reversed] = mode1_mask
        return cast(torch.BoolTensor, directed_mask)

    def _second_edge_mask(
        self,
        edge1: torch.LongTensor,
        obs: ResiNetPygObs,
        layout: DirectedEdgeLayout,
    ) -> torch.BoolTensor:
        (
            mode0_invalid,
            mode1_invalid,
            mode0_ineffective,
            mode1_ineffective,
        ) = RewirePolicy.get_masks(edge1, obs.action)

        mode0_mask = mode0_invalid | mode0_ineffective
        mode1_mask = mode1_invalid | mode1_ineffective
        directed_mask = self._pack_mode_masks(
            cast(torch.BoolTensor, mode0_mask),
            cast(torch.BoolTensor, mode1_mask),
            layout,
        )

        no_effective_candidate = (
            scatter_sum(
                (~directed_mask).float(),
                layout.batch,
                dim=0,
                dim_size=obs.num_graphs,
            )
            == 0
        )
        structural_mask = self._pack_mode_masks(mode0_invalid, mode1_invalid, layout)
        return cast(
            torch.BoolTensor,
            torch.where(
                no_effective_candidate[layout.batch],
                structural_mask,
                directed_mask,
            ),
        )

    @staticmethod
    def _original_directed_edges(
        output: FireGNNOutput,
        obs: ResiNetPygObs,
    ) -> torch.Tensor:
        directed_edge_embedding = output.directed_edge_embedding
        assert directed_edge_embedding is not None
        return directed_edge_embedding[obs.action_to_original_fire_edge]

    def _actor_embeddings(
        self, obs: ResiNetPygObs, layout: DirectedEdgeLayout
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.actor_fire(
            obs.fire,
            obs.fire_edge_index,
            obs.view_to_graph,
            obs.view_valid,
            obs.num_graphs,
        )
        directed_edge_embedding = output.directed_edge_embedding
        assert directed_edge_embedding is not None
        num_directed_edges = len(layout.batch)

        first_edge_embedding, second_edge_embedding = self.edge_aggregator(
            directed_edge_embedding,
            output.view_embedding,
            obs.fire.bi_edge_batch,
            obs.fire_edge_to_action,
            num_directed_edges,
        )
        original_edge_embedding = self._original_directed_edges(output, obs)

        target_sign = (obs.action.glob_attr[:, 0] > 0).to(torch.int64)
        context = self.actor_context(
            torch.cat(
                [output.graph_embedding, self.sign_embedding(target_sign)], dim=-1
            )
        )
        return (
            context,
            first_edge_embedding,
            second_edge_embedding,
            original_edge_embedding,
        )

    def _actor(
        self,
        obs: ResiNetPygObs,
        deterministic: bool,
        actions: torch.LongTensor | None = None,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        layout = DirectedEdgeLayout.from_action_obs(obs)
        context, first_embedding, second_embedding, original_embedding = (
            self._actor_embeddings(obs, layout)
        )

        first_candidates = first_embedding[layout.canonical]
        first_logits = self.first_pointer(
            context,
            context,
            first_candidates,
            obs.action.edge_batch,
        )
        first_distribution = self._categorical(
            first_logits,
            obs.action.edge_batch,
            obs.action.edge_ptr,
            obs.first_edge_mask,
        )

        if actions is None:
            first = self._select(first_distribution, deterministic)
        else:
            first = cast(torch.LongTensor, actions[:, 0] + obs.action.edge_ptr[:-1])

        selected_original_first = original_embedding[layout.canonical[first]].float()
        second_context = self.second_context(
            torch.cat(
                [context, self.first_edge_transform(selected_original_first)], dim=-1
            )
        )
        second_logits = self.second_pointer(
            second_context,
            selected_original_first,
            second_embedding,
            layout.batch,
        )
        second_mask = self._second_edge_mask(first, obs, layout)
        second_distribution = self._categorical(
            second_logits, layout.batch, layout.ptr, second_mask
        )

        if actions is None:
            second = self._select(second_distribution, deterministic)
        else:
            second = cast(torch.LongTensor, actions[:, 1] + layout.ptr[:-1])

        local_first = first - obs.action.edge_ptr[:-1]
        local_second = second - layout.ptr[:-1]
        selected_actions = cast(
            torch.LongTensor, torch.stack([local_first, local_second], dim=-1)
        )
        log_prob = first_distribution.log_prob(first) + second_distribution.log_prob(
            second
        )
        entropy = first_distribution.entropy() + second_distribution.entropy()
        return selected_actions, log_prob, entropy

    def _critic(self, obs: ResiNetPygObs) -> torch.Tensor:
        output = self.critic_fire(
            obs.fire,
            obs.fire_edge_index,
            obs.view_to_graph,
            obs.view_valid,
            obs.num_graphs,
        )
        delta_embedding = self.critic_delta(obs.action.glob_attr[:, :1])
        return self.value_net(
            torch.cat([output.graph_embedding, delta_embedding], dim=-1)
        ).squeeze(-1)

    def obs_to_tensor(  # type: ignore[override]
        self, observation: ResiNetObs | ResiNetVecObs | ResiNetPygObs
    ) -> tuple[ResiNetPygObs, bool]:
        if isinstance(observation, ResiNetPygObs):
            return observation, True
        if isinstance(observation, list):
            return ResiNetPygObs.from_graph_vec_obs(observation, self.device), True
        return ResiNetPygObs.from_graph_obs(observation, self.device), False

    def forward(  # type: ignore[override]
        self,
        obs: ResiNetPygObs,
        deterministic: bool = False,
        only_action: bool = False,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        actions, log_prob, _ = self._actor(obs, deterministic)
        if only_action:
            empty = torch.empty(0, device=obs.device)
            return actions, empty, empty
        return actions, self._critic(obs), log_prob

    def evaluate_actions(  # type: ignore[override]
        self, obs: ResiNetPygObs, actions: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, log_prob, entropy = self._actor(obs, deterministic=False, actions=actions)
        return self._critic(obs), log_prob, entropy

    def predict_values(self, obs: ResiNetPygObs) -> torch.Tensor:  # type: ignore[override]
        return self._critic(obs)

    def _predict(  # type: ignore[override]
        self, observation: ResiNetPygObs, deterministic: bool = False
    ) -> torch.LongTensor:
        actions, _, _ = self.forward(
            observation, deterministic=deterministic, only_action=True
        )
        return actions

    def predict(  # type: ignore[override]
        self,
        observation: ResiNetObs | ResiNetVecObs,
        deterministic: bool = False,
    ) -> npt.NDArray[np.int64]:
        self.set_training_mode(False)
        obs, _ = self.obs_to_tensor(observation)
        with torch.no_grad():
            actions = self._predict(obs, deterministic)
        return actions.cpu().numpy()

    def extract_features(self, obs: ResiNetPygObs) -> FireGNNOutput:  # type: ignore[override]
        return self.actor_fire(
            obs.fire,
            obs.fire_edge_index,
            obs.view_to_graph,
            obs.view_valid,
            obs.num_graphs,
        )

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        raise NotImplementedError("ResiNet actions are sampled autoregressively")
