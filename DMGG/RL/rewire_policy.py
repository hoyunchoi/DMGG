from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from DMGG.env import GraphObs, GraphSpace, GraphVecObs, PygObs
from DMGG.net import (
    ConditionalEdgeScorer,
    EdgeScorer,
    GraphFeaturesExtractor,
    ModeScorer,
    ValueNet,
)
from DMGG.scatter import (
    ScatterBernoulli,
    ScatterCategorical,
    scatter_argmax,
    scatter_sum,
)


class RewirePolicy(ActorCriticPolicy):
    observation_space: GraphSpace

    # NNs
    features_extractor: GraphFeaturesExtractor
    edge_scorer: EdgeScorer
    conditional_edge_scorer: ConditionalEdgeScorer
    mode_scorer: ModeScorer
    value_net: ValueNet

    # torch.compile configuration
    compiled_nns: dict[str, dict[str, Any]] = {
        # Do not compile scatter operations
        "features_extractor": {"disable": True, "dynamic": True, "fullgraph": False},
        "value_net": {"disable": True, "dynamic": True, "fullgraph": False},
        # Simple neural networks with dynamic batch size
        "edge_scorer": {"mode": "default", "dynamic": True, "fullgraph": True},
        "conditional_edge_scorer": {
            "mode": "default",
            "dynamic": True,
            "fullgraph": True,
        },
        "mode_scorer": {"mode": "default", "dynamic": True, "fullgraph": True},
    }
    compiled_fns: dict[str, dict[str, Any]] = {
        "get_masks": {"mode": "default", "dynamic": True, "fullgraph": True},
        "get_bernoulli_distribution": {
            "mode": "default",
            "dynamic": True,
            "fullgraph": True,
        },
    }

    # AMP configuration
    amp_config: list[str] = [
        "features_extractor",
        "edge_scorer",
        "conditional_edge_scorer",
        "mode_scorer",
        "value_net",
    ]

    # ---- Get conditional action logits and probability ----
    @staticmethod
    def get_masks(
        edge1_idx: torch.LongTensor, obs: PygObs
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Get masks for invalid and ineffective rewiring when edge1 is fixed

        mode0 : (n1, n2), (n3, n4) -> (n1, n3), (n2, n4)
        mode1 : (n1, n2), (n3, n4) -> (n1, n4), (n2, n3)
        invalid: self-loop, pointless rewire, multi-edge
        ineffective: assortativity contribution doesn't change

        Args:
            edge1_idx: [B, ], global index of edge1, including graph offset
            obs: PygObs

        Returns:
            is_mode0_invalid: [BE, ]. True if mode0 rewiring is invalid.

            is_mode1_invalid: [BE, ]. True if mode1 rewiring is invalid.

            is_mode0_ineffective: [BE, ]. True if mode0 rewiring is ineffective.

            is_mode1_ineffective: [BE, ]. True if mode1 rewiring is ineffective.
        """
        # Nodes in edge1 (node1, node2)
        node1 = obs.edge_list[edge1_idx, 0]  # [B, ]
        node2 = obs.edge_list[edge1_idx, 1]  # [B, ]

        # Nodes in candidate of edge2 (node3, node4)
        node3 = obs.edge_list[..., 0]  # [BE, ]
        node4 = obs.edge_list[..., 1]  # [BE, ]

        # -------------- Included mask: self-loop or pointless rewire --------------
        batched_node1 = node1[obs.edge_batch]  # [BE, ]
        batched_node2 = node2[obs.edge_batch]  # [BE, ]
        node3_is_node1 = node3 == batched_node1  # [BE, ]
        node4_is_node1 = node4 == batched_node1  # [BE, ]
        node3_is_node2 = node3 == batched_node2  # [BE, ]
        node4_is_node2 = node4 == batched_node2  # [BE, ]

        # Mask edges who contains node1 or node2,
        # which leads to self-loop or pointless rewire
        included_mask = (  # [BE, ]
            node3_is_node1 | node3_is_node2 | node4_is_node1 | node4_is_node2
        )

        # -------------- Multi-edge mask --------------
        src, dst = obs.bi_edge_list.T  # [2, 2BE] -> [2BE, ], [2BE, ]
        is_source_node1 = src == node1[obs.bi_edge_batch]  # [2BE, ]
        is_source_node2 = src == node2[obs.bi_edge_batch]  # [2BE, ]

        # is_node1_neighbor[i] = True if i is neighbor of node1
        is_node1_neighbor = torch.zeros(
            obs.total_num_nodes, device=obs.device, dtype=torch.bool
        )
        is_node1_neighbor[dst[is_source_node1]] = True

        # is_node2_neighbor[i] = True if i is neighbor of node2
        is_node2_neighbor = torch.zeros_like(is_node1_neighbor)
        is_node2_neighbor[dst[is_source_node2]] = True

        # Mode 0 forbidden: (node1, node3) or (node2, node4) exists.
        # i.e., node3 is neighbor of node1 or node4 is neighbor of node2
        is_mode0_invalid = (
            included_mask | is_node1_neighbor[node3] | is_node2_neighbor[node4]
        )

        # Mode 1 forbidden: (node1, node4) or (node2, node3) exists.
        # i.e., node4 is neighbor of node1 or node3 is neighbor of node2
        is_mode1_invalid = (
            included_mask | is_node1_neighbor[node4] | is_node2_neighbor[node3]
        )

        # -------------- Assortativity mask --------------
        # normalized degree is provided as node attribute
        degrees = obs.node_attr.squeeze()  # [BN, ]

        # Degrees of edge1 (node1, node2) and edge2 (node3, node4)
        degree1 = degrees[batched_node1]  # [BE, ]
        degree2 = degrees[batched_node2]  # [BE, ]
        degree3 = degrees[node3]  # [BE, ]
        degree4 = degrees[node4]  # [BE, ]

        # current assortativity contribution: d1*d2 + d3*d4
        current_contrib = degree1 * degree2 + degree3 * degree4

        # Mode0 assortativity contribution: d1*d3 + d2*d4
        mode0_contrib = degree1 * degree3 + degree2 * degree4
        is_mode0_ineffective = current_contrib == mode0_contrib

        # Mode1 assortativity contribution: d1*d4 + d2*d3
        mode1_contrib = degree1 * degree4 + degree2 * degree3
        is_mode1_ineffective = current_contrib == mode1_contrib

        return (
            cast(torch.BoolTensor, is_mode0_invalid),
            cast(torch.BoolTensor, is_mode1_invalid),
            cast(torch.BoolTensor, is_mode0_ineffective),
            cast(torch.BoolTensor, is_mode1_ineffective),
        )

    def compute_edge1_logits(
        self,
        edge_emb: torch.Tensor,  # [BE, H]
        glob_emb: torch.Tensor,  # [B, H]
        edge_batch: torch.LongTensor,  # [BE, ]
    ) -> torch.Tensor:  # [BE, ]

        edge1_logits = self.edge_scorer(edge_emb, glob_emb, edge_batch)  # [BE, 1]

        return edge1_logits.squeeze(-1)  # [BE, ]

    def compute_edge2_logits(
        self,
        edge1_idx: torch.LongTensor,  # [B, ]
        edge_emb: torch.Tensor,  # [BE, H]
        glob_emb: torch.Tensor,  # [B, H]
        edge_batch: torch.LongTensor,  # [BE, ]
    ) -> torch.Tensor:  # [BE, ]

        edge2_logits = self.conditional_edge_scorer(  # [BE, 1]
            edge1_idx, edge_emb, glob_emb, edge_batch
        )

        return edge2_logits.squeeze(-1)  # [BE, ]

    def compute_mode_prob(
        self,
        edge1_idx: torch.LongTensor,  # [B, ]
        edge2_idx: torch.LongTensor,  # [B, ]
        edge_emb: torch.Tensor,  # [BE, H]
        glob_emb: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, ]

        # Edge1 and edge2 embeddings: [B, H]
        edge1_emb = edge_emb[edge1_idx]
        edge2_emb = edge_emb[edge2_idx]

        # Probability of mode, in [0, 1]
        mode_prob = self.mode_scorer(edge1_emb, edge2_emb, glob_emb)  # [B, 1]

        return mode_prob.squeeze(-1)  # [B, ]

    # ---- Masked Categorical/Bernoulli distribution ----
    def get_categorical_distribution(
        self,
        logits: torch.Tensor,  # [BE, ]
        edge_batch: torch.LongTensor,  # [BE, ]
        edge_ptr: torch.LongTensor,  # [B+1, ]
        mask: torch.BoolTensor | None = None,  # [BE, ]
    ) -> ScatterCategorical:
        """
        Args:
            mask: If True, the corresponding logit is masked out
        Return:
            Categorical distribution of shape [B, ]
        """
        if mask is None:
            mask = cast(torch.BoolTensor, torch.zeros_like(logits, dtype=torch.bool))

        # Mask out logits
        logits = logits.masked_fill(mask, -torch.inf)

        return ScatterCategorical(logits=logits, batch=edge_batch, ptr=edge_ptr)

    def get_bernoulli_distribution(
        self,
        probs: torch.Tensor,  # [B, ]
        mask0: torch.BoolTensor,  # [B, ]
        mask1: torch.BoolTensor,  # [B, ]
    ) -> ScatterBernoulli:
        """
        Args:
            mask0: If True, mode 0 is masked out. i.e., prob(m=1)=1
            mask1: If True, mode 1 is masked out. i.e., prob(m=1)=0

        Return:
            Bernoulli distribution of shape [B, ]
        """
        # Mask out probabilities
        probs = torch.where(mask0, 1.0, probs)  # prob of mode1 is 1
        probs = torch.where(mask1, 0.0, probs)  # prob of mode0 is 1

        return ScatterBernoulli(probs=probs)

    # ---- Override SB3's Actor-Critic Policy API ----
    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )

        return data

    def _build_mlp_extractor(self) -> None:
        # Create a dummy MLP extractor that satisfies SB3's interface
        class DummyMlpExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim_vf = 1
                self.latent_dim_pi = 1

            def forward(self, features):
                return features, features

            def forward_actor(self, features):
                return features

            def forward_critic(self, features):
                return features

        self.mlp_extractor = DummyMlpExtractor()

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Bypass default action_net and value_net
        Create dummy mlp_extractor (which we will not use) and create optimizer
        """

        # Create dummy mlp_extractor
        self._build_mlp_extractor()

        # Custom headers
        hidden_dim = self.features_extractor.hidden_dim

        # Conditional actors
        self.edge_scorer = EdgeScorer(hidden_dim)
        self.conditional_edge_scorer = ConditionalEdgeScorer(hidden_dim)
        self.mode_scorer = ModeScorer(hidden_dim)

        # Value head over pooled node embeddings
        self.value_net = ValueNet(hidden_dim)

        # Collect trainable parameters
        trainable_modules = [
            self.features_extractor,
            self.edge_scorer,
            self.conditional_edge_scorer,
            self.mode_scorer,
            self.value_net,
        ]
        params = []
        for module in trainable_modules:
            params += list(module.parameters())

        # Setup optimizer with initial learning rate 1
        self.optimizer = self.optimizer_class(
            params, lr=lr_schedule(1), **self.optimizer_kwargs  # type:ignore[call-arg]
        )

    def obs_to_tensor(self, obs: GraphObs | GraphVecObs) -> tuple[PygObs, bool]:  # type: ignore[override]
        if isinstance(obs, list):
            vectorized_env = True
            pyg_obs = PygObs.from_graph_vec_obs(
                cast(GraphVecObs, obs), device=self.device
            )
        else:
            vectorized_env = False
            pyg_obs = PygObs.from_graph_obs(cast(GraphObs, obs), device=self.device)

        return pyg_obs, vectorized_env

    def forward(  # type: ignore[override]
        self, obs: PygObs, deterministic: bool = False, only_action: bool = False
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions sequentially

        Return:
            action: [B, 3]
            value: [B, ]
            log_prob: [B, ]
        """
        num_batch = obs.num_graphs
        # Feature extraction: [BN, H], [BE, H], [B, GF-1 + H/2]
        node_emb, edge_emb, glob_emb = self.extract_features(obs)

        # --------------- Edge1 sampling ---------------
        edge1_logits = self.compute_edge1_logits(  # [BE, ]
            edge_emb, glob_emb, obs.edge_batch
        )

        # Get edge1 distribution P(e1) and sample edge1
        if deterministic:
            # Choose edge with the highest logit per graph
            edge1_dist = self.get_categorical_distribution(
                edge1_logits, obs.edge_batch, obs.edge_ptr
            )
            edge1_idx = scatter_argmax(
                edge1_logits, obs.edge_batch, dim=0, dim_size=num_batch
            )
        else:
            edge1_dist = self.get_categorical_distribution(
                edge1_logits,
                obs.edge_batch,
                obs.edge_ptr,
            )
            edge1_idx = edge1_dist.sample()

        # Global edge1 indices: index of edge1 in the edge_list [BE, 2] (with offset)
        edge1_idx = cast(torch.LongTensor, edge1_idx)  # [B, ]

        # --------------- Edge2 sampling ---------------
        # compute logits and mask of edge2, conditioned on edge1
        edge2_logits = self.compute_edge2_logits(  # [BE, ]
            edge1_idx, edge_emb, glob_emb, obs.edge_batch
        )

        # Get mask for invalid rewires when edge1 is fixed
        # [BE, ], [BE, ]
        (
            is_mode0_invalid,
            is_mode1_invalid,
            is_mode0_ineffective,
            is_mode1_ineffective,
        ) = self.get_masks(edge1_idx, obs)

        # Default mask of mode0 and mode1: both invalid and ineffective
        mode0_mask = is_mode0_invalid | is_mode0_ineffective
        mode1_mask = is_mode1_invalid | is_mode1_ineffective
        edge2_mask = mode0_mask & mode1_mask

        # Check if all batches have at least one valid edge2
        # If any batch has no valid edge2, use invalid mask only
        has_no_valid_edge2 = (  # [B, ]
            scatter_sum((~edge2_mask).float(), obs.edge_batch, dim_size=num_batch)
            == 0.0
        )
        if has_no_valid_edge2.any():
            needs_fallback = has_no_valid_edge2[obs.edge_batch]
            mode0_mask = torch.where(needs_fallback, is_mode0_invalid, mode0_mask)
            mode1_mask = torch.where(needs_fallback, is_mode1_invalid, mode1_mask)
            edge2_mask = mode0_mask & mode1_mask

        edge2_mask = cast(torch.BoolTensor, edge2_mask)

        # Check if all batches have at least one valid edge2
        # valid_edge2_per_batch = scatter_sum(
        #     (~edge2_mask).float(), obs.edge_batch, dim=0, dim_size=num_batch
        # )
        # assert (
        #     valid_edge2_per_batch > 0
        # ).all(), f"Some batches have no valid edge2! Valid counts: {valid_edge2_per_batch.tolist()}"

        # Get edge2 distribution P(e2 | e1) and sample edge2
        edge2_dist = self.get_categorical_distribution(
            edge2_logits, obs.edge_batch, obs.edge_ptr, edge2_mask
        )
        if deterministic:
            edge2_idx = scatter_argmax(
                edge2_dist.logits, obs.edge_batch, dim=0, dim_size=num_batch
            )
        else:
            edge2_idx = edge2_dist.sample()

        # Global edge2 indices: index of edge2 in the edge_list [BE, 2] (with offset)
        edge2_idx = cast(torch.LongTensor, edge2_idx)  # [B, ]

        # --------------- Mode sampling ---------------
        # Sample mode, conditioned on edge1 and edge2
        mode_prob = self.compute_mode_prob(  # [B, ]
            edge1_idx, edge2_idx, edge_emb, glob_emb
        )

        # Get mode mask for sampled edge2: [BE, ] -> [B, ]
        mode0_mask = cast(torch.BoolTensor, mode0_mask[edge2_idx])
        mode1_mask = cast(torch.BoolTensor, mode1_mask[edge2_idx])

        # Get mode distribution P(m | e1, e2) and sample mode
        mode_dist = self.get_bernoulli_distribution(mode_prob, mode0_mask, mode1_mask)
        if deterministic:
            mode = mode_dist.probs > 0.5  # [B, ], boolean
        else:
            mode = mode_dist.sample()  # [B, ], float32
        mode = cast(torch.LongTensor, mode.to(torch.int64))

        # ---------- Collect action, value, log_prob ------------
        # Change global edge indices to local edge indices: removing graph offsets
        local_edge1_idx = edge1_idx - obs.edge_ptr[:-1]  # [B, ]
        local_edge2_idx = edge2_idx - obs.edge_ptr[:-1]  # [B, ]

        action = torch.stack([local_edge1_idx, local_edge2_idx, mode], dim=1)  # [B, 3]
        action = cast(torch.LongTensor, action)

        if only_action:
            return action, torch.empty(0), torch.empty(0)

        # Value function: use only gap for value function
        value = self.value_net(  # [B, ]
            node_emb,  # [BN, H]
            edge_emb,  # [BE, H]
            obs.glob_attr,  # [B, 2]
            obs.node_batch,  # [BN, ]
            obs.edge_batch,  # [BE, ]
        )

        # Log-prob decomposition
        log_prob = (
            edge1_dist.log_prob(edge1_idx)
            + edge2_dist.log_prob(edge2_idx)
            + mode_dist.log_prob(mode.to(torch.float32))
        )

        return action, value, log_prob

    def extract_features(  # type: ignore[override]
        self, obs: PygObs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features_extractor(obs)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        """
        Action is sampled sequentially, so this method is not used
        """
        raise NotImplementedError(
            "_get_action_dist_from_latent is not used in AutoRegressiveRewirePolicy"
        )

    def predict(  # type: ignore[override]
        self, observation: GraphObs | GraphVecObs, deterministic: bool = False
    ) -> npt.NDArray[np.int64]:
        self.set_training_mode(False)

        obs, _ = self.obs_to_tensor(observation)
        with torch.no_grad():
            actions = self._predict(obs, deterministic=deterministic)

        actions = actions.cpu().numpy()  # [B, 3]
        return actions

    def _predict(  # type: ignore[override]
        self, observation: PygObs, deterministic: bool = False
    ) -> torch.LongTensor:
        actions, *_ = self.forward(
            observation, deterministic=deterministic, only_action=True
        )
        return actions

    def evaluate_actions(  # type: ignore[override]
        self, obs: PygObs, action: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Return:
            value: [B, ]
            log_prob: [B, ]
            entropy: None
        """
        # Feature extraction: [BN, H], [BE, H]
        node_emb, edge_emb, glob_emb = self.extract_features(obs)

        # Get action components
        local_edge1_idx, local_edge2_idx, mode = action.T  # [B, ], [B, ], [B, ]
        edge1_idx = cast(torch.LongTensor, local_edge1_idx + obs.edge_ptr[:-1])  # [B, ]
        edge2_idx = cast(torch.LongTensor, local_edge2_idx + obs.edge_ptr[:-1])  # [B, ]
        mode = cast(torch.LongTensor, mode)

        # -------- Value estimation --------
        # Only use gap for value function
        value = self.value_net(  # [B, ]
            node_emb,  # [BN, H]
            edge_emb,  # [BE, H]
            obs.glob_attr,  # [B, 2]
            obs.node_batch,  # [BN, ]
            obs.edge_batch,  # [BE, ]
        )

        # -------- Log probability --------
        # Distribution of edge1
        edge1_logits = self.compute_edge1_logits(  # [BE, ]
            edge_emb, glob_emb, obs.edge_batch
        )
        edge1_dist = self.get_categorical_distribution(
            edge1_logits, obs.edge_batch, obs.edge_ptr
        )

        # Distribution of edge2
        edge2_logits = self.compute_edge2_logits(  # [BE, ]
            edge1_idx, edge_emb, glob_emb, obs.edge_batch
        )
        # True if this rewiring is invalid
        (
            is_mode0_invalid,
            is_mode1_invalid,
            is_mode0_ineffective,
            is_mode1_ineffective,
        ) = self.get_masks(edge1_idx, obs)

        # Default mask of mode0 and mode1: both invalid and ineffective
        mode0_mask = is_mode0_invalid | is_mode0_ineffective
        mode1_mask = is_mode1_invalid | is_mode1_ineffective
        edge2_mask = mode0_mask & mode1_mask

        # Check if all batches have at least one valid edge2
        has_no_valid_edge2 = (  # [B, ]
            scatter_sum((~edge2_mask).float(), obs.edge_batch, dim_size=obs.num_graphs)
            == 0.0
        )
        # If any batch has no valid edge2, use invalid mask only
        needs_fallback = has_no_valid_edge2[obs.edge_batch]
        mode0_mask = torch.where(needs_fallback, is_mode0_invalid, mode0_mask)
        mode1_mask = torch.where(needs_fallback, is_mode1_invalid, mode1_mask)
        edge2_mask = mode0_mask & mode1_mask

        edge2_mask = cast(torch.BoolTensor, edge2_mask)

        # Mask of edge2 [BE, ]
        edge2_mask = cast(torch.BoolTensor, mode0_mask & mode1_mask)
        edge2_dist = self.get_categorical_distribution(
            edge2_logits, obs.edge_batch, obs.edge_ptr, edge2_mask
        )

        # Distribution of mode
        mode_prob = self.compute_mode_prob(  # [B, ]
            edge1_idx, edge2_idx, edge_emb, glob_emb
        )
        mode0_mask = cast(torch.BoolTensor, mode0_mask[edge2_idx])  # [B, ]
        mode1_mask = cast(torch.BoolTensor, mode1_mask[edge2_idx])  # [B, ]
        mode_dist = self.get_bernoulli_distribution(mode_prob, mode0_mask, mode1_mask)

        log_prob = (
            edge1_dist.log_prob(edge1_idx)
            + edge2_dist.log_prob(edge2_idx)
            + mode_dist.log_prob(mode.to(torch.float32))
        )

        return value, log_prob, None

    def predict_values(self, obs: PygObs) -> torch.Tensor:  # type: ignore[override]
        # Feature extraction: [BN, H], [BE, H]
        node_emb, edge_emb, _ = self.extract_features(obs)

        return self.value_net(  # [B, ]
            node_emb,  # [BN, H]
            edge_emb,  # [BE, H]
            obs.glob_attr,  # [B, 2]
            obs.node_batch,  # [BN, ]
            obs.edge_batch,  # [BE, ]
        )

    # ===================== Joint degree flux matrix =====================
    @staticmethod
    def get_invalid_mask_batched(
        edge1_indices: torch.LongTensor, obs: PygObs  # [E1, ]
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Get masks for invalid rewiring for multiple edge1 at once

        Args:
            edge1_indices: [E1, ], indices of edge1 candidates
            obs: PygObs of a single graph

        Returns:
            mode0_mask: [E1, E]. True if rewiring mode0 is invalid
            mode1_mask: [E1, E]. True if rewiring mode1 is invalid
        """
        num_edge1 = len(edge1_indices)
        num_nodes = obs.total_num_nodes

        # Nodes in edge1: [E1, 2]
        edge1_nodes = obs.edge_list[edge1_indices]  # [E1, 2]
        node1 = edge1_nodes[:, 0]  # [E1, ]
        node2 = edge1_nodes[:, 1]  # [E1, ]

        # Nodes in all edges: [E, 2]
        node3 = obs.edge_list[:, 0]  # [E, ]
        node4 = obs.edge_list[:, 1]  # [E, ]

        # -------------- Included mask: self-loop or pointless rewire --------------
        # Broadcast comparison: [E1, 1] vs [1, E] -> [E1, E]
        node3_is_node1 = node3.unsqueeze(0) == node1.unsqueeze(1)  # [E1, E]
        node4_is_node1 = node4.unsqueeze(0) == node1.unsqueeze(1)  # [E1, E]
        node3_is_node2 = node3.unsqueeze(0) == node2.unsqueeze(1)  # [E1, E]
        node4_is_node2 = node4.unsqueeze(0) == node2.unsqueeze(1)  # [E1, E]

        # Self-loop or pointless rewire mask
        included_mask = (  # [E1, E]
            node3_is_node1 | node3_is_node2 | node4_is_node1 | node4_is_node2
        )

        # -------------- Multi-edge mask --------------
        src, dst = obs.bi_edge_list.T  # [2BE, ], [2BE, ]

        # For each edge1, check if node3/node4 are neighbors
        # is_node1_neighbor[i, j] = True if node j is neighbor of node1[i]
        is_node1_neighbor = torch.zeros(
            (num_edge1, num_nodes), dtype=torch.bool, device=obs.device
        )
        is_node2_neighbor = torch.zeros(
            (num_edge1, num_nodes), dtype=torch.bool, device=obs.device
        )

        for edge1_idx, (n1, n2) in enumerate(zip(node1, node2)):
            mask_n1 = src == n1  # [2BE, ]
            mask_n2 = src == n2  # [2BE, ]
            is_node1_neighbor[edge1_idx, dst[mask_n1]] = True
            is_node2_neighbor[edge1_idx, dst[mask_n2]] = True

        # Mode 0 forbidden: [E1, E]
        mode0_mask = (
            included_mask
            | is_node1_neighbor[:, node3]  # [E1, E]
            | is_node2_neighbor[:, node4]  # [E1, E]
        )

        # Mode 1 forbidden: [E1, E]
        mode1_mask = (
            included_mask
            | is_node1_neighbor[:, node4]  # [E1, E]
            | is_node2_neighbor[:, node3]  # [E1, E]
        )

        # -------------- Assortativity mask --------------
        degrees = obs.node_attr.squeeze()  # [N, ]
        degree1 = degrees[node1].unsqueeze(1)  # [E1, 1]
        degree2 = degrees[node2].unsqueeze(1)  # [E1, 1]
        degree3 = degrees[node3].unsqueeze(0)  # [1, E]
        degree4 = degrees[node4].unsqueeze(0)  # [1, E]

        current_contrib = degree1 * degree2 + degree3 * degree4  # [E1, E]
        mode0_contrib = degree1 * degree3 + degree2 * degree4  # [E1, E]
        mode1_contrib = degree1 * degree4 + degree2 * degree3  # [E1, E]

        mode0_mask = mode0_mask | (current_contrib == mode0_contrib)
        mode1_mask = mode1_mask | (current_contrib == mode1_contrib)

        return cast(torch.BoolTensor, mode0_mask), cast(torch.BoolTensor, mode1_mask)

    def compute_edge2_logits_batched(
        self,
        edge1_indices: torch.LongTensor,  # [E1, ]
        edge_emb: torch.Tensor,  # [E, H]
        glob_emb: torch.Tensor,  # [1, H]
    ) -> torch.Tensor:
        """
        Compute edge2 logits for multiple edge1 at once using ConditionalEdgeScorer

        Args:
            edge1_indices: [E1, ] indices of selected edge1
            edge_emb: [E, H] embeddings of all edges
            glob_emb: [1, H] global embedding

        Returns:
            edge2_logits: [E1, E] logits for each (edge1, edge2) pair
        """
        num_edge1 = len(edge1_indices)

        # Get edge1 embeddings: [E1, H]
        edge1_emb = edge_emb[edge1_indices]

        # Query: [E1, H] + [E1, H] -> [E1, 2H] -> [E1, H]
        query_input = torch.cat([edge1_emb, glob_emb.expand(num_edge1, -1)], dim=-1)
        query = self.conditional_edge_scorer.query_projector(query_input)  # [E1, H]

        # Key: [E, H] (computed once for all edges)
        key = self.conditional_edge_scorer.key_projector(edge_emb)  # [E, H]

        # Attention Score: [E1, H] @ [H, E] = [E1, E]
        attention_score = torch.matmul(query, key.T)  # [E1, E]
        attention_score = self.conditional_edge_scorer.attention_scale * attention_score

        # Bias Score: [E, 1] (same for all edge1)
        bias_score = self.conditional_edge_scorer.bias_scorer(edge_emb)  # [E, 1]

        # Broadcast bias_score to [E1, E]
        # [E1, E] + [1, E] = [E1, E]
        edge2_logits = attention_score + bias_score.T  # [E1, E]

        return edge2_logits

    def get_categorical_distribution_batched(
        self,
        logits: torch.Tensor,  # [E1, E]
        mask: torch.BoolTensor,  # [E1, E]
    ) -> ScatterCategorical:
        """
        Get categorical distribution for different edge1 in a single graph

        Args:
            logits: [E1, E] logits for E1 different distributions over E elements
            mask: [E1, E] if True, the corresponding logit is masked out

        Return:
            distribution: ScatterCategorical with E1 independent distributions
        """
        num_edge1, num_edges = logits.shape
        device = logits.device

        # Create batch indices: [0,0,...,0, 1,1,...,1, ..., E1-1,...,E1-1]
        # Each group has E elements
        batch = torch.arange(
            num_edge1, device=device, dtype=torch.int64
        ).repeat_interleave(
            num_edges
        )  # [E1*E, ]

        # Create ptr: [0, E, 2E, ..., E1*E]
        ptr = (
            torch.arange(num_edge1 + 1, device=device, dtype=torch.int64) * num_edges
        )  # [E1+1, ]
        batch = cast(torch.LongTensor, batch)
        ptr = cast(torch.LongTensor, ptr)

        return ScatterCategorical(
            logits=logits.flatten().masked_fill(mask.flatten(), -torch.inf),
            batch=batch,
            ptr=ptr,
        )

    @torch.no_grad()
    def get_joint_degree_flux(
        self, observation: GraphObs | GraphVecObs
    ) -> torch.Tensor:
        """
        Get joint degree flux matrix of a graph

        Return:
            joint_degree_flux: [D, D], where D is the maximum degree of graph
        """
        self.set_training_mode(False)
        obs, _ = self.obs_to_tensor(observation)

        assert (
            obs.num_graphs == 1
        ), "get_transition_matrices only supports single environment"

        max_degree = int(obs.glob_attr[:, 1].item())
        joint_degree_flux = torch.zeros(
            (max_degree, max_degree), dtype=torch.float32, device=obs.device
        )

        degrees = (max_degree * obs.node_attr.squeeze()).to(torch.int64)  # [N, ]
        degrees_per_edge = degrees[obs.edge_list]  # [E, 2]

        # Feature extraction: [E, H], [1, H]
        _, edge_emb, glob_emb = self.extract_features(obs)

        # --------------- Edge1 distribution ---------------
        edge1_logits = self.compute_edge1_logits(  # [E, ]
            edge_emb, glob_emb, obs.edge_batch
        )
        edge1_dist = self.get_categorical_distribution(
            edge1_logits, obs.edge_batch, obs.edge_ptr
        )
        edge1_probs = edge1_dist.probs  # [E, ]

        # Filter out negligible probabilities
        is_probable_edge1 = edge1_probs >= 1e-8  # [E, ]
        edge1_probs = edge1_probs[is_probable_edge1]  # [E1, ]
        edge1_indices = cast(  # [E1, ]
            torch.LongTensor, torch.where(is_probable_edge1)[0].to(torch.int64)
        )
        num_edge1 = len(edge1_indices)

        # Edge 1 will be removed: update transition matrix
        degrees1, degrees2 = degrees_per_edge[edge1_indices].T  # [E1, ], [E1, ]
        joint_degree_flux.index_put_(
            (degrees1 - 1, degrees2 - 1), -edge1_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degrees2 - 1, degrees1 - 1), -edge1_probs, accumulate=True
        )

        # --------------- Edge2 probability ---------------
        # Get masks for all edge1 at once: [E1, E]
        mode0_mask, mode1_mask = self.get_invalid_mask_batched(edge1_indices, obs)
        edge2_mask = cast(torch.BoolTensor, mode0_mask & mode1_mask)  # [E1, E]

        edge2_logits = self.compute_edge2_logits_batched(  # [E1, E]
            edge1_indices, edge_emb, glob_emb
        )
        edge2_dist = self.get_categorical_distribution_batched(edge2_logits, edge2_mask)
        edge2_probs = edge2_dist.probs.reshape(num_edge1, -1)  # [E1, E]

        # Filter valid (edge1, edge2) pairs: V = E1 * E2
        is_probable = edge2_probs >= 1e-8
        is_valid = is_probable & (~edge2_mask)  # [E1, E]
        valid_edge1_indices, valid_edge2_indices = torch.where(is_valid)  # [V, ], [V, ]
        valid_edge1_indices_orig = edge1_indices[valid_edge1_indices]  # [V, ]
        valid_edge2_indices_orig = valid_edge2_indices

        # Probability of edge pairs
        valid_edge1_probs = edge1_probs[valid_edge1_indices]  # [V, ]
        valid_edge2_probs = edge2_probs[
            valid_edge1_indices, valid_edge2_indices
        ]  # [V, ]
        valid_edge_pair_probs = valid_edge1_probs * valid_edge2_probs  # [V, ]

        # Edge 2 removal
        degrees3, degrees4 = degrees_per_edge[
            valid_edge2_indices_orig
        ].T  # [V, ], [V, ]
        joint_degree_flux.index_put_(
            (degrees3 - 1, degrees4 - 1), -valid_edge_pair_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degrees4 - 1, degrees3 - 1), -valid_edge_pair_probs, accumulate=True
        )

        # --------------- Mode probability ---------------
        num_valid_pairs = len(valid_edge1_indices)
        mode_probs = self.compute_mode_prob(
            cast(torch.LongTensor, valid_edge1_indices_orig),  # [V, ]
            cast(torch.LongTensor, valid_edge2_indices_orig),  # [V, ]
            edge_emb,  # [E, H]
            glob_emb.repeat(num_valid_pairs, 1),  # [V, H]
        )

        # Mode 0 forbidden → P(mode=1) = 1
        mode_probs = torch.where(
            mode0_mask[valid_edge1_indices, valid_edge2_indices],
            torch.ones_like(mode_probs),
            mode_probs,
        )
        # Mode 1 forbidden → P(mode=1) = 0
        mode_probs = torch.where(
            mode1_mask[valid_edge1_indices, valid_edge2_indices],
            torch.zeros_like(mode_probs),
            mode_probs,
        )

        # Get degrees for edge1: [V, ], [V, ]
        degree1, degree2 = degrees_per_edge[valid_edge1_indices_orig].T

        # Add probability of mode 0: new pair is (1, 3), (2, 4)
        mode0_probs = (1.0 - mode_probs) * valid_edge_pair_probs  # [V, ]

        joint_degree_flux.index_put_(
            (degree1 - 1, degrees3 - 1), mode0_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degrees3 - 1, degree1 - 1), mode0_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degree2 - 1, degrees4 - 1), mode0_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degrees4 - 1, degree2 - 1), mode0_probs, accumulate=True
        )

        # Add probability of mode 1: new pair is (1, 4), (2, 3)
        mode1_probs = mode_probs * valid_edge_pair_probs  # [V, ]
        joint_degree_flux.index_put_(
            (degree1 - 1, degrees4 - 1), mode1_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degrees4 - 1, degree1 - 1), mode1_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degree2 - 1, degrees3 - 1), mode1_probs, accumulate=True
        )
        joint_degree_flux.index_put_(
            (degrees3 - 1, degree2 - 1), mode1_probs, accumulate=True
        )

        return joint_degree_flux

    def reset_cache(self) -> None:
        """Only for compatibility with EvalRewirePolicy"""
        return
