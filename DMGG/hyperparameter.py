"""
Default behavior of loading hyperparameters
: If user-provided value is None, use the loaded value

Exception 1 (DONT_LOAD)
: Regardless of user-provided value, ignore the loaded value.

Exception2 (CHECK_CONSISTENCY)
: If user-provided value is not None, ensure it is same as the loaded value.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from path import RESULT_DIR

DONT_LOAD = [
    "seed",
    "resume",
    "resume_vec_norm",
    "verbose",
    "tensorboard_log",
    "progress_bar",
]

CHECK_CONSISTENCY = [
    "num_layers",
    "hidden_dim",
    "optimizer_class",
    "normalize_advantage",
    "amp_dtype",
]


@dataclass
class HyperParameter:
    # Basic configuration
    device: int
    seed: int | None
    resume: str
    resume_vec_norm: bool
    amp_dtype: str
    compile: bool

    # Graph
    graph_types: list[str]
    num_nodes_range: tuple[int, int]
    mean_degree_range: tuple[float, float]

    # Rewards
    zeta: float
    success_bonus: float
    step_penalty: float

    # Target / truncation & termination
    target_rho_range: tuple[float, float]
    tolerance: float
    max_rewires: int

    # Vectorized environment
    num_envs: int
    clip_reward: float

    # Features extractor
    num_layers: int
    hidden_dim: int

    # Optimization
    optimizer_class: str
    weight_decay: float
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int

    # PPO
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_vf: float
    normalize_advantage: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float
    total_timesteps: int

    # Logging
    verbose: int
    tensorboard_log: str | None
    progress_bar: bool

    def __post_init__(self) -> None:
        # If resume is not empty, load the hyperparameters from the result directory
        if self.resume:
            self._load(RESULT_DIR / self.resume)
        else:
            # Boolean type conversion
            self.resume_vec_norm = bool(self.resume_vec_norm)
            self.normalize_advantage = bool(self.normalize_advantage)
            self.compile = bool(self.compile)

        # Tuple type conversion
        self.target_rho_range = (self.target_rho_range[0], self.target_rho_range[1])
        self.num_nodes_range = (self.num_nodes_range[0], self.num_nodes_range[1])
        self.mean_degree_range = (self.mean_degree_range[0], self.mean_degree_range[1])

        # Check the validity of the hyperparameters
        self._validity_check()

    def _load(self, result_dir: Path) -> None:
        """
        Load hyperparameters from result directory.
        Only overwrite the current value if current value is None.
        """
        with open(result_dir / "hp.yaml", "r") as f:
            loaded_hp = yaml.safe_load(f)

        for key, value in loaded_hp.items():
            # If the key is in DONOT_OVERRIDE_KEYS, maintain current value, whether it is None or not.
            if key in DONT_LOAD:
                continue

            # If current value is None, set the loaded value
            if getattr(self, key) is None:
                setattr(self, key, value)

            # If current value is not None, check the consistency
            elif key in CHECK_CONSISTENCY:
                current_value = getattr(self, key)
                assert (
                    current_value == value
                ), f"The loaded value of {key} - {value} is not the same as the current value - {current_value}"

    def _validity_check(self) -> None:
        # For these keys, None value is allowed
        none_allowed = ["seed", "tensorboard_log"]

        for key, value in self.__dict__.items():
            if key in none_allowed:
                continue
            assert value is not None, f"The value of {key} is None"

        assert len(self.graph_types) > 0, "Provide at least one graph type"
        assert self.num_nodes_range[0] > 0, "Minimum number of nodes must be positive"
        assert self.num_nodes_range[1] > self.num_nodes_range[0], "Maximum number of nodes must be greater than minimum number of nodes"
        assert self.mean_degree_range[0] > 0.0, "Minimum mean degree must be positive"
        assert self.mean_degree_range[1] > self.mean_degree_range[0], "Maximum mean degree must be greater than minimum mean degree"
        assert self.success_bonus >= 0.0, "Success bonus must be non-negative"
        assert self.step_penalty <= 0.0, "Step penalty must be non-positive"
        assert self.tolerance > 0.0, "Tolerance must be positive"
        assert self.num_envs > 0, "Number of environments must be positive"
        assert self.hidden_dim % 2 == 0, "Hidden dimension must be even"

    def save(self, result_dir: Path) -> None:
        with open(result_dir / "hp.yaml", "w") as f:
            yaml.safe_dump(asdict(self), f)

    @property
    def env_kwargs(self) -> dict[str, Any]:
        return {
            # Graph
            "graph_types": self.graph_types,
            "num_nodes_range": self.num_nodes_range,
            "mean_degree_range": self.mean_degree_range,
            # Rewards
            "target_rho_range": self.target_rho_range,
            "zeta": self.zeta,
            "success_bonus": self.success_bonus,
            "step_penalty": self.step_penalty,
            # Truncation and termination
            "tolerance": self.tolerance,
            "max_rewires": self.max_rewires,
        }

    @property
    def vec_norm_kwargs(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "clip_reward": self.clip_reward,
        }

    @property
    def features_extractor_kwargs(self) -> dict[str, Any]:
        return {"num_layers": self.num_layers, "hidden_dim": self.hidden_dim}

    @property
    def graph_ppo_kwargs(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "normalize_advantage": self.normalize_advantage,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "verbose": self.verbose,
            "tensorboard_log": self.tensorboard_log,
        }


def get_hp(options: list[str] | None = None) -> HyperParameter:
    parser = argparse.ArgumentParser()

    # Basic configuration
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_vec_norm", type=int, default=0)
    parser.add_argument("--amp_dtype", type=str, default="fp32")
    parser.add_argument("--compile", type=int, default=1)

    # Graph range
    parser.add_argument("--graph_types", type=str, nargs="+")
    parser.add_argument("--num_nodes_range", type=int, nargs=2)
    parser.add_argument("--mean_degree_range", type=float, nargs=2)

    # Reward
    parser.add_argument("--zeta", type=float, default=0.005)
    parser.add_argument("--step_penalty", type=float, default=-0.001)
    parser.add_argument("--success_bonus", type=float, default=100.0)

    # Truncation and termination
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--max_rewires", type=int)

    # Target
    parser.add_argument("--target_rho_range", type=float, nargs=2)

    # Vectorized environment
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--clip_reward", type=float, default=5.0)

    # Features extractor
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=32)

    # Optimization
    parser.add_argument("--optimizer_class", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=8)

    # PPO
    parser.add_argument("--gamma", type=float, help="Discount factor for rewards", default=0.997)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.15)
    parser.add_argument("--clip_range_vf", type=float, default=0.15)
    parser.add_argument("--normalize_advantage", type=int, default=1)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--total_timesteps", type=int)

    # Logging
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--tensorboard_log", type=str, default=None)
    parser.add_argument("--progress_bar", action="store_true")

    args = parser.parse_args(options)
    return HyperParameter(**vars(args))
