import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from path import RESULT_DIR

RESINET_RESULT_DIR = RESULT_DIR / "ResiNet"


DEFAULTS: dict[str, Any] = {
    "device": 0,
    "seed": None,
    "resume": "",
    "resume_vec_norm": False,
    "amp_dtype": "fp32",
    "compile": False,
    "mult": False,
    "graph_types": ["WS", "ER", "BA"],
    "num_nodes_range": (100, 1000),
    "mean_degree_range": (3.0, 10.0),
    "filtration_order": 1,
    "zeta": 0.005,
    "success_bonus": 100.0,
    "step_penalty": -0.001,
    "target_rho_range": (-0.4, 0.4),
    "tolerance": 0.002,
    "max_rewires": 3000,
    "num_envs": 32,
    "clip_reward": 5.0,
    "num_layers": 5,
    "hidden_dim": 64,
    "actor_learning_rate": 7e-4,
    "critic_learning_rate": 7e-4,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": 1.0,
    "normalize_advantage": True,
    "ent_coef": 0.01,
    "vf_coef": 1.0,
    "dual_clip_param": 10.0,
    "max_grad_norm": 0.5,
    "total_timesteps": 800_000,
    "verbose": 2,
    "tensorboard_log": "./tensorboard/ResiNet",
    "progress_bar": False,
}


DONT_LOAD = {
    "seed",
    "resume",
    "resume_vec_norm",
    "verbose",
    "tensorboard_log",
    "progress_bar",
    "mult",
}


CHECK_CONSISTENCY = {
    "num_layers",
    "hidden_dim",
    "filtration_order",
    "normalize_advantage",
    "amp_dtype",
}


@dataclass
class HyperParameter:
    device: int
    seed: int
    resume: str
    resume_vec_norm: bool
    amp_dtype: str
    compile: bool
    mult: bool
    graph_types: list[str]
    num_nodes_range: tuple[int, int]
    mean_degree_range: tuple[float, float]
    filtration_order: int
    zeta: float
    success_bonus: float
    step_penalty: float
    target_rho_range: tuple[float, float]
    tolerance: float
    max_rewires: int
    num_envs: int
    clip_reward: float
    num_layers: int
    hidden_dim: int
    actor_learning_rate: float
    critic_learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_vf: float
    normalize_advantage: bool
    ent_coef: float
    vf_coef: float
    dual_clip_param: float
    max_grad_norm: float
    total_timesteps: int

    verbose: int
    tensorboard_log: str | None
    progress_bar: bool

    def __post_init__(self) -> None:
        if self.resume:
            self._load(RESINET_RESULT_DIR / self.resume)

        for key, default in DEFAULTS.items():
            if getattr(self, key) is None:
                setattr(
                    self, key, default.copy() if isinstance(default, list) else default
                )

        self.resume_vec_norm = bool(self.resume_vec_norm)
        self.normalize_advantage = bool(self.normalize_advantage)
        self.compile = bool(self.compile)
        self.mult = bool(self.mult)
        self.progress_bar = bool(self.progress_bar)
        self.num_nodes_range = tuple(self.num_nodes_range)  # type: ignore[arg-type,assignment]
        self.mean_degree_range = tuple(self.mean_degree_range)  # type: ignore[arg-type,assignment]
        self.target_rho_range = tuple(self.target_rho_range)  # type: ignore[arg-type,assignment]
        self._validity_check()

    def _load(self, result_dir: Path) -> None:
        with open(result_dir / "hp.yaml", encoding="utf-8") as hp_file:
            loaded = yaml.safe_load(hp_file)

        for key, value in loaded.items():
            if key in DONT_LOAD or not hasattr(self, key):
                continue
            current_value = getattr(self, key)
            if current_value is None:
                setattr(self, key, value)
            elif key in CHECK_CONSISTENCY and current_value != value:
                raise ValueError(
                    f"Loaded {key}={value} differs from requested {current_value}"
                )

    def _validity_check(self) -> None:
        for key, value in self.__dict__.items():
            if key in {"seed", "tensorboard_log"}:
                continue
            if value is None:
                raise ValueError(f"{key} must not be None")

        assert self.graph_types is not None
        assert self.num_nodes_range is not None
        assert self.mean_degree_range is not None
        assert self.filtration_order is not None
        assert self.num_envs is not None
        assert self.n_steps is not None
        assert self.batch_size is not None
        assert self.hidden_dim is not None
        assert self.dual_clip_param is not None

        if not self.graph_types:
            raise ValueError("Provide at least one graph type")
        if (
            self.num_nodes_range[0] <= 0
            or self.num_nodes_range[1] <= self.num_nodes_range[0]
        ):
            raise ValueError("num_nodes_range must be increasing and positive")
        if (
            self.mean_degree_range[0] <= 0
            or self.mean_degree_range[1] <= self.mean_degree_range[0]
        ):
            raise ValueError("mean_degree_range must be increasing and positive")
        if self.filtration_order != 1:
            raise ValueError(
                "TC-ResiNet-sign currently supports filtration_order=1 only"
            )
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even")
        if self.num_envs * self.n_steps % self.batch_size:
            raise ValueError("num_envs * n_steps must be divisible by batch_size")
        if self.normalize_advantage and self.batch_size <= 1:
            raise ValueError("advantage normalization requires batch_size > 1")
        if self.dual_clip_param <= 1.0:
            raise ValueError("dual_clip_param must be greater than 1")

    def save(self, result_dir: Path) -> None:
        with open(result_dir / "hp.yaml", "w", encoding="utf-8") as hp_file:
            yaml.safe_dump(asdict(self), hp_file, sort_keys=False)

    @property
    def env_kwargs(self) -> dict[str, Any]:
        return {
            "graph_types": self.graph_types,
            "num_nodes_range": self.num_nodes_range,
            "mean_degree_range": self.mean_degree_range,
            "filtration_order": self.filtration_order,
            "target_rho_range": self.target_rho_range,
            "zeta": self.zeta,
            "success_bonus": self.success_bonus,
            "step_penalty": self.step_penalty,
            "tolerance": self.tolerance,
            "max_rewires": self.max_rewires,
        }

    @property
    def vec_norm_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma, "clip_reward": self.clip_reward}

    @property
    def policy_kwargs(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "node_dim": 15,
            "filtration_order": self.filtration_order,
        }

    @property
    def ppo_kwargs(self) -> dict[str, Any]:
        assert self.actor_learning_rate is not None
        assert self.critic_learning_rate is not None
        assert self.ent_coef is not None
        return {
            "learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
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
            "dual_clip_param": self.dual_clip_param,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": None,
            "verbose": self.verbose,
            "tensorboard_log": self.tensorboard_log,
        }


def get_hp(options: list[str] | None = None) -> HyperParameter:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_vec_norm", type=int)
    parser.add_argument("--amp_dtype", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--compile", type=int)
    parser.add_argument("--mult", type=int)

    parser.add_argument("--graph_types", type=str, nargs="+")
    parser.add_argument("--num_nodes_range", type=int, nargs=2)
    parser.add_argument("--mean_degree_range", type=float, nargs=2)
    parser.add_argument("--filtration_order", type=int)

    parser.add_argument("--zeta", type=float)
    parser.add_argument("--step_penalty", type=float)
    parser.add_argument("--success_bonus", type=float)
    parser.add_argument("--target_rho_range", type=float, nargs=2)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--max_rewires", type=int)

    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--clip_reward", type=float)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--hidden_dim", type=int)

    parser.add_argument("--actor_learning_rate", type=float)
    parser.add_argument("--critic_learning_rate", type=float)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)

    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae_lambda", type=float)
    parser.add_argument("--clip_range", type=float)
    parser.add_argument("--clip_range_vf", type=float)
    parser.add_argument("--normalize_advantage", type=int)
    parser.add_argument("--ent_coef", type=float)
    parser.add_argument("--vf_coef", type=float)
    parser.add_argument("--dual_clip_param", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--total_timesteps", type=int)

    parser.add_argument("--verbose", type=int)
    parser.add_argument("--tensorboard_log", type=str)
    parser.add_argument("--progress_bar", action="store_true", default=None)

    return HyperParameter(**vars(parser.parse_args(options)))
