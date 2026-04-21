import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from DMGG.env.graph_vec_env import GraphVecEnv


def set_seed(seed: int, venv: "GraphVecEnv | None" = None) -> None:
    import os

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if venv is not None:
        venv.seed(seed)
        venv.action_space.seed(seed)


def get_rng_state(
    venv: "GraphVecEnv | None" = None, strict: bool = False
) -> dict[str, Any]:
    # Global RNG state
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }

    rng_state["venv"] = None if venv is None else venv.env_method("get_rng_state")

    # Set deterministic mode for CuDNN: may impact performance
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return rng_state


def set_rng_state(rng_state: dict[str, Any], venv: "GraphVecEnv | None" = None) -> None:
    py_state = tuple(rng_state["python"])
    random.setstate(py_state)

    np_state = tuple(rng_state["numpy"])
    np_state = (
        np_state[0],
        np.array(np_state[1], dtype=np.uint32),
        np_state[2],
        np_state[3],
        np_state[4],
    )
    np.random.set_state(np_state)

    torch_cpu_state = torch.tensor(rng_state["torch_cpu"], dtype=torch.uint8)
    torch.set_rng_state(torch_cpu_state)

    if rng_state["torch_cuda"] is not None:
        for i, s in enumerate(rng_state["torch_cuda"]):
            torch.cuda.set_rng_state(s.to(torch.uint8), device=i)

    if rng_state["venv"] is not None:
        assert venv is not None
        for i, env_rng_state in zip(range(venv.num_envs), rng_state["venv"]):
            venv.env_method("set_rng_state", env_rng_state, indices=[i])


def save_rng_state(result_dir: Path, rng_state: dict[str, Any]) -> None:
    with open(result_dir / "rng.pkl", "wb") as f:
        pickle.dump(rng_state, f)


def load_rng_state(result_dir: Path) -> dict[str, Any]:
    with open(result_dir / "rng.pkl", "rb") as f:
        return pickle.load(f)
