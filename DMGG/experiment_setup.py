import string
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor
from torch import optim

from DMGG.env import GraphDummyVecEnv, GraphSubprocVecEnv, GraphVecEnv, RewireEnv
from DMGG.hyperparameter import HyperParameter
from DMGG.net import GraphFeaturesExtractor
from DMGG.RL import GraphPPO, GraphVecNormalize, RewirePolicy
from DMGG.RL.callback import TrackerCallback
from DMGG.RL.graph_rollout_buffer import GraphRolloutBuffer
from DMGG.rng import (
    get_rng_state,
    load_rng_state,
    save_rng_state,
    set_rng_state,
    set_seed,
)
from path import RESULT_DIR


def get_random_id(length: int = 8) -> str:
    rng = np.random.default_rng(int(time.time() * 1e7))
    return "".join(rng.choice(list(string.ascii_lowercase + string.digits), length))


def force_single_thread() -> None:
    """
    Sets the number of threads used by common numerical libraries and PyTorch to 1.
    This is important when using subprocess-based vectorized environments (such as SubprocVecEnv), as each process may otherwise spawn its own threads for computation.
    Without this limitation, total CPU thread usage can greatly exceed the number of available cores, leading to contention and potential slowdowns.
    This function sets environment variables for thread control in Numba, OpenMP, MKL, and OpenBLAS, and configures PyTorch's intra- and inter-op parallelism to use a single thread.
    """
    import os

    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def optimize_torch() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def setup_training(
    hp: HyperParameter, mult: bool = False
) -> tuple[GraphPPO, TrackerCallback]:
    """
    mult: whether to use multiple processes for training
    """
    optimize_torch()

    device = torch.device(hp.device)
    resume_dir = RESULT_DIR / hp.resume if hp.resume else None

    # --------- Environment ---------
    # Vectorized environment from RewireEnv
    make_env = lambda: RewireEnv(**hp.env_kwargs)
    venv: GraphVecEnv

    # If num_envs is 1, subprocess vec env is not required
    if hp.num_envs == 1:
        mult = False

    if mult:
        force_single_thread()
        venv = GraphSubprocVecEnv(
            [make_env for _ in range(hp.num_envs)], start_method="spawn"
        )
    else:
        venv = GraphDummyVecEnv([make_env for _ in range(hp.num_envs)])
    venv = VecMonitor(venv)  # type:ignore[assignment]

    # Wrap with VecNormalize
    if resume_dir and hp.resume_vec_norm:
        venv = GraphVecNormalize.load(  # type:ignore[assignment]
            str(resume_dir / "vecnormalize.pkl"), venv  # type:ignore[arg-type]
        )
        venv.training = True  # type:ignore[attr-defined]
        for key, value in hp.vec_norm_kwargs.items():
            setattr(venv, key, value)
    else:
        venv = GraphVecNormalize(  # type:ignore[assignment]
            venv,  # type:ignore[arg-type]
            training=True,
            norm_obs=False,
            **hp.vec_norm_kwargs,
        )

    # --------- Randomness ---------
    if hp.seed is not None:
        # If seed is provided, set seed regardless of resume
        set_seed(hp.seed, venv=venv)
    elif resume_dir:
        # If seed is not provided but resume is provided, restore RNG state
        rng_state = load_rng_state(resume_dir)
        set_rng_state(rng_state, venv=venv)

    # --------- PPO Model ---------
    if resume_dir:
        # Load model without environment as the action space could be different
        model = GraphPPO.load(
            resume_dir,
            env=venv,
            device=device,
            custom_objects={
                "rollout_buffer_class": GraphRolloutBuffer,
                "rollout_buffer_kwargs": {"batch_size": hp.batch_size},
                **hp.graph_ppo_kwargs,
            },
            policy_kwargs={
                "features_extractor_class": GraphFeaturesExtractor,
                "features_extractor_kwargs": hp.features_extractor_kwargs,
                "optimizer_class": getattr(optim, hp.optimizer_class),
                "optimizer_kwargs": {"weight_decay": hp.weight_decay, "fused": True},
            },
            compile=hp.compile,
        )

    else:
        model = GraphPPO(
            policy=RewirePolicy,
            policy_kwargs={
                "features_extractor_class": GraphFeaturesExtractor,
                "features_extractor_kwargs": hp.features_extractor_kwargs,
                "optimizer_class": getattr(optim, hp.optimizer_class),
                "optimizer_kwargs": {"weight_decay": hp.weight_decay, "fused": True},
            },
            env=venv,
            device=device,
            rollout_buffer_class=GraphRolloutBuffer,
            rollout_buffer_kwargs={"batch_size": hp.batch_size},
            amp_dtype=hp.amp_dtype,
            compile=hp.compile,
            **hp.graph_ppo_kwargs,
        )

    # --------- Callbacks ---------
    tracker_callback = TrackerCallback()
    if resume_dir:
        tracker_callback.load(resume_dir)

    return model, tracker_callback


def setup_evaluation(hp: HyperParameter) -> GraphPPO:
    optimize_torch()

    device = torch.device(hp.device)
    resume_dir = RESULT_DIR / hp.resume

    # --------- Dummy environment for loading model ---------
    make_env = lambda: RewireEnv(**hp.env_kwargs)
    venv = GraphDummyVecEnv([make_env for _ in range(hp.num_envs)])

    # --------- PPO Model ---------
    model = GraphPPO.load(
        resume_dir,
        env=venv,
        device=device,
        custom_objects={
            "rollout_buffer_class": GraphRolloutBuffer,
            "rollout_buffer_kwargs": {"batch_size": hp.batch_size},
            **hp.graph_ppo_kwargs,
        },
        policy_kwargs={
            "features_extractor_class": GraphFeaturesExtractor,
            "features_extractor_kwargs": hp.features_extractor_kwargs,
            "optimizer_class": getattr(optim, hp.optimizer_class),
            "optimizer_kwargs": {"weight_decay": hp.weight_decay, "fused": True},
        },
        compile=hp.compile,
    )

    return model


def save(result_dir: Path, model: GraphPPO, callback: TrackerCallback) -> None:
    # Save weights, hyperparameters, and num_timesteps in SB3 model
    model.save(result_dir)

    # Save normalization statistics in VecNormalize
    vec_normalize_env = model.get_vec_normalize_env()
    if vec_normalize_env is not None:
        vec_normalize_env.save(f"{result_dir}/vecnormalize.pkl")

    # Save rng state in global scope
    rng_state = get_rng_state(venv=model.env)
    save_rng_state(result_dir, rng_state)

    # Save callbacks
    callback.save(result_dir)
