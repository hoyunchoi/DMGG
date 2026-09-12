from pathlib import Path

import torch
from stable_baselines3.common.vec_env import VecMonitor
from torch import optim

from baselines.ResiNet.hyperparameter import RESINET_RESULT_DIR, HyperParameter
from baselines.ResiNet.policy import ResiNetPolicy
from baselines.ResiNet.resinet_ppo import ResiNetPPO
from baselines.ResiNet.resinet_rewire_env import ResiNetRewireEnv
from baselines.ResiNet.rollout_buffer import ResiNetRolloutBuffer
from DMGG.env import GraphDummyVecEnv, GraphSubprocVecEnv, GraphVecEnv
from DMGG.experiment_setup import force_single_thread, optimize_torch
from DMGG.RL import GraphVecNormalize
from DMGG.RL.callback import TrackerCallback
from DMGG.rng import (
    get_rng_state,
    load_rng_state,
    save_rng_state,
    set_rng_state,
    set_seed,
)


def setup_training(
    hp: HyperParameter, mult: bool | None = None
) -> tuple[ResiNetPPO, TrackerCallback]:
    optimize_torch()

    device = torch.device(hp.device)
    resume_dir = RESINET_RESULT_DIR / hp.resume if hp.resume else None

    # --------- Environment ---------
    # Vectorized environment ffrom ResiNetRewireEnv
    make_env = lambda: ResiNetRewireEnv(**hp.env_kwargs)
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
    venv = VecMonitor(venv)  # type: ignore[arg-type]

    # Wrap with VecNormalize
    if resume_dir and hp.resume_vec_norm:
        venv = GraphVecNormalize.load(
            str(resume_dir / "vecnormalize.pkl"), venv  # type: ignore[arg-type]
        )
        venv.training = True  # type: ignore[attr-defined]
        for key, value in hp.vec_norm_kwargs.items():
            setattr(venv, key, value)
    else:
        venv = GraphVecNormalize(  # type: ignore[assignment]
            venv,  # type: ignore[arg-type]
            training=True,
            norm_obs=False,
            **hp.vec_norm_kwargs,
        )

    if hp.seed is not None:
        # If seed is provided, set seed regardless of resume
        set_seed(hp.seed, venv=venv)
    elif resume_dir:
        # If seed is not provided but resume is provided, restore RNG state
        rng_state = load_rng_state(resume_dir)
        set_rng_state(rng_state, venv=venv)

    if resume_dir:
        # Load model without environment as the action space could be different
        model = ResiNetPPO.load(
            resume_dir,
            env=venv,
            device=device,
            custom_objects={
                "rollout_buffer_class": ResiNetRolloutBuffer,
                "rollout_buffer_kwargs": {"batch_size": hp.batch_size},
            },
            compile=hp.compile,
        )
        model.learning_rate = hp.actor_learning_rate
        model.critic_learning_rate = hp.critic_learning_rate
        model.ent_coef = hp.ent_coef
    else:
        policy_kwargs = {
            **hp.policy_kwargs,
            "optimizer_class": optim.Adam,
            "optimizer_kwargs": {"eps": 1e-5},
        }
        model = ResiNetPPO(
            policy=ResiNetPolicy,
            policy_kwargs=policy_kwargs,
            env=venv,
            device=device,
            rollout_buffer_class=ResiNetRolloutBuffer,
            rollout_buffer_kwargs={"batch_size": hp.batch_size},
            amp_dtype=hp.amp_dtype,
            compile=hp.compile,
            **hp.ppo_kwargs,
        )

    # --------- Callbacks ---------
    tracker_callback = TrackerCallback()
    if resume_dir:
        tracker_callback.load(resume_dir)

    return model, tracker_callback


def setup_evaluation(hp: HyperParameter) -> ResiNetPPO:
    optimize_torch()

    device = torch.device(hp.device)
    resume_dir = RESINET_RESULT_DIR / hp.resume

    # --------- Dummy environment for loading model ---------
    make_env = lambda: ResiNetRewireEnv(**hp.env_kwargs)
    venv = GraphDummyVecEnv([make_env for _ in range(hp.num_envs)])

    return ResiNetPPO.load(
        resume_dir,
        env=venv,
        device=device,
        custom_objects={
            "rollout_buffer_class": ResiNetRolloutBuffer,
            "rollout_buffer_kwargs": {"batch_size": hp.batch_size},
        },
        compile=hp.compile,
    )


def save(result_dir: Path, model: ResiNetPPO, callback: TrackerCallback) -> None:
    # Save weights, hyperparameters, and num_timesteps in SB3 model
    model.save(result_dir)

    # Save normalization statistics in VecNormalize
    vec_normalize_env = model.get_vec_normalize_env()
    if vec_normalize_env is not None:
        vec_normalize_env.save(str(result_dir / "vecnormalize.pkl"))

    # Save rng state in global scope
    rng_state = get_rng_state(venv=model.env)
    save_rng_state(result_dir, rng_state)

    # Save callbacks
    callback.save(result_dir)
