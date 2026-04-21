import argparse
import sys
import typing
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DMGG.data import DMGGData, save
from DMGG.env import (
    EvalRewireEnv,
    GraphDummyVecEnv,
    GraphSubprocVecEnv,
    GraphVecEnv,
    RewireEnv,
)
from DMGG.experiment_setup import setup_evaluation
from DMGG.hyperparameter import get_hp
from DMGG.RL import RewirePolicy
from DMGG.type_aliases import RESET_METHOD, STORE
from graph.assortativity import get_maximum_assortativity, get_minimum_assortativity
from graph.edge_list import edge_list_to_csr, edge_list_to_degrees, edge_list_to_graph
from graph.generator import GENERATORS, configuration_model
from graph.properties import get_average_clustering_coefficient
from path import DATA_DIR


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Load DMGG model
    parser.add_argument("--exp_id", type=str, required=True)

    # Initial graph
    parser.add_argument("--graph_type", type=str, choices=list(GENERATORS.keys()))
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--mean_degree", type=float, required=True)
    parser.add_argument("--graph_seed", type=int, required=True)

    # Sampling configuration
    parser.add_argument(
        "--reset_method",
        default="soft",
        choices=typing.get_args(RESET_METHOD),
        help="Reset strategy for initial graph. Hard: create new initial graph for every sampling. Soft: keep degree sequence but use new configuration model. None: reuse initial graph for every sampling.",
    )
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--max_rewires", type=int)

    # Environment configuration
    parser.add_argument("--use_mult", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_envs", type=int, default=1)

    # Evaluation configuration
    parser.add_argument("--num_graphs", type=int, required=True)
    parser.add_argument(
        "--store", nargs="+", default=["metadata"], choices=typing.get_args(STORE)
    )

    return parser.parse_args(options)


def sample_graph(
    policy: RewirePolicy,
    env: GraphVecEnv,
    num_valid_graphs: int,
    store_joint_degree: bool,
) -> list[DMGGData]:
    step = 0
    num_infeasible = 0
    data_list: list[DMGGData] = []
    is_successes: list[bool] = []
    joint_degree_matrices: list[npt.NDArray[np.uint32]] = []
    joint_degree_fluxes: list[npt.NDArray[np.float32]] = []

    obs = env.reset()

    while len(is_successes) < num_valid_graphs:
        # ========= Compute joint degree matrix and flux =========
        if store_joint_degree:
            joint_degree_matrix = env.get_attr("joint_degree_matrix")[0]
            joint_degree_flux = policy.get_joint_degree_flux(obs).cpu().numpy()

            joint_degree_matrices.append(joint_degree_matrix.astype(np.uint32).copy())
            joint_degree_fluxes.append(joint_degree_flux.astype(np.float32).copy())

            step += 1
            print(f"{step=}", flush=True)
        # ==================================================

        actions = policy.predict(obs, deterministic=False)
        obs, _, dones, infos = env.step(actions)

        # Check if the episode is done
        for env_id in np.where(dones)[0]:
            policy.reset_cache()  # used for EvalRewirePolicy

            info = infos[env_id]
            is_success = not info["TimeLimit.truncated"]

            if len(info["history"]) == 1:
                feasible_range = info["feasible_rho_range"]
                print(
                    f"Feasible range: {feasible_range[0]:.4f} - {feasible_range[1]:.4f}"
                )
                num_infeasible += 1

            if store_joint_degree:
                joint_degree_matrix = info["joint_degree_matrix"]
                joint_degree_matrices.append(
                    joint_degree_matrix.astype(np.uint32).copy()
                )
                joint_degree_matrix_history = np.stack(joint_degree_matrices)
                joint_degree_flux_history = np.stack(joint_degree_fluxes)
            else:
                joint_degree_matrix_history = np.array([])
                joint_degree_flux_history = np.array([])

            # Get clustering coefficient of the sampled graph
            degree = edge_list_to_degrees(info["num_nodes"], info["edge_list"])
            offset, neighbor = edge_list_to_csr(
                info["num_nodes"], info["edge_list"], degree
            )
            clustering_coefficient = get_average_clustering_coefficient(
                degree, offset, neighbor
            )

            data = DMGGData(
                graph_type=info["graph_type"],
                num_nodes=info["num_nodes"],
                num_edges=info["num_edges"],
                mean_degree=info["mean_degree"],
                target_assortativity=info["target_rho"],
                is_success=is_success,
                edge_list=info["edge_list"],
                assortativity_history=np.array(info["history"], dtype=np.float32),
                joint_degree_matrix_history=joint_degree_matrix_history,
                joint_degree_flux_history=joint_degree_flux_history,
                clustering_coefficient=clustering_coefficient,
                runtime=info["runtime"],
            )

            data_list.append(data)
            is_successes.append(is_success)

            print(
                f"{sum(is_successes)}/{len(is_successes)} ({num_infeasible})",
                end="\t",
                flush=True,
            )

    return data_list


def main() -> None:
    args = get_args()
    args.use_mult = bool(args.use_mult)
    store_joint_degree = "joint_degree" in args.store
    if store_joint_degree:
        assert (
            args.num_envs == 1
        ), "Storing joint degree only supports single environment"

    # Load hyperparameters and model
    hp = get_hp([f"--resume={args.exp_id}"])
    model = setup_evaluation(hp)
    policy = model.policy

    env_kwargs = hp.env_kwargs

    # ========== Setup initial graphs & graph range ==========
    # Create fixed initial graph
    graph_rng = np.random.default_rng(args.graph_seed)

    while True:
        edge_list = GENERATORS[args.graph_type](
            args.num_nodes, args.mean_degree, graph_rng
        )
        degrees = edge_list_to_degrees(args.num_nodes, edge_list)

        # Configuration model to remove initial topology-dependence
        edge_list = configuration_model(degrees, graph_rng)
        if len(edge_list) == 0:
            continue

        feasible_rho_min = get_minimum_assortativity(degrees, edge_list, graph_rng)
        feasible_rho_max = get_maximum_assortativity(degrees, edge_list, graph_rng)

        if (
            (feasible_rho_min is not None)
            and (feasible_rho_max is not None)
            and (feasible_rho_min + 0.02 < feasible_rho_max)
        ):
            init_feasible_rho_range = (feasible_rho_min, feasible_rho_max)
            init_graph = edge_list_to_graph(args.num_nodes, edge_list)
            break

    # Overriding RewireEnv kwargs (used in hard reset)
    env_kwargs["graph_types"] = [args.graph_type]
    env_kwargs["num_nodes_range"] = (args.num_nodes, args.num_nodes)
    env_kwargs["mean_degree_range"] = (args.mean_degree, args.mean_degree)
    env_kwargs["target_rho_range"] = (args.target, args.target)

    # EvalRewireEnv kwargs (used in soft or none reset)
    env_kwargs["reset_method"] = args.reset_method
    env_kwargs["init_graph"] = init_graph
    env_kwargs["init_graph_type"] = args.graph_type
    env_kwargs["init_feasible_rho_range"] = init_feasible_rho_range

    # Override Termination & truncation if provided
    if args.tolerance is not None:
        env_kwargs["tolerance"] = args.tolerance

    if args.max_rewires is not None:
        env_kwargs["max_rewires"] = args.max_rewires

    # =========== Setup environment ===========
    env_fns: list[Callable[[], RewireEnv]] = [
        lambda: EvalRewireEnv(**env_kwargs) for _ in range(args.num_envs)
    ]
    env: GraphVecEnv
    if (args.num_envs > 1) and args.use_mult:
        env = GraphSubprocVecEnv(env_fns)
    else:
        env = GraphDummyVecEnv(env_fns)

    # =========== Sample graphs ===========
    data_list = sample_graph(policy, env, args.num_graphs, store_joint_degree)

    # Write dataclass fields
    for data in data_list:
        data.graph_seed = args.graph_seed
        data.reset_method = args.reset_method
        data.max_rewires = env_kwargs["max_rewires"]

    # =========== Save results ===========
    data_dir = DATA_DIR / "DMGG"
    data_dir.mkdir(exist_ok=True)

    g = args.graph_type
    n = args.num_nodes
    t = args.target
    file_name = (
        f"{args.exp_id}_{g}_N{n}_t{t}_{datetime.now().strftime('%m%d_%H%M%S_%f')}.h5"
    )

    save(data_dir / file_name, data_list, args.store)


if __name__ == "__main__":
    main()
