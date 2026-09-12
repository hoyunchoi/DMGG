import argparse
import sys
import typing
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from path import DATA_DIR

from baselines.ResiNet.experiment_setup import setup_evaluation
from baselines.ResiNet.hyperparameter import get_hp
from baselines.ResiNet.observation import ResiNetVecObs
from baselines.ResiNet.policy import ResiNetPolicy
from baselines.ResiNet.resinet_rewire_env import ResiNetEvalRewireEnv
from DMGG.data import DMGGData, save
from DMGG.env import GraphDummyVecEnv, GraphSubprocVecEnv, GraphVecEnv, RewireEnv
from DMGG.type_aliases import RESET_METHOD, STORE
from graph.assortativity import get_maximum_assortativity, get_minimum_assortativity
from graph.edge_list import edge_list_to_csr, edge_list_to_degrees, edge_list_to_graph
from graph.generator import GENERATORS, configuration_model
from graph.properties import get_average_clustering_coefficient

ResiNetData = DMGGData


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Load TC-ResiNet-sign model
    parser.add_argument("--exp_id", type=str, required=True)

    # Initial graph
    parser.add_argument(
        "--graph_type", type=str, required=True, choices=list(GENERATORS.keys())
    )
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
    policy: ResiNetPolicy, env: GraphVecEnv, num_graphs: int
) -> list[ResiNetData]:
    num_infeasible = 0
    data_list: list[ResiNetData] = []
    is_successes: list[bool] = []

    obs = cast(ResiNetVecObs, env.reset())

    while len(is_successes) < num_graphs:
        actions = policy.predict(obs, deterministic=False)
        obs, _, dones, infos = env.step(actions)
        obs = cast(ResiNetVecObs, obs)

        for env_id in np.where(dones)[0]:
            info = infos[env_id]
            is_success = not info["TimeLimit.truncated"]

            if len(info["history"]) == 1:
                feasible_range = info["feasible_rho_range"]
                print(
                    f"Feasible range: {feasible_range[0]:.4f} - {feasible_range[1]:.4f}"
                )
                num_infeasible += 1

            # Get clustering coefficient of the sampled graph
            degree = edge_list_to_degrees(info["num_nodes"], info["edge_list"])
            offset, neighbor = edge_list_to_csr(
                info["num_nodes"], info["edge_list"], degree
            )
            clustering_coefficient = get_average_clustering_coefficient(
                degree, offset, neighbor
            )


            data = ResiNetData(
                graph_type=info["graph_type"],
                num_nodes=info["num_nodes"],
                num_edges=info["num_edges"],
                mean_degree=info["mean_degree"],
                target_assortativity=info["target_rho"],
                is_success=is_success,
                edge_list=info["edge_list"],
                assortativity_history=np.array(info["history"], dtype=np.float32),
                joint_degree_matrix_history=np.array([]),
                joint_degree_flux_history=np.array([]),
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
    assert "joint_degree" not in args.store, "Joint degree flux is not supported in ResiNet"

    # Load hyperparameters and model
    hp = get_hp([f"--resume={args.exp_id}"])
    model = setup_evaluation(hp)
    env_kwargs = hp.env_kwargs

    # ========== Setup initial graphs & graph range ==========
    graph_rng = np.random.default_rng(args.graph_seed)

    while True:
        edge_list = GENERATORS[args.graph_type](
            args.num_nodes, args.mean_degree, graph_rng
        )
        degrees = edge_list_to_degrees(args.num_nodes, edge_list)

        # Configuration model removes dependence on the generator's topology.
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

    # RewireEnv kwargs used by hard reset.
    env_kwargs["graph_types"] = [args.graph_type]
    env_kwargs["num_nodes_range"] = (args.num_nodes, args.num_nodes)
    env_kwargs["mean_degree_range"] = (args.mean_degree, args.mean_degree)
    env_kwargs["target_rho_range"] = (args.target, args.target)

    # EvalRewireEnv kwargs used by soft or no reset.
    env_kwargs["reset_method"] = args.reset_method
    env_kwargs["init_graph"] = init_graph
    env_kwargs["init_graph_type"] = args.graph_type
    env_kwargs["init_feasible_rho_range"] = init_feasible_rho_range

    if args.tolerance is not None:
        env_kwargs["tolerance"] = args.tolerance
    if args.max_rewires is not None:
        env_kwargs["max_rewires"] = args.max_rewires

    # =========== Setup environment ===========
    env_fns: list[Callable[[], RewireEnv]] = [
        lambda: ResiNetEvalRewireEnv(**env_kwargs) for _ in range(args.num_envs)
    ]
    env: GraphVecEnv
    if (args.num_envs > 1) and args.use_mult:
        env = GraphSubprocVecEnv(env_fns)
    else:
        env = GraphDummyVecEnv(env_fns)

    # =========== Sample graphs ===========
    data_list = sample_graph(model.policy, env, args.num_graphs)

    for data in data_list:
        data.graph_seed = args.graph_seed
        data.reset_method = args.reset_method
        data.max_rewires = env_kwargs["max_rewires"]

    # =========== Save results ===========
    data_dir = DATA_DIR / "ResiNet"
    data_dir.mkdir(exist_ok=True)
    file_name = f"{args.exp_id}_{args.graph_type}_N{args.num_nodes}_t{args.target}_{datetime.now().strftime('%m%d_%H%M%S_%f')}.h5"
    save(data_dir / file_name, data_list, args.store)


if __name__ == "__main__":
    main()
