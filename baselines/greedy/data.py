from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from ERGM.type_aliases import RESET_METHOD, STORE


@dataclass(slots=True)
class GreedyData:
    # Initial graph configuration
    graph_type: str
    num_nodes: int
    num_edges: int
    graph_seed: int
    reset_method: RESET_METHOD

    # Sampling configuration
    seed: int
    max_steps: int
    target: float
    tolerance: float

    # Sampling results
    physical_tolerance: float
    is_success: bool
    num_steps: int
    edge_list: npt.NDArray[np.uint32]
    assortativity_history: npt.NDArray[np.float32]
    joint_degree_matrix_history: npt.NDArray[np.uint32]
    joint_degree_flux_history: npt.NDArray[np.float32]
    runtime: float



def save(file: Path, data_list: list[GreedyData], store: list[STORE]) -> None:
    """Save requested histories and final edge lists in a single HDF5 file."""
    with h5py.File(file, "w") as f:
        if "metadata" in store:
            metadata_group = f.create_group("metadata")

            metadata_group.create_dataset(
                "graph_type",
                data=np.array([d.graph_type.encode("utf-8") for d in data_list]),
            )
            metadata_group.create_dataset(
                "num_nodes", data=np.array([d.num_nodes for d in data_list])
            )
            metadata_group.create_dataset(
                "num_edges", data=np.array([d.num_edges for d in data_list])
            )
            metadata_group.create_dataset(
                "graph_seed", data=np.array([d.graph_seed for d in data_list])
            )
            metadata_group.create_dataset(
                "reset_method",
                data=np.array([d.reset_method.encode("utf-8") for d in data_list]),
            )

            metadata_group.create_dataset(
                "seed", data=np.array([d.seed for d in data_list])
            )
            metadata_group.create_dataset(
                "max_steps", data=np.array([d.max_steps for d in data_list])
            )
            metadata_group.create_dataset(
                "target_assortativity",
                data=np.array([d.target for d in data_list]),
            )
            metadata_group.create_dataset(
                "tolerance", data=np.array([d.tolerance for d in data_list])
            )

            metadata_group.create_dataset(
                "physical_tolerance",
                data=np.array([d.physical_tolerance for d in data_list]),
            )
            metadata_group.create_dataset(
                "is_success",
                data=np.array([d.is_success for d in data_list], dtype=np.bool_),
            )
            metadata_group.create_dataset(
                "num_steps", data=np.array([d.num_steps for d in data_list])
            )
            metadata_group.create_dataset(
                "runtime", data=np.array([d.runtime for d in data_list])
            )

        if "history" in store:
            history_group = f.create_group("history")
            for i, data in enumerate(data_list):
                history_group.create_dataset(
                    f"assortativity_{i}", data=data.assortativity_history
                )

        if "edge_list" in store:
            edge_list_group = f.create_group("edge_list")
            for i, data in enumerate(data_list):
                edge_list_group.create_dataset(f"edge_list_{i}", data=data.edge_list)

        if "joint_degree" in store:
            joint_degree_group = f.create_group("joint_degree")
            for i, data in enumerate(data_list):
                joint_degree_group.create_dataset(
                    f"joint_degree_matrix_{i}", data=data.joint_degree_matrix_history
                )
                joint_degree_group.create_dataset(
                    f"joint_degree_flux_{i}", data=data.joint_degree_flux_history
                )


def load_metadata(file: Path) -> pd.DataFrame:
    """Load metadata as a pandas DataFrame."""
    with h5py.File(file, "r") as f:
        metadata_group = cast(h5py.Group, f["metadata"])
        metadata = {}

        for key, value in metadata_group.items():
            arr = value[()]
            if h5py.check_string_dtype(value.dtype):
                arr = arr.astype(str)
            metadata[key] = arr
        return pd.DataFrame(metadata)


def load_histories(file: Path) -> list[npt.NDArray[np.float32]]:
    """Load assortativity histories."""
    with h5py.File(file, "r", driver="core") as f:
        history_group = cast(h5py.Group, f["history"])
        assortativity_dict: dict[int, npt.NDArray[np.float32]] = {}

        for key, value in history_group.items():
            if key.startswith("assortativity_"):
                idx = int(key.rpartition("_")[2])
                assortativity_dict[idx] = value[()]
        num_samples = len(assortativity_dict)
        assortativity = [assortativity_dict[i] for i in range(num_samples)]
        return assortativity


def load_edge_lists(file: Path) -> list[npt.NDArray[np.uint32]]:
    """Load final edge lists."""
    with h5py.File(file, "r", driver="core") as f:
        edge_list_group = cast(h5py.Group, f["edge_list"])
        edge_list_dict: dict[int, npt.NDArray[np.uint32]] = {}

        for key, value in edge_list_group.items():
            idx = int(key.rpartition("_")[2])
            edge_list_dict[idx] = value[()]

        return [edge_list_dict[i] for i in range(len(edge_list_dict))]


def load_joint_degree(
    file: Path,
) -> tuple[list[npt.NDArray[np.uint32]], list[npt.NDArray[np.float32]]]:
    """Load all joint_degree matrix and flux histories."""
    with h5py.File(file, "r", driver="core") as f:
        joint_degree_group = cast(h5py.Group, f["joint_degree"])
        matrix_dict: dict[int, npt.NDArray[np.uint32]] = {}
        flux_dict: dict[int, npt.NDArray[np.float32]] = {}

        for key, value in joint_degree_group.items():
            idx = int(key.rpartition("_")[2])
            if key.startswith("joint_degree_matrix_"):
                matrix_dict[idx] = value[()]
            elif key.startswith("joint_degree_flux_"):
                flux_dict[idx] = value[()]

        num_samples = len(matrix_dict)
        joint_degree_matrix = [matrix_dict[i] for i in range(num_samples)]
        joint_degree_flux = [flux_dict[i] for i in range(num_samples)]
        return joint_degree_matrix, joint_degree_flux
