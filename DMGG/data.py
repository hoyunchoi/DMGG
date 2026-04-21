from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from DMGG.type_aliases import RESET_METHOD, STORE


@dataclass(slots=True)
class DMGGData:
    # Initial graph configuration
    graph_type: str
    num_nodes: int
    num_edges: int
    mean_degree: float
    graph_seed: int = field(init=False)

    # Sampling configuration
    reset_method: RESET_METHOD = field(init=False)
    max_rewires: int = field(init=False)
    target_assortativity: float

    # Sampled results
    is_success: bool
    edge_list: npt.NDArray[np.uint32]
    assortativity_history: npt.NDArray[np.float32]
    joint_degree_matrix_history: npt.NDArray[np.uint32]
    joint_degree_flux_history: npt.NDArray[np.float32]
    clustering_coefficient: np.float32
    runtime: float


def save(file: Path, data_list: list[DMGGData], store: list[STORE]) -> None:
    """

    Structure:
    - /metadata: DataFrame with all metadata
    - /history/assortativity_0, ...
    - /edge_list/edge_list_0, ...
    - /joint_degree/joint_degree_matrix_0, joint_degree_flux_0, ...
    """
    with h5py.File(file, "w") as f:
        # ======================= Metadata =======================
        if "metadata" in store:
            metadata_group = f.create_group("metadata")

            # Initial graph configuration
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
                "mean_degree", data=np.array([d.mean_degree for d in data_list])
            )
            metadata_group.create_dataset(
                "graph_seed", data=np.array([d.graph_seed for d in data_list])
            )

            # Sampling configuration
            metadata_group.create_dataset(
                "reset_method",
                data=np.array([d.reset_method.encode("utf-8") for d in data_list]),
            )
            metadata_group.create_dataset(
                "max_rewires", data=np.array([d.max_rewires for d in data_list])
            )
            metadata_group.create_dataset(
                "target_assortativity",
                data=np.array([d.target_assortativity for d in data_list]),
            )

            # Sampled results
            metadata_group.create_dataset(
                "is_success",
                data=np.array([d.is_success for d in data_list], dtype=np.bool_),
            )
            metadata_group.create_dataset(
                "num_steps",
                data=np.array([len(d.assortativity_history) for d in data_list]),
            )
            metadata_group.create_dataset(
                "assortativity",
                data=np.array([d.assortativity_history[-1] for d in data_list]),
            )
            metadata_group.create_dataset(
                "clustering_coefficient",
                data=np.array([d.clustering_coefficient for d in data_list]),
            )
            metadata_group.create_dataset(
                "runtime", data=np.array([d.runtime for d in data_list])
            )

        # ======================= History =======================
        if "history" in store:
            history_group = f.create_group("history")
            for i, data in enumerate(data_list):
                history_group.create_dataset(
                    f"assortativity_{i}", data=data.assortativity_history
                )

        # ======================= Edge list =======================
        if "edge_list" in store:
            edge_list_group = f.create_group("edge_list")
            for i, data in enumerate(data_list):
                edge_list_group.create_dataset(f"edge_list_{i}", data=data.edge_list)

        # ======================= Joint degree matrix and flux =======================
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
    """Load all metadata as a pandas DataFrame."""
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
    """Loads all histories from the HDF5 file"""
    with h5py.File(file, "r", driver="core") as f:
        history_group = cast(h5py.Group, f["history"])
        histories: dict[int, npt.NDArray[np.float32]] = {}

        # key: "assortativity_0", "assortativity_1", ...
        for key, value in history_group.items():
            idx = int(key.rpartition("_")[2])
            histories[idx] = value[()]

        return [histories[i] for i in range(len(histories))]


def load_edge_lists(file: Path) -> list[npt.NDArray[np.uint32]]:
    """Loads all edge lists from the HDF5 file"""
    with h5py.File(file, "r", driver="core") as f:
        edge_list_group = cast(h5py.Group, f["edge_list"])

        # To avoid ordering issues, temporarily store the values in a dictionary keyed by index
        edge_list_dict: dict[int, npt.NDArray[np.uint32]] = {}

        # key: "edge_list_0", "edge_list_1", ...
        for key, value in edge_list_group.items():
            idx = int(key.rpartition("_")[2])
            edge_list_dict[idx] = value[()]

        # Reassemble the lists in the order of indices
        n_samples = len(edge_list_dict)
        return [edge_list_dict[i] for i in range(n_samples)]


def load_joint_degree(
    file: Path,
) -> tuple[list[npt.NDArray[np.uint32]], list[npt.NDArray[np.float32]]]:
    """Loads all joint degree matrix/flux histories from the HDF5 file."""
    with h5py.File(file, "r", driver="core") as f:
        joint_degree_group = cast(h5py.Group, f["joint_degree"])

        # To avoid ordering issues, temporarily store the values in a dictionary keyed by index
        matrix_dict: dict[int, npt.NDArray[np.uint32]] = {}
        flux_dict: dict[int, npt.NDArray[np.float32]] = {}

        for key, value in joint_degree_group.items():
            # key: "joint_degree_matrix_0", "joint_degree_flux_0", ...
            if key.startswith("joint_degree_matrix_"):
                idx = int(key.rpartition("_")[2])
                matrix_dict[idx] = value[()]

            elif key.startswith("joint_degree_flux_"):
                idx = int(key.rpartition("_")[2])
                flux_dict[idx] = value[()]

        # Reassemble the lists in the order of indices
        n_samples = len(matrix_dict)
        joint_degree_matrix = [matrix_dict[i] for i in range(n_samples)]
        joint_degree_flux = [flux_dict[i] for i in range(n_samples)]

        return joint_degree_matrix, joint_degree_flux
