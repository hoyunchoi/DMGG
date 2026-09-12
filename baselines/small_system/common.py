from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from path import DATA_DIR


class SmallData:
    def __init__(self, num_nodes: int, mean_degree: float, graph_seed: int) -> None:
        self.num_nodes = num_nodes
        self.mean_degree = int(mean_degree)
        self.graph_seed = graph_seed

        self.data_dir = DATA_DIR / "small_system"
        self.data_dir.mkdir(exist_ok=True)

    def file_path(self, suffix: str) -> Path:
        return self.data_dir / f"{self.prefix}_{suffix}.pkl"

    @property
    def prefix(self) -> str:
        return f"N{self.num_nodes}_k{self.mean_degree}_s{self.graph_seed}"


@njit
def K_to_rho(K_value: int, degrees: npt.NDArray[np.uint32]) -> np.float64:
    """
    convert K=sum_(i, j in E) d_i d_j to assortativity rho
    """
    num_edges = degrees.sum() // 2

    edge_end_mean = (degrees**2).sum() / (2.0 * num_edges)
    edge_end_variance = (degrees**3.0).sum() / (2.0 * num_edges) - edge_end_mean**2.0

    return (K_value / num_edges - edge_end_mean**2.0) / edge_end_variance


@njit
def rho_to_K(assortativity: float, degrees: npt.NDArray[np.uint32]) -> int:
    """
    inverse of K_to_rho
    """
    num_edges = degrees.sum() // 2

    edge_end_mean = (degrees**2).sum() / (2.0 * num_edges)
    edge_end_variance = (degrees**3.0).sum() / (2.0 * num_edges) - edge_end_mean**2.0

    return round(num_edges * (edge_end_mean**2 + assortativity * edge_end_variance))
