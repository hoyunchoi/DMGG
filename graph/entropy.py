import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.optimize import minimize


@njit(fastmath=True)
def _softplus(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable implementation of softplus function"""
    return np.logaddexp(0.0, values)


@njit(fastmath=True)
def _sigmoid(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable implementation of sigmoid function"""
    return 1.0 / (1.0 + np.exp(-values))


@njit(fastmath=True)
def _get_free_energy(
    alpha: npt.NDArray[np.float64],
    degrees: npt.NDArray[np.float64],
    upper_indices: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> np.float64:
    """
    Compute free energy
    F(a) = sum_{i<j} log(1 + exp(alpha_i + alpha_j)) - sum_i k_i alpha_i
         = sum_{i<j} softplus(alpha_i + alpha_j) - sum_i k_i alpha_i

    Args
        alpha: [N, ], alpha (Lagrange multipliers), target variable
        degrees: [N, ], degrees of nodes
        upper_indices: [N, 2], upper triangle indices
    """
    row_indices, col_indices = upper_indices

    return _softplus(alpha[row_indices] + alpha[col_indices]).sum() - np.dot(
        degrees, alpha
    )


@njit(fastmath=True)
def _get_gradient(
    alpha: npt.NDArray[np.float64],
    degrees: npt.NDArray[np.float64],
    upper_indices: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> npt.NDArray[np.float64]:
    """
    Compute gradient
    g(a) = sum_{i<j} sigmoid(alpha_i + alpha_j) - k_i
    """
    row_indices, col_indices = upper_indices
    sig = _sigmoid(alpha[row_indices] + alpha[col_indices])

    grad = -degrees.copy()
    for pair_idx in range(len(row_indices)):
        grad[row_indices[pair_idx]] += sig[pair_idx]
        grad[col_indices[pair_idx]] += sig[pair_idx]

    return grad


def get_max_entropy_adjacency_matrix(
    degrees: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """
    Compute maximum entropy adjacency matrix
    """
    num_nodes = len(degrees)
    upper_indices = np.triu_indices(num_nodes, k=1)

    result = minimize(
        fun=_get_free_energy,
        x0=np.zeros(num_nodes, dtype=np.float64),  # initial alpha
        args=(degrees.astype(np.float64), upper_indices),
        jac=_get_gradient,
        method="L-BFGS-B",
        options={"gtol": 1e-8},
    )

    alpha = result.x

    adjacency_matrix = _sigmoid(alpha[:, None] + alpha[None, :])
    np.fill_diagonal(adjacency_matrix, 0.0)
    return adjacency_matrix


@njit(fastmath=True)
def get_entropy(adjacency_matrix: npt.NDArray[np.float64], num_edges: int) -> float:
    num_nodes = len(adjacency_matrix)
    entropy = 0.0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            value = adjacency_matrix[i, j]
            if value <= 0.0 or value >= 1.0:
                continue
            entropy -= value * np.log(value) + (1.0 - value) * np.log(1.0 - value)

    return 2.0 * entropy / num_edges
