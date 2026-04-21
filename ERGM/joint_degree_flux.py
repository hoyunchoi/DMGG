import numpy as np
import numpy.typing as npt
from numba import njit


@njit(fastmath=True)
def get_joint_degree_flux(
    num_edges: int,
    edge_list: npt.NDArray[np.uint32],
    degrees: npt.NDArray[np.uint32],
    adjacency_matrix: npt.NDArray[np.bool_],
    assortativity_weight: float,
) -> npt.NDArray[np.float32]:
    max_degree = degrees.max()

    # Initialize joint degree flux
    joint_degree_flux = np.zeros((max_degree, max_degree), dtype=np.float32)
    num_valid_pairs = 0

    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            # Get edge information
            node1 = edge_list[i, 0]
            node2 = edge_list[i, 1]
            node3 = edge_list[j, 0]
            node4 = edge_list[j, 1]

            # Check for intersection (skip if edges share a node)
            if (node1 in [node3, node4]) or (node2 in [node3, node4]):
                continue

            # Adjacency check (using adjacency_matrix)
            # In Numba, accessing a boolean matrix is much faster than list[set]
            mask_mode0 = False
            mask_mode1 = False

            if adjacency_matrix[node1, node3] or adjacency_matrix[node2, node4]:
                mask_mode0 = True

            if adjacency_matrix[node1, node4] or adjacency_matrix[node2, node3]:
                mask_mode1 = True

            if mask_mode0 and mask_mode1:
                continue

            num_valid_pairs += 1
            degree1, degree2 = int(degrees[node1]), int(degrees[node2])
            degree3, degree4 = int(degrees[node3]), int(degrees[node4])

            # Calculate probabilities for modes
            if not mask_mode0 and not mask_mode1:
                prob_mode0, prob_mode1 = 0.5, 0.5
            elif not mask_mode0:
                prob_mode0, prob_mode1 = 1.0, 0.0
            else:
                prob_mode0, prob_mode1 = 0.0, 1.0

            # Mode 0: (1, 2), (3, 4) -> (1, 3), (2, 4)
            if prob_mode0 > 0:
                delta_s = 2.0 * (
                    (degree1 * degree2 + degree3 * degree4)
                    - (degree1 * degree3 + degree2 * degree4)
                )
                delta_energy = assortativity_weight * delta_s
                # Using np.exp here, Numba will optimize
                acceptance_prob = min(1.0, np.exp(-delta_energy))

                weight = prob_mode0 * acceptance_prob

                # Remove: (1, 2), (3, 4)
                joint_degree_flux[degree1 - 1, degree2 - 1] -= weight
                joint_degree_flux[degree2 - 1, degree1 - 1] -= weight
                joint_degree_flux[degree3 - 1, degree4 - 1] -= weight
                joint_degree_flux[degree4 - 1, degree3 - 1] -= weight

                # Add: (1, 3), (2, 4)
                joint_degree_flux[degree1 - 1, degree3 - 1] += weight
                joint_degree_flux[degree3 - 1, degree1 - 1] += weight
                joint_degree_flux[degree2 - 1, degree4 - 1] += weight
                joint_degree_flux[degree4 - 1, degree2 - 1] += weight

            # Mode 1: (1, 2), (3, 4) -> (1, 4), (2, 3)
            if prob_mode1 > 0:
                delta_s = 2.0 * (
                    (degree1 * degree2 + degree3 * degree4)
                    - (degree1 * degree4 + degree2 * degree3)
                )
                delta_energy = assortativity_weight * delta_s
                acceptance_prob = min(1.0, np.exp(-delta_energy))

                weight = prob_mode1 * acceptance_prob

                # Remove: (1, 2), (3, 4)
                joint_degree_flux[degree1 - 1, degree2 - 1] -= weight
                joint_degree_flux[degree2 - 1, degree1 - 1] -= weight
                joint_degree_flux[degree3 - 1, degree4 - 1] -= weight
                joint_degree_flux[degree4 - 1, degree3 - 1] -= weight

                # Add: (1, 4), (2, 3)
                joint_degree_flux[degree1 - 1, degree4 - 1] += weight
                joint_degree_flux[degree4 - 1, degree1 - 1] += weight
                joint_degree_flux[degree2 - 1, degree3 - 1] += weight
                joint_degree_flux[degree3 - 1, degree2 - 1] += weight

    return joint_degree_flux / np.float32(num_valid_pairs)
