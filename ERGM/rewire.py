import numpy as np
import numpy.typing as npt

from graph.type_aliases import EDGE, NODE


def rewire(
    edge_indices: tuple[int, int],
    new_edges: tuple[EDGE, EDGE],
    edge_list: npt.NDArray[np.uint32],
    adjacency_list: list[set[NODE]],
    degrees: npt.NDArray[np.uint32],
    joint_degree_matrix: npt.NDArray[np.uint32] | None = None,
) -> None:
    """
    Rewire the edges and change graph properties accordingly.

    Args:
        edge_indices: index of old_edge1, old_edge2 in edge_list
        new_edges: new_edge1, new_edge2. will replace old_edges
        edge_list: [E, 2] edge list, with sorted nodes (node1, node2) while node1 < node2
        adjacency_list: [N, ] adjacency list, where adjacency_list[node] is set of neighbors of node
        degrees: [N, ] degrees of each node
        joint_degree_matrix: [D, D]
    """
    old_edge1, old_edge2 = edge_list[edge_indices[0]], edge_list[edge_indices[1]]
    new_edge1, new_edge2 = new_edges
    edge1_idx, edge2_idx = edge_indices

    (node1, node2), (node3, node4) = old_edge1, old_edge2
    (new_node1, new_node2), (new_node3, new_node4) = new_edge1, new_edge2

    degree1, degree2 = degrees[node1], degrees[node2]
    degree3, degree4 = degrees[node3], degrees[node4]
    new_degree1, new_degree2 = degrees[new_node1], degrees[new_node2]
    new_degree3, new_degree4 = degrees[new_node3], degrees[new_node4]

    # ---------------- Replace edge list  ----------------
    edge_list[edge1_idx] = np.array(new_edge1, dtype=np.uint32)
    edge_list[edge2_idx] = np.array(new_edge2, dtype=np.uint32)

    # ---------------- Adjacency list ----------------
    adjacency_list[node1].remove(node2)
    adjacency_list[node2].remove(node1)
    adjacency_list[node3].remove(node4)
    adjacency_list[node4].remove(node3)
    adjacency_list[new_node1].add(new_node2)
    adjacency_list[new_node2].add(new_node1)
    adjacency_list[new_node3].add(new_node4)
    adjacency_list[new_node4].add(new_node3)

    # ---------------- Mixing matrix ----------------
    if joint_degree_matrix is not None:
        joint_degree_matrix[degree1 - 1, degree2 - 1] -= 1
        joint_degree_matrix[degree2 - 1, degree1 - 1] -= 1
        joint_degree_matrix[degree3 - 1, degree4 - 1] -= 1
        joint_degree_matrix[degree4 - 1, degree3 - 1] -= 1
        joint_degree_matrix[new_degree1 - 1, new_degree2 - 1] += 1
        joint_degree_matrix[new_degree2 - 1, new_degree1 - 1] += 1
        joint_degree_matrix[new_degree3 - 1, new_degree4 - 1] += 1
        joint_degree_matrix[new_degree4 - 1, new_degree3 - 1] += 1
