import numpy as np
import time


def _calculate_distance_matrix(coords):
    """
    Compute the Euclidean distance matrix for a set of coordinates.

    Parameters
    ----------
    coords : ndarray
        An (N, 2) array of (x, y) coordinates for N cities.

    Returns
    -------
    dist_matrix : ndarray
        An (N, N) symmetric matrix where dist_matrix[i, j] is the
        distance between city i and city j.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def greedy_tsp_algorithm(coords, num_starts=None):
    """
    Greedy Nearest Neighbor Algorithm for the Traveling Salesman Problem.

    The algorithm starts from a given city and always moves to the nearest
    unvisited city until all cities are visited, then returns to the start.

    Parameters
    ----------
    coords     : ndarray, (N, 2) array of city coordinates
    num_starts : int or None
        Number of different starting cities to evaluate.
        None (default) = try every city — maximises solution quality.

    Returns
    -------
    results : list of dicts, one per starting city:
              {"start_node": int, "path": list, "cost": float, "time": float}
    """
    dist_matrix = _calculate_distance_matrix(coords)
    num_nodes = len(coords)
    actual_starts = num_nodes if num_starts is None else min(num_starts, num_nodes)

    results = []

    for start_node in range(actual_starts):
        start_time = time.perf_counter()

        visited = np.zeros(num_nodes, dtype=bool)
        visited[start_node] = True
        path = [start_node]
        total_cost = 0.0
        current_node = start_node

        for _ in range(num_nodes - 1):
            row = np.where(visited, np.inf, dist_matrix[current_node])
            nearest_node = int(np.argmin(row))
            total_cost += dist_matrix[current_node, nearest_node]
            visited[nearest_node] = True
            path.append(nearest_node)
            current_node = nearest_node

        total_cost += dist_matrix[current_node, start_node]
        path.append(start_node)

        end_time = time.perf_counter()

        results.append({
            "start_node": start_node,
            "path": path,
            "cost": total_cost,
            "time": end_time - start_time,
        })

    return results
