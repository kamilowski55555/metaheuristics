import numpy as np
import time


def get_operators():
    """Returns a list of functions representing 4 different route operators."""

    def swap(path):
        """Swaps two random cities in the path."""
        new_path = path.copy()
        idx1, idx2 = np.random.choice(len(path) - 1, 2, replace=False) + 1
        new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        return list(new_path)

    def inversion(path):
        """Inverts a random sub-segment of the path."""
        new_path = path.copy()
        idx1, idx2 = sorted(np.random.choice(len(path) - 1, 2, replace=False) + 1)
        new_path[idx1:idx2] = new_path[idx1:idx2][::-1]
        return list(new_path)

    def insertion(path):
        """Removes a city and inserts it at a random position."""
        new_path = list(path)
        city = new_path.pop(np.random.randint(1, len(path) - 1))
        new_path.insert(np.random.randint(1, len(path) - 1), city)
        return new_path

    def scramble(path):
        """Scrambles a small sub-segment of the path."""
        new_path = path.copy()
        idx1, idx2 = sorted(np.random.choice(len(path) - 1, 2, replace=False) + 1)
        segment = new_path[idx1:idx2]
        np.random.shuffle(segment)
        new_path[idx1:idx2] = segment
        return list(new_path)

    return [swap, inversion, insertion, scramble]


def simulated_annealing(coords, dist_matrix,
                        temp=1000, cooling_rate=0.995,
                        iterations=5000, seed=None):
    """
    Simulated Annealing for TSP with 4 different movement operators.

    Parameters
    ----------
    coords       : ndarray, city coordinates
    dist_matrix  : ndarray, distance matrix
    temp         : float, initial temperature
    cooling_rate : float, rate at which temperature decreases
    iterations   : int, number of iterations per instance

    Returns
    -------
    history : list of dicts, including best cost and path at steps
    """
    if seed is not None: np.random.seed(seed)

    operators = get_operators()
    n = len(coords)

    current_path = list(range(n))
    np.random.shuffle(current_path)
    current_path.append(current_path[0])  # close cycle

    def calc_cost(p):
        a = np.array(p)
        return dist_matrix[a[:-1], a[1:]].sum()

    current_cost = calc_cost(current_path)
    best_path = current_path[:]
    best_cost = current_cost

    history = []

    for i in range(iterations):
        op = np.random.choice(operators)
        candidate_path = op(current_path[:-1])
        candidate_path.append(candidate_path[0])  # close cycle
        candidate_cost = calc_cost(candidate_path)

        diff = candidate_cost - current_cost
        if diff < 0 or np.random.rand() < np.exp(-diff / temp):
            current_path = candidate_path
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_path = current_path[:]

        if i % (iterations // 50) == 0:
            history.append({
                "iteration": i,
                "cost": current_cost,
                "best_cost": best_cost,
                "path": current_path[:],
                "temp": temp
            })

        temp *= cooling_rate

    return history, best_path, best_cost