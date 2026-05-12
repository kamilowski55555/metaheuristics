import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tsp_solution(coords, path, title="Greedy TSP Solution", save_path=None):
    """
    Generate a 2D plot of the TSP solution showing cities and the path.

    Parameters
    ----------
    coords    : ndarray
        An (N, 2) array of city coordinates (x, y).
    path      : list
        The sequence of city indices representing the tour.
    title     : str
        The title of the plot.
    save_path : str, optional
        Path to save the generated image. If None, the plot is shown.
    """
    plt.figure(figsize=(10, 7))

    path_coords = coords[path]

    plt.plot(path_coords[:, 0], path_coords[:, 1],
             color='steelblue', linestyle='-', linewidth=1.5, alpha=0.7,
             label='Tour Path', zorder=1)

    plt.scatter(coords[:, 0], coords[:, 1],
                color='crimson', s=30, edgecolors='black',
                label='Cities', zorder=2)

    start_city = coords[path[0]]
    plt.scatter(start_city[0], start_city[1],
                color='lime', s=100, edgecolors='black',
                label='Start City', zorder=3)

    plt.title(title, fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved TSP plot: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_multiple_starts(coords, results, instance_name, out_dir):
    """
    Save plots for all provided greedy solutions (typically 5).

    Parameters
    ----------
    coords        : ndarray
        The (N, 2) array of coordinates.
    results       : list of dicts
        The output from the greedy algorithm for different starts.
    instance_name : str
        The name of the TSP instance (for filename and title).
    out_dir       : str
        Directory to save the plots.
    """
    for res in results:
        start_node = res['start_node']
        filename = f"greedy_{instance_name}_start_{start_node}.png"
        save_path = os.path.join(out_dir, filename)

        plot_tsp_solution(
            coords,
            res['path'],
            title=f"Greedy TSP: {instance_name} (Start Node: {start_node})",
            save_path=save_path
        )