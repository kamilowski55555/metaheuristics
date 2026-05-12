import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os


def plot_tsp_solution(coords, path, title="TSP Solution", save_path=None):
    """
    Generate a 2D plot of the TSP solution showing cities and the path.
    (Static version using Matplotlib only)
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


def plot_sa_step_simulation(coords, history, title="SA Step Simulation", save_path=None):
    """
    Generate an interactive Plotly step-by-step simulation of the SA process.
    Each slider step shows the tour state at a recorded checkpoint.

    Parameters
    ----------
    coords    : ndarray, city coordinates (N x 2)
    history   : list of dicts with keys: iteration, cost, best_cost, path, temp
    title     : str, plot title
    save_path : str, path for output .html file
    """
    city_x = coords[:, 0].tolist()
    city_y = coords[:, 1].tolist()

    def _tour_xy(path):
        xs = [coords[c, 0] for c in path]
        ys = [coords[c, 1] for c in path]
        return xs, ys

    frames = []
    for h in history:
        px, py = _tour_xy(h["path"])
        frames.append(go.Frame(
            data=[
                go.Scatter(x=px, y=py, mode="lines",
                           line=dict(color="steelblue", width=1), name="Tour"),
                go.Scatter(x=city_x, y=city_y, mode="markers",
                           marker=dict(color="crimson", size=4), name="Cities"),
            ],
            name=str(h["iteration"]),
            layout=go.Layout(title_text=(
                f"{title} | Iter: {h['iteration']} | "
                f"Cost: {h['cost']:.1f} | Best: {h['best_cost']:.1f} | "
                f"T: {h['temp']:.4f}"
            )),
        ))

    init_px, init_py = _tour_xy(history[0]["path"])
    fig = go.Figure(
        data=[
            go.Scatter(x=init_px, y=init_py, mode="lines",
                       line=dict(color="steelblue", width=1), name="Tour"),
            go.Scatter(x=city_x, y=city_y, mode="markers",
                       marker=dict(color="crimson", size=4), name="Cities"),
        ],
        frames=frames,
    )

    slider_steps = [
        dict(
            method="animate",
            args=[[str(h["iteration"])],
                  {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            label=str(h["iteration"]),
        )
        for h in history
    ]

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 150, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            steps=slider_steps,
            active=0,
            currentvalue={"prefix": "Iteration: "},
            pad={"t": 50},
        )],
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved SA step simulation: {save_path}")
    else:
        fig.show()