import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Convergence plot (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(history, title="Convergence", save_path=None):
    """
    Line chart of best / mean / worst fitness per generation.
    Works for any dimensionality — it only uses the scalar fitness values.
    """
    generations = [h["generation"] for h in history]
    best  = [h["best"]  for h in history]
    mean  = [h["mean"]  for h in history]
    worst = [h["worst"] for h in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(generations, best,  label="Best",  color="green", linewidth=2)
    ax.plot(generations, mean,  label="Mean",  color="steelblue", linewidth=1.5)
    ax.plot(generations, worst, label="Worst", color="salmon",  linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower = better)")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("symlog")   # log scale makes early drop visible
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved convergence plot: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3-D step-by-step simulation (plotly, dim=3 only)
# ─────────────────────────────────────────────────────────────────────────────

def _make_surface(func, bounds, n_grid=60):
    """
    Build a meshgrid over (x1, x2), fix x3=0, evaluate func.
    Returns x_grid, y_grid, z_grid suitable for go.Surface.
    """
    low, high = bounds
    xs = np.linspace(low, high, n_grid)
    ys = np.linspace(low, high, n_grid)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            # Third dimension fixed at 0; this gives a 2-D slice of the landscape
            Z[i, j] = func(np.array([X[i, j], Y[i, j], 0.0]))

    return X, Y, Z


def step_simulation_3d(func, bounds, history, title="Step simulation",
                       save_path="simulation.html", n_grid=50,
                       max_frames=None):
    """
    Interactive Plotly animation with a slider for stepping through generations.

    Each frame shows:
      - The function surface (fixed)
      - The population as scatter points on the surface

    Parameters
    ----------
    func       : the benchmark function (must accept dim=3 arrays)
    bounds     : (low, high)
    history    : list of dicts from ea.py / de.py  (must have "population" key)
    save_path  : output HTML file
    n_grid     : surface resolution (lower = faster rendering)
    max_frames : cap number of frames to keep the HTML small; None = all gens
    """
    X, Y, Z = _make_surface(func, bounds, n_grid)

    surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Viridis",
        opacity=0.75,
        showscale=False,
        name="Function surface",
    )

    # ── Build one frame per generation ──────────────────────────────────────
    frames_to_use = history if max_frames is None else history[::max(1, len(history)//max_frames)]

    frames = []
    slider_steps = []

    for h in frames_to_use:
        pop = h["population"]          # shape (pop_size, 3)
        gen = h["generation"]

        # Evaluate fitness of each individual so we can colour by fitness
        fit = np.array([func(ind) for ind in pop])

        scatter = go.Scatter3d(
            x=pop[:, 0],
            y=pop[:, 1],
            z=fit,                     # height = actual fitness value
            mode="markers",
            marker=dict(
                size=4,
                color=fit,
                colorscale="Hot",
                cmin=float(np.min(Z)),
                cmax=float(np.percentile(Z, 95)),  # cap so colours don't wash out
                opacity=0.9,
            ),
            name=f"Gen {gen}",
        )

        frame = go.Frame(
            data=[surface, scatter],
            name=str(gen),
        )
        frames.append(frame)

        slider_steps.append(dict(
            args=[[str(gen)], {"frame": {"duration": 0}, "mode": "immediate"}],
            label=str(gen),
            method="animate",
        ))

    # ── Initial frame ────────────────────────────────────────────────────────
    first_pop = frames_to_use[0]["population"]
    first_fit = np.array([func(ind) for ind in first_pop])

    fig = go.Figure(
        data=[
            surface,
            go.Scatter3d(
                x=first_pop[:, 0],
                y=first_pop[:, 1],
                z=first_fit,
                mode="markers",
                marker=dict(size=4, color=first_fit, colorscale="Hot", opacity=0.9),
                name="Gen 0",
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x₁",
            yaxis_title="x₂",
            zaxis_title="f(x)",
        ),
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue={"prefix": "Generation: "},
            pad={"t": 50},
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 200}, "fromcurrent": True}],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                ),
            ],
        )],
    )

    fig.write_html(save_path)
    print(f"Saved step simulation: {save_path}")
    return fig
