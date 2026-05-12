import csv
import os

def save_sa_history_csv(history, path):
    """
    Write per-iteration statistics of Simulated Annealing to a CSV file.

    Columns: iteration, cost, best_cost, temperature
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "cost", "best_cost", "temperature"])
        for h in history:
            writer.writerow([h["iteration"], h["cost"], h["best_cost"], h["temp"]])

    print(f"Saved SA history CSV: {path}")


def save_sa_comparison_report(results, path, iterations, init_temp, cooling_rate,
                              known_optima=None):
    """
    Write a detailed report of Simulated Annealing results for multiple TSP instances.

    Parameters
    ----------
    results      : dict {instance_name: {"history": hist, "best_path": path, "best_cost": cost}}
    path         : str, output file path
    iterations   : int, total number of iterations
    init_temp    : float, starting temperature
    cooling_rate : float, cooling schedule factor
    known_optima : dict {instance_name: optimal_cost}, optional
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("  SIMULATED ANNEALING TSP REPORT")
    lines.append(f"  Iterations: {iterations} | Initial Temp: {init_temp} | Cooling: {cooling_rate}")
    lines.append("=" * 70)

    for name, data in results.items():
        best_cost = data["best_cost"]
        init_cost = data["history"][0]["cost"]
        improvement = ((init_cost - best_cost) / init_cost) * 100

        lines.append("")
        lines.append(f"  INSTANCE: {name.upper()}")
        lines.append(f"  {'Property':25s} {'Value':>20s}")
        lines.append("  " + "-" * 46)
        lines.append(f"  {'Initial Cost':25s} {init_cost:>20.4f}")
        lines.append(f"  {'Best Cost Found':25s} {best_cost:>20.4f}")
        lines.append(f"  {'SA Improvement':25s} {improvement:>19.2f}%")

        if known_optima and name in known_optima:
            optimal = known_optima[name]
            gap = (best_cost - optimal) / optimal * 100
            lines.append(f"  {'Known Optimal':25s} {optimal:>20}")
            lines.append(f"  {'Gap from Optimal':25s} {gap:>19.2f}%")

        lines.append(f"  {'Final Temperature':25s} {data['history'][-1]['temp']:>20.6f}")
        lines.append("  " + "-" * 46)

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved SA comparison report: {path}")