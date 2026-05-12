import csv
import os


def save_tsp_history_csv(results, path):
    """
    Write results of the 5 greedy starts to a CSV file.

    Columns: start_node, cost, execution_time
    Each row represents a different starting city.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_node", "cost", "execution_time"])
        for res in results:
            writer.writerow([res["start_node"], res["cost"], res["time"]])

    print(f"Saved TSP results CSV: {path}")


def save_greedy_report(results, info, path):
    """
    Write a detailed report of the Greedy TSP execution to a text file.

    Parameters
    ----------
    results : list of dicts
        The output from greedy_tsp_algorithm (5 different starts).
    info    : dict
        Metadata from the .tsp file (name, dimension, etc.).
    path    : str
        Output file path for the report.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Sort results by cost to identify the best starting city
    best_run = min(results, key=lambda x: x["cost"])
    worst_run = max(results, key=lambda x: x["cost"])
    avg_cost = sum(r["cost"] for r in results) / len(results)

    lines = []
    lines.append("=" * 70)
    lines.append("  GREEDY TSP ALGORITHM REPORT (Nearest Neighbor)")
    lines.append(f"  Instance Name: {info.get('name', 'Unknown').upper()}")
    lines.append(f"  Dimension:     {info.get('dimension', 'N/A')} cities")
    lines.append("=" * 70)

    lines.append("")
    lines.append(f"  {'Start Node':<12} {'Cost (Distance)':<18} {'Time (s)':<12}")
    lines.append("  " + "-" * 42)

    for res in results:
        lines.append(f"  {res['start_node']:<12} {res['cost']:<18.4f} {res['time']:<12.6f}")

    lines.append("  " + "-" * 42)
    lines.append("")
    lines.append(f"  Best Start Node:  {best_run['start_node']}")
    lines.append(f"  Best Cost:        {best_run['cost']:.4f}")
    lines.append(f"  Worst Cost:       {worst_run['cost']:.4f}")
    lines.append(f"  Average Cost:     {avg_cost:.4f}")
    lines.append("")
    lines.append(f"  Best Path (First 10 nodes): {str(best_run['path'][:10])}...")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved Greedy TSP report: {path}")
    print()
    print(text)