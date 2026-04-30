import csv
import os


def save_history_csv(history, path):
    """
    Write per-generation statistics to a CSV file.

    Columns: generation, best, mean, worst
    One row per generation — easy to open in Excel or re-plot later.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best", "mean", "worst"])
        for h in history:
            writer.writerow([h["generation"], h["best"], h["mean"], h["worst"]])

    print(f"Saved history CSV: {path}")


def save_comparison_report(results, path, dim, generations, pop_size):
    """
    Write a side-by-side EA vs DE comparison to a plain-text file.

    Parameters
    ----------
    results : dict  {function_name: {"ea": history, "de": history}}
    path    : output file path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("  EA vs DE Comparison Report")
    lines.append(f"  Dimensions: {dim}  |  Generations: {generations}  |  Population: {pop_size}")
    lines.append("=" * 70)

    for func_name, runs in results.items():
        ea_final = runs["ea"][-1]
        de_final = runs["de"][-1]

        ea_best = ea_final["best"]
        de_best = de_final["best"]
        winner  = "DE" if de_best < ea_best else "EA" if ea_best < de_best else "TIE"

        lines.append("")
        lines.append(f"  Function: {func_name.upper()}")
        lines.append(f"  {'':20s} {'EA':>16s}  {'DE':>16s}")
        lines.append(f"  {'Best (final)':20s} {ea_best:>16.6f}  {de_best:>16.6f}  <- {winner} wins")
        lines.append(f"  {'Mean (final)':20s} {ea_final['mean']:>16.6f}  {de_final['mean']:>16.6f}")
        lines.append(f"  {'Worst (final)':20s} {ea_final['worst']:>16.6f}  {de_final['worst']:>16.6f}")

        # How many generations until each algorithm first beat 1.0 fitness?
        for algo, hist in [("EA", runs["ea"]), ("DE", runs["de"])]:
            first_good = next((h["generation"] for h in hist if h["best"] < 1.0), None)
            label = f"Gen < 1.0 ({algo})"
            value = str(first_good) if first_good is not None else "never"
            lines.append(f"  {label:20s} {value:>16s}")

        lines.append("  " + "-" * 56)

    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved comparison report: {path}")
    print()
    print(text)
