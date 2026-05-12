"""
Zadanie 2 – Algorytm zachłanny (Nearest Neighbor)
Inteligencja obliczeniowa, dr Przemysław Juszczuk

Autor: Kamil Kot, Dawid Wieczorek

================================================================================
PSEUDOKOD – Algorytm Najbliższego Sąsiada (Nearest Neighbor)
================================================================================

WEJŚCIE: Macierz odległości D, liczba miast N, węzeł startowy S

1. Inicjalizacja:
   path = [S]
   visited = {S}
   current_node = S
   total_cost = 0

2. Dopóki len(visited) < N:
   a) Znajdź nieodwiedzony węzeł V, dla którego D[current_node][V] jest najmniejsze
   b) Dodaj V do path
   c) Oznacz V jako odwiedzony
   d) total_cost += D[current_node][V]
   e) current_node = V

3. Powrót:
   total_cost += D[current_node][S]
   path.append(S)

WYJŚCIE: path, total_cost
================================================================================
"""

import os
import time
from data_loader import load_tsp_file
from tsp import greedy_tsp_algorithm
from report import save_greedy_report, save_tsp_history_csv
from visualization import plot_multiple_starts

# CONFIG
INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tsplib")
OUT_DIR = "results"

# Instances with 300+ cities - to be replaced if necessary
TSP_INSTANCES = ["lin318.tsp", "rd400.tsp", "pcb442.tsp", "u574.tsp", "rat575.tsp"]

# None = try every city as a starting point
# Set to an integer to cap it
NUM_STARTS = None


def main():
    print("=" * 60)
    print("Starting Task 2: Greedy TSP Analysis")
    print("=" * 60)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for instance_name in TSP_INSTANCES:
        file_path = os.path.join(INPUT_DIR, instance_name)

        print(f"\n>>> PROCESSING INSTANCE: {instance_name}")

        try:
            coords, info = load_tsp_file(file_path)
            print(f"Loaded {len(coords)} cities.")
        except Exception as e:
            print(f"Error loading {instance_name}: {e}")
            continue

        n_starts_label = "all" if NUM_STARTS is None else NUM_STARTS
        print(f"Running Nearest Neighbor from {n_starts_label} starting nodes...")
        results = greedy_tsp_algorithm(coords, num_starts=NUM_STARTS)

        instance_slug = instance_name.replace(".tsp", "")

        csv_path = os.path.join(OUT_DIR, f"results_{instance_slug}.csv")
        save_tsp_history_csv(results, csv_path)

        report_path = os.path.join(OUT_DIR, f"report_{instance_slug}.txt")
        save_greedy_report(results, info, report_path)

        print("Generating plots...")
        top5 = sorted(results, key=lambda r: r["cost"])[:5]
        plot_multiple_starts(coords, top5, instance_slug, OUT_DIR)

    print("\n" + "=" * 60)
    print("All tasks completed. Results are in:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()