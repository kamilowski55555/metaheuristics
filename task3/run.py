"""
Zadanie 3 – Symulowane Wyżarzanie (Simulated Annealing)
Inteligencja obliczeniowa, dr Przemysław Juszczuk

Autor: Kamil Kot, Dawid Wieczorek

================================================================================
PSEUDOKOD – Symulowane Wyżarzanie dla TSP
================================================================================

WEJŚCIE: instancja TSP, T_start, cooling_rate, iterations

1. Inicjalizacja:
   current_path = losowa_permutacja(miasta)
   current_cost = oblicz_koszt(current_path)
   best_path = current_path, best_cost = current_cost
   T = T_start

2. Dla i = 0 do iterations-1:
   a) Wybierz losowo 1 z 4 operatorów (Swap, Inversion, Insertion, Scramble)
   b) candidate_path = operator(current_path)
   c) candidate_cost = oblicz_koszt(candidate_path)
   d) delta = candidate_cost - current_cost
   e) Jeśli delta < 0 LUB rand() < exp(-delta / T):
         current_path = candidate_path
         current_cost = candidate_cost
   f) Jeśli current_cost < best_cost:
         best_path = current_path, best_cost = current_cost
   g) T = T * cooling_rate (chłodzenie)

WYJŚCIE: best_path, best_cost
================================================================================
"""

import os
import numpy as np
from data_loader import load_tsp_file, _calculate_distance_matrix
from sa import simulated_annealing
from report import save_sa_history_csv, save_sa_comparison_report
from visualization import plot_tsp_solution, plot_sa_step_simulation

# CONFIG
INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tsplib")
OUT_DIR = "results"

# Instances with 300+ cities - to be replaced if necessary
TSP_INSTANCES = ["lin318.tsp", "rd400.tsp", "pcb442.tsp", "u574.tsp", "rat575.tsp"]

# Known optimal tour lengths from TSPlib (used to compute % gap)
KNOWN_OPTIMA = {
    "lin318": 42029,
    "rd400":  15281,
    "pcb442": 50778,
    "u574":   36905,
    "rat575": 6773,
}

# SA Parameters
ITERATIONS = 50000
INIT_TEMP = 1000.0
COOLING_RATE = 0.9999
SEED = 42


def main():
    print("=" * 60)
    print("Starting Task 3: Simulated Annealing TSP Analysis")
    print("=" * 60)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    all_results = {}

    for instance_name in TSP_INSTANCES:
        file_path = os.path.join(INPUT_DIR, instance_name)
        instance_slug = instance_name.replace(".tsp", "")

        print(f"\n>>> PROCESSING: {instance_name}")

        try:
            coords, info = load_tsp_file(file_path)
            dist_matrix = _calculate_distance_matrix(coords)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"Running SA with {ITERATIONS} iterations...")
        history, best_p, best_c = simulated_annealing(
            coords, dist_matrix,
            temp=INIT_TEMP,
            cooling_rate=COOLING_RATE,
            iterations=ITERATIONS,
            seed=SEED
        )

        all_results[instance_slug] = {
            "history": history,
            "best_path": best_p,
            "best_cost": best_c,
            "info": info
        }

        save_sa_history_csv(history, os.path.join(OUT_DIR, f"history_{instance_slug}.csv"))

        plot_tsp_solution(
            coords, best_p,
            title=f"SA Best Solution: {instance_slug} (Cost: {best_c:.2f})",
            save_path=os.path.join(OUT_DIR, f"plot_{instance_slug}.png")
        )

        plot_sa_step_simulation(
            coords, history,
            title=f"SA Step Simulation: {instance_slug}",
            save_path=os.path.join(OUT_DIR, f"simulation_{instance_slug}.html")
        )


    save_sa_comparison_report(
        all_results,
        os.path.join(OUT_DIR, "sa_comparison_report.txt"),
        ITERATIONS, INIT_TEMP, COOLING_RATE,
        known_optima=KNOWN_OPTIMA
    )

    print("\n" + "=" * 60)
    print("Task 3 completed. Check folder:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()