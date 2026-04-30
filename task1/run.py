"""
Zadanie 1 – Algorytmy ewolucyjne
Inteligencja obliczeniowa, dr Przemysław Juszczuk

Autor: Kamil Kot, Dawid Wieczorek

================================================================================
PSEUDOKOD – Ewolucja Różnicowa (DE/rand/1/bin)
================================================================================

WEJŚCIE: funkcja f, przedział [low, high], wymiar d, pop_size, max_gen, F, CR

1. Inicjalizacja:
   population[i] = losowy punkt w [low, high]^d   dla i = 0..pop_size-1
   fitness[i]    = f(population[i])

2. Dla każdej generacji gen = 0..max_gen-1:
   Dla każdego osobnika i:
     a) Mutacja:
        Wybierz losowo trzech różnych osobników a, b, c (różnych od i)
        mutant = population[a] + F * (population[b] - population[c])
        mutant = clip(mutant, low, high)

     b) Krzyżowanie (binomialne):
        j_rand = losowy indeks wymiaru (gwarantuje co najmniej 1 zmianę)
        trial[j] = mutant[j]         jeśli rand() < CR lub j == j_rand
                   population[i][j]  w przeciwnym razie

     c) Selekcja zachłanno-lokalna:
        jeśli f(trial) <= fitness[i]:
            population[i] = trial
            fitness[i]    = f(trial)

   Zapisz: best = min(fitness), mean = avg(fitness), worst = max(fitness)

WYJŚCIE: historia statystyk per generacja, najlepszy osobnik

================================================================================
PSEUDOKOD – Algorytm Ewolucyjny (EA)
================================================================================

WEJŚCIE: funkcja f, przedział [low, high], wymiar d, pop_size, max_gen,
         crossover_rate, sigma, tournament_k

1. Inicjalizacja:
   population[i] = losowy punkt w [low, high]^d
   fitness[i]    = f(population[i])

2. Dla każdej generacji:
   Dla i = 0..pop_size-1 (tworzenie dzieci):
     a) Selekcja (turniejowa):
        p1 = najlepszy z k losowych osobników
        p2 = najlepszy z k innych losowych osobników

     b) Krzyżowanie (jednostajne):
        jeśli rand() < crossover_rate:
            child[j] = population[p1][j]  jeśli rand() < 0.5
                       population[p2][j]  w przeciwnym razie
        w przeciwnym razie: child = kopia p1

     c) Mutacja (gaussowska):
        child[j] += N(0, sigma)
        child     = clip(child, low, high)

   Selekcja przeżycia (mu + lambda):
     Połącz population i children (razem 2*pop_size osobników)
     Zachowaj pop_size najlepszych jako nowa populacja

   Zapisz: best, mean, worst

WYJŚCIE: historia statystyk per generacja, najlepszy osobnik
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from functions import FUNCTIONS
from de import differential_evolution
from ea import evolutionary_algorithm
from visualization import plot_convergence, step_simulation_3d
from report import save_history_csv, save_comparison_report

# ── Configuration ─────────────────────────────────────────────────────────────

DIM_3D   = 3    # used for step simulation (must be 3 for 3D surface)
DIM_ND   = 10   # used for n-dimensional runs and comparison report
POP_SIZE = 50
GENERATIONS = 300
SEED = 42

FUNC_NAMES = ["sphere", "rosenbrock", "rastrigin", "ackley", "schwefel"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_pair(func_name, dim):
    """Run EA and DE on one function, return (ea_history, de_history)."""
    info = FUNCTIONS[func_name]
    print(f"  Running EA ...", end=" ", flush=True)
    ea_h = evolutionary_algorithm(
        info["fn"], info["bounds"], dim,
        pop_size=POP_SIZE, max_generations=GENERATIONS, seed=SEED,
    )
    print(f"best={ea_h[-1]['best']:.4f}")

    print(f"  Running DE ...", end=" ", flush=True)
    de_h = differential_evolution(
        info["fn"], info["bounds"], dim,
        pop_size=POP_SIZE, max_generations=GENERATIONS, seed=SEED,
    )
    print(f"best={de_h[-1]['best']:.4f}")

    return ea_h, de_h


# ── Phase 1: 3D runs — step simulations + convergence plots ──────────────────

print("=" * 60)
print(f"Phase 1: 3D visualisation + comparison (dim={DIM_3D})")
print("=" * 60)

results_3d = {}

for name in FUNC_NAMES:
    print(f"\n[{name}]")
    ea_h, de_h = run_pair(name, DIM_3D)
    info = FUNCTIONS[name]

    save_history_csv(ea_h, os.path.join(OUT_DIR, f"ea_{name}_d{DIM_3D}.csv"))
    save_history_csv(de_h, os.path.join(OUT_DIR, f"de_{name}_d{DIM_3D}.csv"))

    results_3d[name] = {"ea": ea_h, "de": de_h}

    # Convergence plots
    plot_convergence(
        ea_h,
        title=f"EA on {name} (dim={DIM_3D})",
        save_path=os.path.join(OUT_DIR, f"convergence_ea_{name}_d{DIM_3D}.png"),
    )
    plot_convergence(
        de_h,
        title=f"DE on {name} (dim={DIM_3D})",
        save_path=os.path.join(OUT_DIR, f"convergence_de_{name}_d{DIM_3D}.png"),
    )

    step_simulation_3d(
        func=info["fn"], bounds=info["bounds"], history=ea_h,
        title=f"EA on {name} (dim={DIM_3D}) - step simulation",
        save_path=os.path.join(OUT_DIR, f"simulation_ea_{name}.html"),
        n_grid=40, max_frames=60,
    )
    step_simulation_3d(
        func=info["fn"], bounds=info["bounds"], history=de_h,
        title=f"DE on {name} (dim={DIM_3D}) - step simulation",
        save_path=os.path.join(OUT_DIR, f"simulation_de_{name}.html"),
        n_grid=40, max_frames=60,
    )

save_comparison_report(
    results_3d,
    path=os.path.join(OUT_DIR, f"comparison_d{DIM_3D}.txt"),
    dim=DIM_3D,
    generations=GENERATIONS,
    pop_size=POP_SIZE,
)

# ── Phase 2: n-dimensional runs — comparison report ──────────────────────────

print()
print("=" * 60)
print(f"Phase 2: n-dimensional comparison (dim={DIM_ND})")
print("=" * 60)

results_nd = {}
for name in FUNC_NAMES:
    print(f"\n[{name}]")
    ea_h, de_h = run_pair(name, DIM_ND)

    save_history_csv(ea_h, os.path.join(OUT_DIR, f"ea_{name}_d{DIM_ND}.csv"))
    save_history_csv(de_h, os.path.join(OUT_DIR, f"de_{name}_d{DIM_ND}.csv"))

    results_nd[name] = {"ea": ea_h, "de": de_h}

save_comparison_report(
    results_nd,
    path=os.path.join(OUT_DIR, f"comparison_d{DIM_ND}.txt"),
    dim=DIM_ND,
    generations=GENERATIONS,
    pop_size=POP_SIZE,
)

print()
print("=" * 60)
print("All outputs saved to:", OUT_DIR)
print("=" * 60)
