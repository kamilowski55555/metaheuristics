# Metaheuristics

Implementations of metaheuristic optimisation algorithms for the *Computational Intelligence* course.

---

## Tasks

### Task 1 — Evolutionary Algorithms

Implementation of an Evolutionary Algorithm (EA) and Differential Evolution (DE) for continuous optimisation.

**Algorithms:**
- EA: tournament selection, uniform crossover, Gaussian mutation, (µ+λ) elitism
- DE: DE/rand/1/bin — mutation, binomial crossover, greedy selection

**Benchmark functions** (from [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/optimization.html)):
Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel

**Features:**
- Supports arbitrary dimensionality (3 ≤ n < 50)
- Interactive 3D step-by-step simulation of population movement (Plotly)
- Convergence plots (best / mean / worst per generation)
- EA vs DE comparison reports

```bash
cd task1
python run.py
```

Outputs saved to `task1/results/`.

---

### Task 2 — Greedy TSP (Nearest Neighbour)

Greedy algorithm for the Travelling Salesman Problem using the Nearest Neighbour heuristic.

**Algorithm:** starts from every city, always moves to the closest unvisited city, returns the best tour found.

**Features:**
- Tries all N starting cities for best coverage
- Vectorised distance matrix and neighbour search (NumPy)
- Per-instance report: dataset info, runtime, tour cost, best path
- Top-5 tour visualisations per instance

```bash
cd task2
python run.py
```

Outputs saved to `task2/results/`.

---

### Task 3 — Simulated Annealing for TSP

Simulated Annealing applied to five TSPlib instances with a minimum of 300 cities each.

**Instances:** `lin318`, `rd400`, `pcb442`, `u574`, `rat575`

**Route operators (4):**
| Operator | Description |
|----------|-------------|
| Swap | Swaps two random cities |
| Inversion | Reverses a random sub-segment (2-opt style) |
| Insertion | Removes a city and reinserts it elsewhere |
| Scramble | Shuffles a random sub-segment |

**Features:**
- Metropolis acceptance criterion
- Interactive step-by-step simulation (Plotly HTML, 50 checkpoints)
- Final tour plot per instance
- Comparison report with % gap from known optimal (TSPlib)

```bash
cd task3
python run.py
```

Outputs saved to `task3/results/`.

---

## Requirements

Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

No optimisation libraries are used (no `scipy.optimize`, `DEAP`, `pymoo`, etc.). All algorithms are implemented from scratch.

---

## Project structure

```
metaheuristics/
├── task1/               # EA + DE for benchmark functions
│   ├── ea.py
│   ├── de.py
│   ├── functions.py
│   ├── visualization.py
│   ├── report.py
│   └── run.py
├── task2/               # Greedy Nearest Neighbour TSP
│   ├── tsp.py
│   ├── data_loader.py
│   ├── visualization.py
│   ├── report.py
│   └── run.py
├── task3/               # Simulated Annealing TSP
│   ├── sa.py
│   ├── data_loader.py
│   ├── visualization.py
│   ├── report.py
│   └── run.py
└── tsplib/              # TSPlib dataset files
```
