import numpy as np


def differential_evolution(func, bounds, dim,
                           pop_size=50, max_generations=500,
                           F=0.8, CR=0.9, seed=None):
    """
    Differential Evolution (DE/rand/1/bin).

    Parameters
    ----------
    func            : callable, f(x) -> float, to be minimized
    bounds          : (low, high) tuple, same for all dimensions
    dim             : number of dimensions
    pop_size        : number of individuals (N); must be >= 4
    max_generations : stopping criterion
    F               : differential weight, controls mutation step size
    CR              : crossover probability [0, 1]
    seed            : random seed for reproducibility

    Returns
    -------
    history : list of dicts, one per generation:
              {"best": float, "mean": float, "worst": float,
               "best_x": ndarray, "population": ndarray}
    """
    rng = np.random.default_rng(seed)
    low, high = bounds

    # ------------------------------------------------------------------ #
    # 1. INITIALIZATION                                                    #
    # ------------------------------------------------------------------ #
    population = rng.uniform(low, high, size=(pop_size, dim))

    fitness = np.array([func(ind) for ind in population])

    history = []

    for gen in range(max_generations):

        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(pop_size):

            # ---------------------------------------------------------- #
            # 2. MUTATION: pick 3 distinct indices, all different from i  #
            # ---------------------------------------------------------- #
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = rng.choice(candidates, size=3, replace=False)

            mutant = population[a] + F * (population[b] - population[c])

            mutant = np.clip(mutant, low, high)

            # ---------------------------------------------------------- #
            # 3. CROSSOVER                                   #
            # ---------------------------------------------------------- #
            j_rand = rng.integers(0, dim)
            cross_mask = rng.random(dim) < CR
            cross_mask[j_rand] = True        

            trial = np.where(cross_mask, mutant, population[i])

            # ---------------------------------------------------------- #
            # 4. SELECTION                                                 #
            # ---------------------------------------------------------- #
            trial_fitness = func(trial)
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        population = new_population
        fitness = new_fitness

        # ---------------------------------------------------------------- #
        # 5. RECORD statistics for this generation                         #
        # ---------------------------------------------------------------- #
        best_idx = np.argmin(fitness)
        history.append({
            "generation": gen,
            "best":   float(fitness[best_idx]),
            "mean":   float(np.mean(fitness)),
            "worst":  float(np.max(fitness)),
            "best_x": population[best_idx].copy(),
            "population": population.copy(),
        })

    return history
