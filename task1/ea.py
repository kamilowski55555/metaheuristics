import numpy as np


def _tournament(fitness, k, rng):
    """Pick one parent index via tournament selection."""
    candidates = rng.choice(len(fitness), size=k, replace=False)
    return candidates[np.argmin(fitness[candidates])]


def evolutionary_algorithm(func, bounds, dim,
                            pop_size=50, max_generations=500,
                            crossover_rate=0.8, sigma=None, tournament_k=3,
                            seed=None):
    """
    Evolutionary Algorithm with:
      - Tournament selection (k=3)
      - Uniform crossover
      - Gaussian mutation
      - (mu + lambda) elitism: parents and children compete, best N survive

    Parameters
    ----------
    func            : callable, f(x) -> float, to be minimized
    bounds          : (low, high) tuple, same for all dimensions
    dim             : number of dimensions
    pop_size        : mu = lambda = pop_size (N parents produce N children)
    max_generations : stopping criterion
    crossover_rate  : probability that crossover happens vs. child = copy of parent
    sigma           : Gaussian mutation std dev; defaults to 10% of search range
    tournament_k    : number of competitors per tournament
    seed            : random seed for reproducibility

    Returns
    -------
    history : list of dicts, one per generation:
              {"best": float, "mean": float, "worst": float,
               "best_x": ndarray, "population": ndarray}
    """
    rng = np.random.default_rng(seed)
    low, high = bounds

    # Sigma defaults to 10% of the search range.
    # Think of it like a learning rate: too small = slow, too large = random walk.
    if sigma is None:
        sigma = 0.1 * (high - low)

    # ------------------------------------------------------------------ #
    # 1. INITIALIZATION                                                    #
    # ------------------------------------------------------------------ #
    population = rng.uniform(low, high, size=(pop_size, dim))
    fitness = np.array([func(ind) for ind in population])

    history = []

    for gen in range(max_generations):

        children = np.empty_like(population)

        for i in range(pop_size):

            # ---------------------------------------------------------- #
            # 2. SELECTION: pick two parents via tournament               #
            # ---------------------------------------------------------- #
            p1 = _tournament(fitness, tournament_k, rng)
            p2 = _tournament(fitness, tournament_k, rng)

            # ---------------------------------------------------------- #
            # 3. CROSSOVER: uniform — each dimension is a coin flip       #
            # ---------------------------------------------------------- #
            if rng.random() < crossover_rate:
                mask = rng.random(dim) < 0.5
                child = np.where(mask, population[p1], population[p2])
            else:
                # No crossover: child is a copy of the better parent
                child = population[p1].copy()

            # ---------------------------------------------------------- #
            # 4. MUTATION: add Gaussian noise, clip to stay in bounds     #
            # ---------------------------------------------------------- #
            child += rng.normal(0, sigma, size=dim)
            child = np.clip(child, low, high)

            children[i] = child

        # ---------------------------------------------------------------- #
        # 5. SURVIVOR SELECTION: (mu + lambda) — best N from combined pool #
        # ---------------------------------------------------------------- #
        children_fitness = np.array([func(c) for c in children])

        combined_pop = np.vstack([population, children])
        combined_fit = np.concatenate([fitness, children_fitness])

        survivors = np.argsort(combined_fit)[:pop_size]
        population = combined_pop[survivors]
        fitness = combined_fit[survivors]

        # ---------------------------------------------------------------- #
        # 6. RECORD statistics                                             #
        # ---------------------------------------------------------------- #
        history.append({
            "generation": gen,
            "best":   float(fitness[0]),
            "mean":   float(np.mean(fitness)),
            "worst":  float(fitness[-1]),
            "best_x": population[0].copy(),
            "population": population.copy(),
        })

    return history
