import numpy as np

def sphere(x):
    return np.sum(x ** 2)

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def rastrigin(x):
    d = len(x)
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    return (
        -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
        - np.exp(np.sum(np.cos(c * x)) / d)
        + a + np.exp(1)
    )

def schwefel(x):
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


FUNCTIONS = {
    "sphere":     {"fn": sphere,     "bounds": (-5.12, 5.12),   "optimum": 0.0},
    "rosenbrock": {"fn": rosenbrock, "bounds": (-5.0,  10.0),   "optimum": 0.0},
    "rastrigin":  {"fn": rastrigin,  "bounds": (-5.12, 5.12),   "optimum": 0.0},
    "ackley":     {"fn": ackley,     "bounds": (-32.768, 32.768),"optimum": 0.0},
    "schwefel":   {"fn": schwefel,   "bounds": (-500.0, 500.0), "optimum": 0.0},
}
