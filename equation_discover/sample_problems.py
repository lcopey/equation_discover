import numpy as np
from .sample import Uniform, Bounds


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) * (1 / (sigma * np.sqrt(2 * np.pi)))


def generate_gaussian_problem(n: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = Bounds([
        Uniform(-5, 5),
        Uniform(-2, 2),
        Uniform(1, 2)
    ])

    X = bounds.sample(n)
    y = gaussian(*X.T)

    return X, y
