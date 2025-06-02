import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    xrange = x.max() - x.min()
    return (x - x.min()) / xrange


def one_hot_encode(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError(f"NumPy array is not flat. It has shape {x.shape}")

    x_min = x.min()
    x_range = x.max() - x_min + 1
    return np.eye(x_range)[x - x_min]
