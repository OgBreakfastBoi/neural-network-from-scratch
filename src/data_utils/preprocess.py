import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    xrange = x.max() - x.min()
    return (x - x.min()) / xrange
