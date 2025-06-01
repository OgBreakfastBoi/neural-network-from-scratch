import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def relu(x: float) -> float:
    return max(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))
