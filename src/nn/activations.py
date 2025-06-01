from collections.abc import Callable

import numpy as np


def linear(x: np.ndarray) -> np.ndarray:
    return x


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


ALL_OBJECTS = [
    linear,
    relu,
    sigmoid,
    softmax,
]
ALL_OBJECTS_DICT = {fn.__name__: fn for fn in ALL_OBJECTS}


def get(identifier: str | Callable[[np.ndarray], np.ndarray] | None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Retrieve activation function via an identifier.

    Currently only supports:
      - relu
      - sigmoid
      - softmax
    """

    if identifier is None:
        return linear
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        return ALL_OBJECTS_DICT[identifier]
    raise ValueError(f"Unable to resolve an activation via provided identifier '{identifier}'.")
