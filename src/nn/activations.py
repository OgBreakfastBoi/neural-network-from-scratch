import numpy as np


class Activation:
    @staticmethod
    def call(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def deriv(x: np.ndarray) -> np.ndarray:
        return x


def linear(): return Linear


class Linear(Activation):
    @staticmethod
    def call(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def deriv(x: np.ndarray) -> np.ndarray:
        return np.ones(len(x))


def relu(): return ReLU


class ReLU(Activation):
    @staticmethod
    def call(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def deriv(x: np.ndarray) -> np.ndarray:
        non_zero_indices = np.nonzero(x)
        x[non_zero_indices] = 1
        return x


def sigmoid(): return Sigmoid


class Sigmoid(Activation):
    @staticmethod
    def call(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def deriv(x: np.ndarray) -> np.ndarray:
        return Sigmoid.call(x) * (1 - Sigmoid.call(x))


def softmax(): return Softmax


class Softmax(Activation):
    @staticmethod
    def call(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - x.max())
        return exp_x / np.sum(exp_x)

    @staticmethod
    def deriv(x: np.ndarray) -> np.ndarray:
        s = Softmax.call(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)


ALL_OBJECTS = [
    linear,
    relu,
    sigmoid,
    softmax,
]
ALL_OBJECTS_DICT = {fn.__name__: fn for fn in ALL_OBJECTS}


def get(identifier: str | None) -> type[Activation]:
    """
    Retrieve activation function via an identifier. If identifier is
    ``None`` then return a linear activation where ``f(x)=x``

    Currently only supports:
      - relu
      - sigmoid
      - softmax
    """

    if identifier is None:
        return linear()
    elif isinstance(identifier, str):
        return ALL_OBJECTS_DICT[identifier]()
    raise ValueError(f"Activation '{identifier}' is currently unsupported or has been mistyped")
