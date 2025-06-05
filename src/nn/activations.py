import numpy as np


class Activation:
    def __call__(self, *args: np.ndarray, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def deriv(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Activation):
    def call(self, x: np.ndarray) -> np.ndarray:
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        return x


linear = Linear()


class ReLU(Activation):
    def call(self, x: np.ndarray) -> np.ndarray:
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


relu = ReLU()


class Sigmoid(Activation):
    def call(self, x: np.ndarray) -> np.ndarray:
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        sig = self.static_call(x)
        return sig * (1 - sig)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


sigmoid = Sigmoid()


class Softmax(Activation):
    def call(self, x: np.ndarray) -> np.ndarray:
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise NotImplementedError("Softmax derivative supports only 1D input")

        s = self.static_call(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - x.max())
        return exp_x / np.sum(exp_x)


softmax = Softmax()

ALL_OBJECTS = [
    linear,
    relu,
    sigmoid,
    softmax,
]
ALL_OBJECTS_DICT = {fn.__class__.__name__.lower(): fn for fn in ALL_OBJECTS}


def get(identifier: str | None) -> Activation:
    """
    Retrieve activation function via an identifier. If identifier is
    ``None`` then return a linear activation where ``f(x)=x``

    Currently only supports:
      - relu
      - sigmoid
      - softmax
    """

    if identifier is None:
        return linear
    elif isinstance(identifier, str):
        return ALL_OBJECTS_DICT[identifier.lower()]
    raise ValueError(f"Activation '{identifier}' is currently unsupported or has been mistyped")
