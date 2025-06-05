import numpy as np


class ActivationFunction:
    """
    Base class for activation functions used in neural networks.

    Provides an interface for:
      - ``__call__``: calls the activation function.
      - ``call``: the core activation logic to override.
      - ``deriv``: derivative of the activation function for backpropagation.
      - ``static_call``: static version of the activation, usable without instantiation.

    Subclasses should implement the ``call``, ``deriv``, and ``static_call`` methods.
    """

    def __call__(self, *args: np.ndarray, **kwargs):
        """
        Allows instances to be used as functions.
        Calls the ``call`` method with given arguments.
        """
        return self.call(*args, **kwargs)

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the activation for the input array.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation with respect to its input.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        """
        Static version of the activation function.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class Linear(ActivationFunction):
    """
    Linear activation function: f(x) = x
    """

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the input unchanged.
        """
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the linear function is 1.
        """
        return np.ones_like(x)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        """
        Static version that returns input unchanged.
        """
        return x


linear = Linear()


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLu) activation function: f(x) = max(0, x)
    """

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Applies ReLU to the input.
        """
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative is 1 where x > 0, else 0.
        """
        return (x > 0).astype(x.dtype)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        """
        Static version that returns max(0, x).
        """
        return np.maximum(0, x)


relu = ReLU()


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    """

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function to input.
        """
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid: f(x) * (1 - f(x))
        """
        sig = self.static_call(x)
        return sig * (1 - sig)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        """
        Static version that computes sigmoid of the input.
        """
        return 1 / (1 + np.exp(-x))


sigmoid = Sigmoid()


class Softmax(ActivationFunction):
    """
    Softmax activation function for converting a vector into a probability distribution.
    """

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to input.
        """
        return self.static_call(x)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the softmax function for a 1D input.

        Note:
            Only supports 1D input. For 2D input, this would need to be extended
            with batched processing.
        """

        if x.ndim != 1:
            raise NotImplementedError("Softmax derivative supports only 1D input")

        s = self.static_call(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        """
        Static version of softmax that is numerically stable.
        """
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


def get(identifier: str | None) -> ActivationFunction:
    """
    Retrieve activation function via an identifier.

    Args:
        identifier (str | None): The name of the activation function.

            Supported identifiers:
              - 'relu'
              - 'sigmoid'
              - 'softmax'
            If None, returns the linear activation.

    Returns:
        Activation: The corresponding activation function instance.

    Raises:
        ValueError: If the identifier is not recognized.
        TypeError: If identifier is not a string or None.
    """

    if identifier is None:
        return linear
    elif isinstance(identifier, str):
        key = identifier.lower()
        if key in ALL_OBJECTS_DICT:
            return ALL_OBJECTS_DICT[key]
        raise ValueError(
            f"Unknown activation '{identifier}'. Supported: {list(ALL_OBJECTS_DICT.keys())}"
        )
    raise TypeError(f"ActivationFunction identifier must be a string or None, not {type(identifier)}")
