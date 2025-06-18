import numpy as np

from src.nn.activations.activation import ActivationFunction


class Linear(ActivationFunction):
    """
    Linear activation function: f(x) = x
    """

    def __init__(self, name = "linear"):
        super().__init__(name)

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


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLu) activation function: f(x) = max(0, x)
    """

    def __init__(self, name = "relu"):
        super().__init__(name)

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


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    """

    def __init__(self, cache_outputs = True, name = "sigmoid"):
        super().__init__(name)
        self.cache_outputs = cache_outputs
        self.output = None

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function to input.
        """
        result = self.static_call(x)
        if self.cache_outputs: self.output = result
        return result

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid: f(x) * (1 - f(x))
        """
        sig = self.static_call(x) if self.output is None else self.output
        return sig * (1 - sig)

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        """
        Static version that computes sigmoid of the input.
        """
        return 1 / (1 + np.exp(-x))


class Softmax(ActivationFunction):
    """
    Softmax activation function for converting a vector into a probability distribution.
    """

    def __init__(self, cache_outputs = True, name = "softmax"):
        super().__init__(name)
        self.cache_outputs = cache_outputs
        self.output = None

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to input.
        """
        result = self.static_call(x)
        if self.cache_outputs:
            self.output = result
        return result

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the softmax function.

        For 1D input: returns (C, C) Jacobian.
        For 2D input: returns (B, C, C) Jacobians for each sample in batch.
        """

        if x.ndim == 1:
            s = self.static_call(x) if self.output is None else self.output
            s = s.reshape(-1, 1)
            return np.diagflat(s) - s @ s.T
        elif x.ndim == 2:
            s = self.static_call(x) if self.output is None else self.output
            batch_size, num_classes = s.shape
            jacobians = np.empty((batch_size, num_classes, num_classes))

            for i in range(batch_size):
                s_i = s[i].reshape(-1, 1)
                jacobians[i] = np.diagflat(s_i) - s_i @ s_i.T

            return jacobians
        raise ValueError(
            f"Softmax derivative only supports 1D or 2D input. Got shape {x.shape}."
        )

    @staticmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        """
        Static version of softmax that is numerically stable.
        """

        if x.ndim == 1:
            exp_x = np.exp(x - x.max())
            return exp_x / np.sum(exp_x)
        elif x.ndim == 2:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        raise ValueError(f"Softmax only supports 1D or 2D input. Got shape {x.shape}.")
