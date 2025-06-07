import numpy as np

from src.nn.activations.base import ActivationFunction


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
