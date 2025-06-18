import numpy as np

from src.nn import activations
from src.nn.errors import LayerNotBuiltError
from src.nn.layers.layer import Layer
from src.nn.optimizers.optimizer import Optimizer
from src.nn.utils import (
    initialize_biases,
    initialize_weights,
)


class Dense(Layer):
    def __init__(self, units: int, activation: str = None, name = "Dense"):
        super().__init__(name)
        self.units = units
        self.activation = activations.get(activation)
        self.prev_input = np.array([])
        self.prev_output = np.array([])

        self._weights = np.array([])
        self._biases = np.array([])

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    def build(self, inputs_per_node: int, index: int):
        self._idx = index
        self._weights = initialize_weights(self.units, inputs_per_node)
        self._biases = initialize_biases(self.units)
        self._built = True

    def call(self, x: np.ndarray) -> np.ndarray:
        self.prev_input = x
        x = x @ self._weights + self._biases
        self.prev_output = self.activation(x)
        return self.prev_output

    def backward(self, grad_input: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        if not self._built:
            raise LayerNotBuiltError()

        activation_deriv = self.activation.deriv(self.prev_output)
        if activation_deriv.ndim == 3:
            delta = np.einsum("bij,bi->bj", activation_deriv, grad_input)
        else:
            delta = grad_input * activation_deriv

        self._weights = optimizer.step(
            self._weights,
            self.prev_input.T @ delta,
            f"weights{self._idx}"
        )
        self._biases = optimizer.step(
            self._biases,
            np.sum(delta, axis=0),
            f"biases{self._idx}"
        )

        return delta @ self._weights.T

    def output_shape(self) -> tuple[int]:
        return (self.units,)
