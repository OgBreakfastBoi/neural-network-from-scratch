import numpy as np

import src.nn.activations as activations
from src.nn.layers.layer import Layer
from src.nn.utils import (
    initialize_biases,
    initialize_weights,
)


class Dense(Layer):
    def __init__(self, units: int, activation: str = None):
        super().__init__()
        self.units = units
        self.activation = activations.get(activation)

        self._weights = np.array([])
        self._biases = np.array([])

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    def build(self, inputs_per_node: int):
        self._weights = initialize_weights(self.units, inputs_per_node)
        self._biases = initialize_biases(self.units)

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: A flattened NumPy array of input data with shape `(len(x),)`

        Returns:
            A NumPy array containing the outputs of all the neurons in this layer after
            the forward pass process with the shape `(len(x),)`
        """

        x = np.sum(x * self.weights, axis=1) + self.biases
        return self.activation(x)

    def compute_output_shape(self) -> tuple[int]:
        return (self.units,)
