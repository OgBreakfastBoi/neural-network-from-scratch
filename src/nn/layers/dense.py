import numpy as np

import src.nn.activations as activations
from src.nn.utils import (
    initialize_biases,
    initialize_weights,
)


class Dense:
    def __init__(self, nodes: int, activation: str = None):
        self.nodes = nodes
        self.activation = activations.get(activation)
        self.weights = np.array([])
        self.biases = np.array([])

    def build(self, inputs_per_node: int):
        self.weights = initialize_weights(self.nodes, inputs_per_node)
        self.biases = initialize_biases(self.nodes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: A flattened NumPy array of input data with shape `(len(x),)`

        Returns:
            A NumPy array containing the outputs of all the neurons in this layer after
            the forward pass process with the shape `(len(x),)`
        """

        x = np.sum(x * self.weights, axis=1) + self.biases
        return self.activation(x)
