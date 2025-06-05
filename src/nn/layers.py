import numpy as np

import src.nn.activations as activations

class Dense:
    def __init__(self, nodes: int, activation: str = None):
        self.nodes = nodes
        self.activation = activations.get(activation)
        self.weights = np.array([])
        self.biases = np.array([])

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: A flattened NumPy array of input data with shape `(len(x),)`

        Returns:
            A NumPy array containing the outputs of all the neurons in this layer after
            the feed forward process with the shape `(len(x),)`
        """

        x = np.sum(x * self.weights, axis=1) + self.biases
        return self.activation(x)
