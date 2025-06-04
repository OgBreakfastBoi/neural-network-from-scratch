import numpy as np


def initialize_weights(nodes: int, inputs_per_node: int) -> np.ndarray:
    return np.random.randn(nodes, inputs_per_node)


def initialize_biases(nodes: int) -> np.array:
    return np.random.randn(nodes)
