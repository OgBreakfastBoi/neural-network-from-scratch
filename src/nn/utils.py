import numpy as np


def initialize_weights(nodes: int, inputs_per_node: int) -> np.ndarray:
    return np.random.normal(0, np.sqrt(2 / inputs_per_node), (inputs_per_node, nodes))


def initialize_biases(nodes: int) -> np.array:
    return np.zeros(nodes)
