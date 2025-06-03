import math

import numpy as np

import src.nn.utils as utils
from src.nn.errors import (
    ModelNotCompiledError,
    NoLayersError,
)
from src.nn.layers import Dense


class NeuralNetwork:
    def __init__(self, input_size: tuple[int] | tuple[int, int]):
        self.input_size = math.prod(input_size)
        self.layers: list[Dense] = []

        self._compiled = False
        self._config = {}

    def add(self, layer: Dense):
        self.layers.append(layer)

    def compile(self, optimizer: str, loss: str):
        if len(self.layers) == 0:
            raise NoLayersError()

        self._config['optimizer'] = optimizer
        self._config['loss'] = loss
        self._config['activations'] = [fn.activation.__name__ for fn in self.layers]

        prev_layer_size = self.input_size

        for i in range(len(self.layers)):
            self.layers[i].weights = utils.initialize_weights(self.layers[i].nodes, prev_layer_size)
            self.layers[i].biases = utils.initialize_biases(self.layers[i].nodes)
            prev_layer_size = self.layers[i].nodes

        self._compiled = True

    def train(self, x: np.ndarray, y: np.ndarray):
        # if not self._compiled:
        #     raise ModelNotCompiledError()

        raise NotImplementedError

    def run(self, x: np.ndarray, y: np.ndarray = None, list_misclassifications = False) -> list | None:
        """
        Runs the network on the provided data and assumes the data is normalized.
        If labels are provided for the data then a percentage accuracy report
        will be printed after the feed forward process.
        """

        if not self._compiled:
            raise ModelNotCompiledError()

        if list_misclassifications and y is None:
            raise ValueError("Cannot return misclassifications if labels are not specified.")

        if y.any() and (len(x) != len(y)):
            raise ValueError("The length of the dataset and labels must be the same")

        if y.ndim != 1:
            raise ValueError(f"Labels NumPy array is not flat. It has shape {y.shape}")

        if x.ndim != 2: x = x.reshape((x.shape[0]), -1)  # flattens the dataset if not already flattened

        predictions = []  # will only be populated if `y` is provided

        # feed forward process
        for i, v in enumerate(x):
            result = self.layers[0].feed_forward(v)

            for j in range(1, len(self.layers)):
                result = self.layers[j].feed_forward(result)

            if len(result) > 1:
                prediction = result.argmax() + y.min()
            else:
                prediction = result + y.min()

            if y.any():
                predictions.append(prediction)
            else:
                print(f"Prediction for data point {i} is: {prediction}")

        # accuracy report
        if y.any():
            predictions = np.array(predictions)
            misclassification_indices = np.nonzero(np.subtract(y, predictions))[0]

            print(f"Accuracy: {((len(x) - len(misclassification_indices)) / len(x)) * 100}%")

            # misclassifications
            if list_misclassifications:
                misclassifications = []
                for i in misclassification_indices:
                    info_dict = {
                        "label": y[i],
                        "prediction": predictions[i],
                        "data_point": x[i]
                    }
                    misclassifications.append(info_dict)
                return misclassifications
            return None
        return None

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
