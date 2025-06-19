from time import perf_counter
from typing import Any

import numpy as np

from src.nn import (
    losses,
    optimizers,
)
from src.nn.errors import (
    ModelNotCompiledError,
    NoLayersError,
)
from src.nn.layers.dense import Dense
from src.nn.layers.flatten import Flatten
from src.nn.layers.layer import (
    InputLayer,
    Layer,
)
from src.nn.losses.loss import LossFunction
from src.nn.optimizers.optimizer import Optimizer


class NeuralNetwork:
    def __init__(self, name: str = None):
        self.name = str(id(self)) if name is None else name

        self._layers: list[Layer] = []
        self._optimizer: Optimizer = None
        self._loss_fn: LossFunction = None
        self._compiled = False
        self._last_activation = None

    def __repr__(self):
        return (
            f"<NeuralNetwork: {self.name}, compiled={self._compiled}, "
            f"optimizer={self._optimizer}, loss_function={self._loss_fn}>"
        )

    def add(self, layer: Layer):
        self._layers.append(layer)

    def compile(self, optimizer: str | tuple[str, float], loss: str):
        if len(self._layers) == 0:
            raise NoLayersError()
        if not isinstance(self._layers[0], InputLayer):
            raise ValueError(
                f"The first layer must be an InputLayer. "
                f"Supported InputLayers: {InputLayer.__subclasses__()}"
            )

        if isinstance(optimizer, str):
            self._optimizer = optimizers.get(optimizer)
        elif isinstance(optimizer, tuple):
            self._optimizer = optimizers.get(optimizer[0])
            self._optimizer.lr = optimizer[1]
        else:
            raise TypeError(
                f"Optimizer must be a string or tuple(string, integer), "
                f"not {type(optimizer)}"
            )

        self._loss_fn = losses.get(loss)

        # build layers
        for i, layer in enumerate(self._layers):
            if isinstance(layer, Flatten):
                layer.build(i)
            elif isinstance(layer, Dense):
                layer.build(self._layers[i - 1].output_shape()[-1], i)
            else:
                raise RuntimeError(
                    f"Trying to build unsupported Layer: {layer}"
                )

        last_layer = self._layers[-1]
        if isinstance(last_layer, Dense):
            self._last_activation = last_layer.activation.name

        self._compiled = True

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 1,
        report_freq: int = 1,
    ):
        if not self._compiled:
            raise ModelNotCompiledError()

        total_time = 0
        for epoch in range(1, epochs + 1):
            start_time = perf_counter()

            perm = np.random.permutation(x.shape[0])
            x_shuffled = x[perm]
            y_shuffled = y[perm]

            loss = 0
            for start in range(0, x.shape[0], batch_size):
                end = start + batch_size
                loss += self._train_batch(x_shuffled[start:end], y_shuffled[start:end])

            end_time = perf_counter()
            time = end_time - start_time
            total_time += time

            if epoch % report_freq == 0:
                labels = y
                pred = self._forward(x)

                if self._last_activation == "softmax":
                    pred = np.argmax(pred, axis=-1)
                if labels.ndim == 2:  # in case of one hots
                    labels = np.argmax(labels, axis=1)

                misclassification_indices = np.nonzero(labels - pred)[0]
                accuracy = (x.shape[0] - misclassification_indices.shape[0]) / x.shape[0]
                avg_loss = loss / (x.shape[0] / batch_size)
                print(
                    f"Epoch {epoch}/{epochs}, Accuracy: {accuracy:.6f}, "
                    f"Loss: {avg_loss:.6f}, Duration: {time:.6f} seconds"
                )

        avg_time = total_time / epochs
        print(
            f"Training duration: {total_time:.6f} seconds. "
            f"Average duration per epoch: {avg_time:.6f} seconds."
        )

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, dict[str, np.ndarray]]:
        if not self._compiled:
            raise ModelNotCompiledError()
        if x.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError(
                f"Zero-size inputs for either x or y. Received lengths "
                f"{x.shape[0]} and {y.shape[0]} respectively"
            )
        if y.ndim > 2:
            raise ValueError(f"Unsupported y ndim: {y.ndim}")

        pred = self._forward(x)
        loss = self._loss_fn(y, pred)

        if self._last_activation == "softmax":
            pred = np.argmax(pred, axis=-1)
        if y.ndim == 2:  # in case of one hots
            y = np.argmax(y, axis=1)

        misclassification_indices = np.nonzero(y - pred)[0]
        accuracy = (x.shape[0] - misclassification_indices.shape[0]) / x.shape[0]
        misclassifications_dict = {
            "dataset": x[misclassification_indices],
            "labels": y[misclassification_indices],
            "predictions": pred[misclassification_indices]
        }

        return loss, accuracy, misclassifications_dict

    def predict(self, x: np.ndarray) -> list[float | int]:
        if not self._compiled:
            raise ModelNotCompiledError()

        predictions = []

        for i in range(x.shape[0]):
            pred = self._forward(x[i])

            if self._last_activation == "softmax":
                argmax = np.argmax(pred, axis=-1)
                predictions.append(argmax)
                print(f"Prediction({i + 1}/{x.shape[0]}): {argmax}. Confidence: {pred[argmax]}")
            else:
                predictions.append(pred)
                print(f"Prediction({i + 1}/{x.shape[0]}): {pred}")

        return predictions

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        if not self._compiled:
            raise ModelNotCompiledError(
                "The model must be compiled before config can be generated."
            )

        config = {
            "name": self.name,
            "layers": self._layers,
            "optimizer": self._optimizer,
            "loss_function": self._loss_fn
        }
        return config

    @property
    def compiled(self) -> bool:
        return self._compiled

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def loss_function(self) -> LossFunction:
        return self._loss_fn

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def _forward(self, x: np.ndarray) -> np.ndarray:
        result = x
        for layer in self._layers:
            result = layer(result)
        return result

    def _train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self._forward(x)
        loss = self._loss_fn(y, pred)
        grad = self._loss_fn.deriv(y, pred)

        # Backward pass
        for layer in reversed(self._layers):
            grad = layer.backward(grad, self._optimizer)

        return loss
