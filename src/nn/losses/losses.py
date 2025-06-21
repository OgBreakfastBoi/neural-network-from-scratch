from typing import Any

import numpy as np

from src.nn.losses.loss import LossFunction


class MeanSquaredError(LossFunction):
    """
    Calculates the mean squared error between true and predicted values.

    This loss function computes the average of the squares of the differences
    between the true and predicted values.
    """

    def __init__(self, reduction = "mean", name = "mean_squared_error"):
        super().__init__(reduction, name)

    def call(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return (true - predicted) ** 2

    def deriv(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return predicted - true

    def get_config(self) -> dict[str, Any]:
        config = {
            "name": self.name,
            "reduction": self.reduction
        }
        return config


class CategoricalCrossEntropy(LossFunction):
    """
    Calculates the categorical cross-entropy loss.

    This loss function is commonly used for multi-class classification problems.
    It measures the dissimilarity between the true distribution and the
    predicted distribution.

    It supports both one-hot encoded (`true.ndim == 2`) and sparse
    labels (`true.ndim == 1`).
    """

    def __init__(
        self,
        epsilon = 1e-15,
        reduction = "mean",
        name = "categorical_cross_entropy",
    ):
        super().__init__(reduction, name)
        self.epsilon = epsilon  # Very small number for numerical stability

    def call(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, self.epsilon, 1.0 - self.epsilon)

        if true.ndim == 1:  # support for sparse labels
            batch_indexes = np.arange(true.shape[0])
            return -np.log(predicted[batch_indexes, true])
        elif predicted.ndim == true.ndim == 2:
            return -np.sum(true * np.log(predicted), axis=-1)
        raise ValueError(f"Unsupported label ndim: {true.ndim}")

    def deriv(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, self.epsilon, 1.0 - self.epsilon)

        if true.ndim == 1:  # support for sparse labels
            batch_size = predicted.shape[0]
            y_one_hot = np.zeros_like(predicted)
            y_one_hot[np.arange(batch_size), true] = 1
            return predicted - y_one_hot
        elif true.ndim == 2:
            return predicted - true
        raise ValueError(f"Unsupported label ndim: {true.ndim}")

    def get_config(self) -> dict[str, Any]:
        config = {
            "name": self.name,
            "reduction": self.reduction,
            "epsilon": self.epsilon
        }
        return config
