from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

import numpy as np


class LossFunction(ABC):
    """
    Loss function base class.

    All subclasses inheriting this class must implement the ``call`` method
    and assign an appropriate ``name`` for the loss function.
    """

    def __init__(self, reduction: str | None, name: str):
        self.reduction = check_reduction(reduction)
        self.name = name

    def __repr__(self):
        return f"<LossFunction: {self.name}, reduction={self.reduction}>"

    def __call__(
        self,
        true: np.ndarray,
        predicted: np.ndarray,
    ) -> float | np.ndarray:
        if true.shape[0] == 0 or predicted.shape[0] == 0:
            raise ValueError(
                f"Zero-size inputs for either true or predicted. Received "
                f"lengths {true.shape[0]} and {predicted.shape[0]} respectively"
            )

        losses = self.call(true, predicted)
        return reduce_losses(losses, self.reduction)

    @abstractmethod
    def call(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Loss function implementation.

        Args:
            true: The true labels of the dataset
            predicted: The predicted outputs from the network

        Returns:
            An array of losses for each sample.
        """
        raise NotImplementedError

    @abstractmethod
    def deriv(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        raise NotImplementedError


def check_reduction(reduction: str | None):
    """
    Checks if the provided reduction is currently supported.
    If supported, return the input else raise a ValueError.
    """
    supported = [
        "mean",
        "sum",
        "none",
        None
    ]
    if reduction not in supported:
        raise ValueError(
            f"Unknown reduction '{reduction}'. Supported: {supported}"
        )
    if reduction == "none":
        return None
    return reduction


def reduce_losses(
    values: np.ndarray, reduction: str | None,
) -> float | np.ndarray:
    """
    Applies a reduction method to an array of loss values.

    Returns:
        The reduced loss as a ``float`` if 'mean' or 'sum' is used,
        otherwise the original array of values.
    """
    if reduction == "mean":
        return float(np.mean(values))
    elif reduction == "sum":
        return float(np.sum(values))
    else:
        return values
