"""
This package provides loss functions for the neural network.

It exposes the base ``LossFunction`` class, instances of common loss functions,
and a ``get`` function to retrieve loss functions by name.
"""

from src.nn.losses.loss import LossFunction
from src.nn.losses.registry import (
    categorical_cross_entropy,
    get,
    mean_squared_error,
)

__all__ = [
    "LossFunction",
    "categorical_cross_entropy",
    "mean_squared_error",
    "get"
]
