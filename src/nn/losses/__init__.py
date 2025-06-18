"""
This package provides loss functions for the neural network.

It exposes the base ``LossFunction`` class, classes of common loss functions,
and a ``get`` function to retrieve loss functions by name.
"""

from src.nn.losses.loss import LossFunction
from src.nn.losses.losses import (
    CategoricalCrossEntropy,
    MeanSquaredError,
)
from src.nn.losses.registry import get

__all__ = [
    "LossFunction",
    "CategoricalCrossEntropy",
    "MeanSquaredError",
    "get"
]
