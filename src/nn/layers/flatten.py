import math
from typing import Any

import numpy as np

from src.nn.errors import LayerNotBuiltError
from src.nn.layers.layer import (
    InputLayer,
    Layer,
)
from src.nn.optimizers.optimizer import Optimizer


class Flatten(Layer, InputLayer):
    def __init__(self, input_shape: tuple[int, int], name = "Flatten"):
        super().__init__(name)
        self.input_shape = input_shape

    def build(self, index: int):
        self._idx = index
        self._built = True

    def call(self, x: np.ndarray) -> np.ndarray:
        output_shape = self.output_shape()[-1]

        if x.ndim == 1 and x.shape[0] != output_shape:
            raise ValueError(
                f"`x` input shape is invalid for {self.__repr__()}. Expected "
                f"shape ({output_shape},), got {x.shape} instead."
            )

        if x.ndim == 2 and x.shape[1] != output_shape:
            # Flatten `x` and recheck shape
            x = x.flatten()
            if x.shape[0] != output_shape:
                raise ValueError(
                    f"`x` input shape is invalid for {self.__repr__()}. "
                    f"Expected shape {(x.shape[0], output_shape)}, "
                    f"got {x.shape} instead."
                )

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
            if x.shape[1] != output_shape:
                raise ValueError(
                    f"`x` input shape is invalid for {self.__repr__()}. "
                    f"Expected shape {(x.shape[0], output_shape)}, "
                    f"got {x.shape} instead."
                )

        return x

    def backward(
        self,
        grad_input: np.ndarray,
        optimizer: Optimizer,
    ) -> np.ndarray:
        if not self._built:
            raise LayerNotBuiltError()
        return np.array([])

    def output_shape(self) -> tuple[int]:
        return (math.prod(self.input_shape),)

    def get_config(self) -> dict[str, Any]:
        if not self._built:
            raise LayerNotBuiltError(
                "Layer must be built before a config can be generated."
            )

        config = {
            "name": self.name,
            "index": self._idx,
            "input_shape": self.input_shape
        }
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Flatten":
        layer = cls(input_shape=config['input_shape'], name=config['name'])
        layer._idx = config['index']
        layer._built = True
        return layer
