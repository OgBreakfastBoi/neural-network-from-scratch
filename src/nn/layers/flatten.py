import math

import numpy as np

from src.nn.layers.layer import Layer


class Flatten(Layer):
    def __init__(self, input_shape: tuple[int, int]):
        super().__init__()
        self.input_shape = input_shape

    def build(self, *args, **kwargs):
        pass

    def call(self, x: np.ndarray) -> np.ndarray:
        output_shape = self.compute_output_shape()[-1]

        if x.ndim == 1 and x.shape[0] != output_shape:
            raise ValueError(
                f"`x` input shape is invalid for {self.__repr__()}. Expected shape "
                f"({output_shape},), got {x.shape} instead."
            )

        if x.ndim == 2 and x.shape[1] != output_shape:
            # Flatten `x` and recheck shape
            x = x.flatten()
            if x.shape[0] != output_shape:
                raise ValueError(
                    f"`x` input shape is invalid for {self.__repr__()}. Expected shape "
                    f"{(x.shape[0], output_shape)}, got {x.shape} instead."
                )

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
            if x.shape[1] != output_shape:
                raise ValueError(
                    f"`x` input shape is invalid for {self.__repr__()}. Expected shape "
                    f"{(x.shape[0], output_shape)}, got {x.shape} instead."
                )

        return x

    def compute_output_shape(self) -> tuple[int]:
        return (math.prod(self.input_shape),)
