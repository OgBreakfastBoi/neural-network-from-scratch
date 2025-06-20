from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

import numpy as np

from src.nn.errors import LayerNotBuiltError
from src.nn.optimizers.optimizer import Optimizer


class Layer(ABC):
    def __init__(self, name: str):
        self.name = name

        self._idx = -1  # The index of the layer
        self._built = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self._built:
            raise LayerNotBuiltError()

        return self.call(x)

    def __repr__(self):
        return f"<Layer: {self.name}, index={self._idx}>"

    @abstractmethod
    def build(self, *args, **kwargs):
        """
        `self._idx` should be set here and `self._built` should
        be set to True after the build process is complete.
        """
        raise NotImplementedError

    @abstractmethod
    def call(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(
        self,
        grad_input: np.ndarray,
        optimizer: Optimizer,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple[int]:
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def built(self):
        return self._built


class InputLayer:
    pass
