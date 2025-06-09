from abc import (
    ABC,
    abstractmethod,
)

import numpy as np


class Layer(ABC):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __repr__(self):
        return f"<Layer: {self.__class__.__name__}>"

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def call(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_output_shape(self) -> tuple[int]:
        raise NotImplementedError
