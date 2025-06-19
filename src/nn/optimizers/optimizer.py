from abc import (
    ABC,
    abstractmethod,
)

import numpy as np


class Optimizer(ABC):
    def __init__(self, lr: float, name: str):
        self.lr = lr
        self.name = name

    def __repr__(self):
        return f"<Optimizer: {self.name}, lr={self.lr}>"

    @abstractmethod
    def step(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        param_name: str,
    ) -> np.ndarray:
        raise NotImplementedError
