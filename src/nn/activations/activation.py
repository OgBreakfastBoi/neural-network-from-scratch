from abc import (
    ABC,
    abstractmethod,
)

import numpy as np


class ActivationFunction(ABC):
    """
    Abstract base class for all activation functions.

    Subclasses must implement the following:
      - ``call(x)``: Applies the activation function to the input array.
      - ``deriv(x)``: Computes the derivative of the activation function with respect to input.
      - ``static_call(x)``: Static version of the activation.

    Methods:
        __call__(x): Enables instances to be called like functions, forwarding to `call`.
        __repr__(): Returns a string representation of the activation function.
        call(x): Abstract method to compute the forward pass.
        deriv(x): Abstract method to compute the derivative of the activation function.
        static_call(x): Abstract static method for applying the activation without instantiation.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.call(x)

    def __repr__(self):
        return f"<ActivationFunction: {self.name}>"

    @abstractmethod
    def call(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def deriv(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def static_call(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
