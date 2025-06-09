from src.nn.activations.activations import (
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
)
from src.nn.activations.activation import ActivationFunction

linear = Linear()
relu = ReLU()
sigmoid = Sigmoid()
softmax = Softmax()

ALL_OBJECTS = [
    linear,
    relu,
    sigmoid,
    softmax,
]
ALL_OBJECTS_DICT = {fn.__class__.__name__.lower(): fn for fn in ALL_OBJECTS}


def get(identifier: str | None) -> ActivationFunction:
    """
    Retrieve activation function via an identifier.

    Args:
        identifier (str | None): The name of the activation function.

            Supported identifiers:
              - 'linear'
              - 'relu'
              - 'sigmoid'
              - 'softmax'
            If None, returns the linear activation.

    Returns:
        Activation: The corresponding activation function instance.

    Raises:
        ValueError: If the identifier is not recognized.
        TypeError: If identifier is not a string or None.
    """

    if identifier is None:
        return linear
    elif isinstance(identifier, str):
        key = identifier.lower()
        if key in ALL_OBJECTS_DICT:
            return ALL_OBJECTS_DICT[key]
        raise ValueError(
            f"Unknown activation '{identifier}'. Supported: {list(ALL_OBJECTS_DICT.keys())}"
        )
    raise TypeError(f"ActivationFunction identifier must be a string or None, not {type(identifier)}")
