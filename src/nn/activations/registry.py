from src.nn.activations.activation import ActivationFunction
from src.nn.activations.activations import (
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
)

ACTIVATIONS = [
    Linear,
    ReLU,
    Sigmoid,
    Softmax
]
ACTIVATIONS_DICT = {fn().name: fn for fn in ACTIVATIONS}


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
        return Linear()
    elif isinstance(identifier, str):
        key = identifier.lower()
        if key in ACTIVATIONS_DICT:
            return ACTIVATIONS_DICT[key]()
        raise ValueError(
            f"Unknown activation '{identifier}'. "
            f"Supported: {list(ACTIVATIONS_DICT.keys())}"
        )
    raise TypeError(
        f"ActivationFunction identifier must be a string or None, "
        f"not {type(identifier)}"
    )
