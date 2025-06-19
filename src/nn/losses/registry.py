from src.nn.losses.loss import LossFunction
from src.nn.losses.losses import (
    CategoricalCrossEntropy,
    MeanSquaredError,
)

LOSSES = [
    CategoricalCrossEntropy,
    MeanSquaredError
]
LOSSES_DICT = {fn().name: fn for fn in LOSSES}


def get(identifier: str) -> LossFunction:
    """
    Retrieves a loss function instance by its string identifier.

    Args:
        identifier: The name of the loss function to retrieve.

    Returns:
        An instance of the requested loss function.

    Raises:
        ValueError: If the identifier is not a supported loss function.
        TypeError: If the identifier is not a string.
    """

    if isinstance(identifier, str):
        key = identifier.lower()
        if key in LOSSES_DICT:
            return LOSSES_DICT[key]()
        raise ValueError(
            f"Unknown loss function '{identifier}'. "
            f"Supported: {list(LOSSES_DICT.keys())}"
        )
    raise TypeError(
        f"Loss function identifier must be a string, "
        f"not {type(identifier)}"
    )
