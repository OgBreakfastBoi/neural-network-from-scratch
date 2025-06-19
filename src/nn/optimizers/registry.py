from src.nn.optimizers.optimizer import Optimizer
from src.nn.optimizers.optimizers import (
    Adam,
    StochasticGradientDescent,
)

OPTIMIZERS = [
    Adam,
    StochasticGradientDescent,
]
OPTIMIZERS_DICT = {fn().name: fn for fn in OPTIMIZERS}


def get(identifier: str) -> Optimizer:
    if isinstance(identifier, str):
        key = identifier.lower()
        if key in OPTIMIZERS_DICT:
            return OPTIMIZERS_DICT[key]()
        raise ValueError(
            f"Unknown optimizer '{identifier}'. "
            f"Supported: {list(OPTIMIZERS_DICT.keys())}"
        )
    raise TypeError(
        f"Optimizer identifier must be a string, not {type(identifier)}"
    )
