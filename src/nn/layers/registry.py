from typing import Any

from src.nn.layers.dense import Dense
from src.nn.layers.flatten import Flatten
from src.nn.layers.layer import Layer

LAYERS = [
    Dense,
    Flatten
]
LAYERS_DICT = {l.__name__: l for l in LAYERS}


def from_config(config: dict[str, Any]) -> Layer:
    if isinstance(config['name'], str):
        name = config['name']
        if name in LAYERS_DICT.keys():
            return LAYERS_DICT[name].from_config(config)
        raise ValueError(
            f"Unknown Layer '{name}'. Supported: {list(LAYERS_DICT.keys())}"
        )
    raise TypeError(
        f"ActivationFunction identifier must be a string or None, "
        f"not {type(config['name'])}"
    )
