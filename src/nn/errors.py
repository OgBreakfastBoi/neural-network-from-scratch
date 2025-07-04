class NoLayersError(Exception):
    """Raised when attempting to compile a model that has no layers."""

    def __init__(self, message = "Cannot compile model: no layers have been added."):
        super().__init__(message)


class ModelNotCompiledError(Exception):
    """Raised when attempting to use a neural network that hasn't been compiled."""

    def __init__(self, message = "The model must be compiled before it can be trained or evaluated."):
        super().__init__(message)


class LayerNotBuiltError(Exception):
    """Raised when attempting to use a layer that hasn't been built."""

    def __init__(self, message = "The layer must be built before it can be called."):
        super().__init__(message)
