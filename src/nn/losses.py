import numpy as np


def mean_squared_error(x_true: np.ndarray, x_predicted: np.ndarray):
    squared_error = (x_true - x_predicted) ** 2
    return np.mean(squared_error)


def cross_categorical_entropy_loss(x_true: np.ndarray, x_predicted: np.ndarray):
    if len(x_true) != len(x_predicted):
        raise ValueError(f"x_true {x_true.shape} and x_predicted {x_predicted.shape} have different shapes")

    return -1 / len(x_true) * np.sum(x_true * np.log(x_predicted))
