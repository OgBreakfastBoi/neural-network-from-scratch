import numpy as np


def mean_squared_error(true: np.ndarray, predicted: np.ndarray):
    squared_error = (true - predicted) ** 2
    return np.mean(squared_error)


def cross_categorical_entropy_loss(true: np.ndarray, predicted: np.ndarray):
    if true.shape != predicted.shape:
        raise ValueError(f"x_true {true.shape} and x_predicted {predicted.shape} have different shapes")

    epsilon = 1e-15  # Very small lower bound to prevent divide by zero error
    predicted = np.clip(predicted, epsilon, 1.0)  # Ensures values are within [epsilon, 1] for log calc
    return -np.mean(np.sum(true * np.log(predicted), axis=1))
