import numpy as np

from src.nn.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        lr = 0.001,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-15,
        name = "adam",
    ):
        super().__init__(lr, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def step(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        param_name: str,
    ) -> np.ndarray:
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(grad)
            self.v[param_name] = np.zeros_like(grad)
            self.t[param_name] = 0

        self.t[param_name] += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t[param_name])
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t[param_name])

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class StochasticGradientDescent(Optimizer):
    def __init__(
        self,
        lr = 0.01,
        name = "stochastic_gradient_decent",
    ):
        super().__init__(lr, name)

    def step(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        param_name: str,
    ) -> np.ndarray:
        return param - self.lr * grad
