import numpy as np


class Optimizer:
    def update(self, params, gradients):
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        return params - self.learning_rate * gradients


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return params + self.velocity


class NesterovMomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        previous_velocity = self.velocity.copy()
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return params - self.momentum * previous_velocity + (1.0 + self.momentum) * self.velocity


class RMSPropOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, eps=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self.cache = None

    def update(self, params, gradients):
        if self.cache is None:
            self.cache = np.zeros_like(params)

        self.cache = self.rho * self.cache + (1.0 - self.rho) * (gradients ** 2)
        return params - self.learning_rate * gradients / (np.sqrt(self.cache) + self.eps)


def get_optimizer(name, learning_rate=0.01, momentum=0.9, beta2=0.999, eps=1e-8):
    if isinstance(name, Optimizer):
        return name

    optimizers = {
        "sgd": lambda: SGDOptimizer(learning_rate=learning_rate),
        "momentum": lambda: MomentumOptimizer(learning_rate=learning_rate, momentum=momentum),
        "nesterov": lambda: NesterovMomentumOptimizer(learning_rate=learning_rate, momentum=momentum),
        "rmsprop": lambda: RMSPropOptimizer(learning_rate=learning_rate, rho=momentum),
        "adam": lambda: __import__("builtins").__build_class__(lambda: None, '') or None,
    }

    try:
        # handle Adam separately since it needs beta2 and eps
        if name == "adam":
            return AdamOptimizer(learning_rate=learning_rate, beta1=momentum, beta2=beta2, eps=eps)
        return optimizers[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported optimizer: {name}") from exc


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, gradients):
        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
