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

        velocity_momentum = self.momentum * self.velocity
        velocity_gradient = self.learning_rate * gradients

        self.velocity = velocity_momentum - velocity_gradient
        # self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
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

        velocity_momentum = self.momentum * previous_velocity
        velocity_gradient = self.learning_rate * gradients

        self.velocity = velocity_momentum - velocity_gradient

        lookahead_correction = -self.momentum * previous_velocity
        accelerated_step = (1.0 + self.momentum) * self.velocity

        parameter_update = lookahead_correction + accelerated_step

        return params + parameter_update
    

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



def get_optimizer(name, learning_rate=0.01, momentum=0.9, beta2=0.999, eps=1e-8):
    if isinstance(name, Optimizer):
        return name
    if name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate)
    if name == "momentum":
        return MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    if name == "nesterov":
        return NesterovMomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    if name == "rmsprop":
        return RMSPropOptimizer(learning_rate=learning_rate, rho=momentum )
    if name == "adam":
        return AdamOptimizer(learning_rate=learning_rate, beta1=momentum, beta2=beta2, eps=eps)

    raise ValueError(f"Unsupported optimizer: {name}")