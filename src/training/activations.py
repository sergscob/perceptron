import numpy as np


class Activation:
    name = "activation"

    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class SigmoidActivation(Activation):
    name = "sigmoid"

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        output = self.forward(x)
        return output * (1 - output)


class ReLUActivation(Activation):
    name = "relu"

    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class EmptyActivation(Activation):
    name = "none"

    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


def get_activation(name):
    activations = {
        "sigmoid": SigmoidActivation,
        "relu": ReLUActivation,
        "none": EmptyActivation,
    }

    if isinstance(name, Activation):
        return name

    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)