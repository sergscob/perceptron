import numpy as np
from training.activations import Activation, get_activation
from training.optim import get_optimizer

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None, w_init=None, optimizer="nesterov", learning_rate=0.05, momentum=0.9, beta2=0.999, eps=1e-8):

        self.activation = get_activation(activation or "sigmoid")
        self.w_init = w_init
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta2 = beta2
        self.eps = eps
        self.weight_optimizer = get_optimizer(optimizer, learning_rate=learning_rate, momentum=momentum, beta2=beta2, eps=eps)
        self.bias_optimizer = get_optimizer(optimizer, learning_rate=learning_rate, momentum=momentum, beta2=beta2, eps=eps)

        self.W = self._initialize_weights(input_size, output_size, w_init)
        self.b = np.zeros((1, output_size))

        self.input = None
        self.z = None
        self.output = None
        # print("layer initialized with weights sum:", self.W.sum())


    @staticmethod
    def _initialize_weights(input_size, output_size, w_init):
        if w_init == "zero":
            return np.zeros((input_size, output_size))

        if w_init == "heUniform":
            limit = np.sqrt(6.0 / input_size)
            return np.random.uniform(-limit, limit, size=(input_size, output_size))

        if w_init == "xavierUniform":
            limit = np.sqrt(6.0 / (input_size + output_size))
            return np.random.uniform(-limit, limit, size=(input_size, output_size))

        if w_init == "heNormal":
            std = np.sqrt(2.0 / input_size)
            return np.random.normal(0.0, std, size=(input_size, output_size))

        raise ValueError(
            f"Unsupported w_init: {w_init}. try heUniform, xavierUniform, heNormal"
        )


    @classmethod
    def fromJSON(cls, saved_params):
        layer = cls.__new__(cls)
        layer.activation = get_activation(saved_params["activation"])
        layer.w_init = saved_params.get("w_init")
        layer.optimizer_name = saved_params.get("optimizer", "nesterov")
        layer.learning_rate = saved_params.get("learning_rate", 0.05)
        layer.momentum = saved_params.get("momentum", 0.9)
        layer.beta2 = saved_params.get("beta2", 0.999)
        layer.eps = saved_params.get("eps", 1e-8)
        layer.weight_optimizer = get_optimizer(layer.optimizer_name, learning_rate=layer.learning_rate, momentum=layer.momentum, beta2=layer.beta2, eps=layer.eps)
        layer.bias_optimizer = get_optimizer(layer.optimizer_name, learning_rate=layer.learning_rate, momentum=layer.momentum, beta2=layer.beta2, eps=layer.eps)
        layer.input = None
        layer.z = None
        layer.output = None
        layer.W = np.asarray(saved_params["W"])
        layer.b = np.asarray(saved_params["b"])

        return layer


    def forward(self, x):

        self.input = x

        self.z = np.dot(x, self.W) + self.b
        self.output = self.activation.forward(self.z)

        return self.output


    def backward(self, dA):
        # dA = gradient from next layer

        m = self.input.shape[0]

        dZ = dA * self.activation.derivative(self.z)

        dW = np.dot(self.input.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        # gradient for previous layer
        dInput = np.dot(dZ, self.W.T)

        self.W = self.weight_optimizer.update(self.W, dW)
        self.b = self.bias_optimizer.update(self.b, db)

        return dInput

    
    def save(self):
        return {
            "type": self.__class__.__name__,
            "input_size": int(self.W.shape[0]),
            "output_size": int(self.W.shape[1]),
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "activation": self.activation.name,
            "w_init": self.w_init,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "beta2": self.beta2,
            "eps": self.eps,
        }
    
