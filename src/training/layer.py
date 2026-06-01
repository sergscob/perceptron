import numpy as np
from training.activations import sigmoid, sigmoid_derivative, softmax, relu, relu_derivative

class DenseLayer:
    def __init__(self, input_size, output_size, activation="sigmoid", w_init=None):

        self.activation = activation
        self.w_init = w_init

        self.W = self._initialize_weights(input_size, output_size, w_init)
        self.b = np.zeros((1, output_size))

        self.input = None
        self.z = None
        self.output = None
        # print("layer initialized with weights sum:", self.W.sum())


    @staticmethod
    def _initialize_weights(input_size, output_size, w_init):
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
            f"Unsupported w_init: {w_init}. "
            "Supported values: heUniform, xavierUniform, heNormal"
        )


    @classmethod
    def fromJSON(cls, saved_params):
        layer = cls.__new__(cls)
        layer.activation = saved_params["activation"]
        layer.w_init = saved_params.get("w_init")
        layer.input = None
        layer.z = None
        layer.output = None
        layer.W = np.asarray(saved_params["W"])
        layer.b = np.asarray(saved_params["b"])

        return layer


    def forward(self, x):

        self.input = x

        self.z = np.dot(x, self.W) + self.b

        if self.activation == "sigmoid":
            self.output = sigmoid(self.z)
        elif self.activation == "relu":
            self.output = relu(self.z)
        else:
            self.output = self.z

        return self.output


    def backward(self, dA, learning_rate):

        """
        dA = gradient from next layer
        """

        m = self.input.shape[0]

        if self.activation == "sigmoid":
            dZ = dA * sigmoid_derivative(self.z)
        elif self.activation == "relu":
            dZ = dA * relu_derivative(self.z)
        else:
            dZ = dA

        dW = np.dot(self.input.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        # gradient for previous layer
        dInput = np.dot(dZ, self.W.T)

        # gradient descent update
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dInput

    
    def save(self):
        return {
            "type": self.__class__.__name__,
            "input_size": int(self.W.shape[0]),
            "output_size": int(self.W.shape[1]),
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "activation": self.activation,
            "w_init": self.w_init,
        }
    
