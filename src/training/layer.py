import numpy as np
from training.activations import sigmoid, sigmoid_derivative, softmax, relu, relu_derivative

class DenseLayer:
    def __init__(self, input_size, output_size, activation="sigmoid"):

        self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros((1, output_size))

        self.activation = activation

        self.input = None
        self.z = None
        self.output = None


    @classmethod
    def fromJSON(cls, saved_params):
        layer = cls(
            input_size=saved_params["input_size"],
            output_size=saved_params["output_size"],
            activation=saved_params["activation"]
        )
        layer.W = np.asarray(saved_params["W"])
        layer.b = np.asarray(saved_params["b"])

        return layer


    def forward(self, x):

        self.input = x

        # linear part
        self.z = np.dot(x, self.W) + self.b

        # activation
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

        # activation gradient
        if self.activation == "sigmoid":
            dZ = dA * sigmoid_derivative(self.z)
        elif self.activation == "relu":
            dZ = dA * relu_derivative(self.z)
        else:
            # no activation (linear)
            dZ = dA
        # gradients
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
            "activation": self.activation
        }
    
