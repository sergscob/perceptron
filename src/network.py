import numpy as np


class Network:
    def __init__(self):
        self.layers = []


    def add(self, layer):
        self.layers.append(layer)


    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x


    def backward(self, gradient, learning_rate):
        # reverse order
        for l in reversed(self.layers):
            gradient = l.backward(gradient, learning_rate)


    # def predict(self, x):
    #     output = self.forward(x)
    #     return np.argmax(output, axis=1)