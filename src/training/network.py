import json

from training.layer import DenseLayer


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


    def save(self, filename):
        data = { "layers": [] }

        for layer in self.layers:
            data["layers"].append(layer.save())

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file)


    @classmethod
    def fromJSON(cls, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        network = cls()

        for saved_layer in data.get("layers", []):
            if saved_layer.get("type") != "DenseLayer":
                raise ValueError(f"Unsupported layer type: {saved_layer.get('type')}")

            layer = DenseLayer.fromJSON(saved_layer)
            network.add(layer)

        return network