import json

from training.layer import DenseLayer


class Network:
    def __init__(self):
        self.layers = []
        self.normalization = None


    def add(self, layer):
        self.layers.append(layer)


    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x


    def layer_stats(self, x):
        stds = []
        means = []
        alives = []

        for layer in self.layers[:-1]:
            x = layer.forward(x)
            stds.append(float(x.std()))
            means.append(float(x.mean()))
            alives.append(float((layer.z > 0).mean()))

        return {"stds": stds, "means": means, "alive": alives}


    def backward(self, gradient, learning_rate):
        for l in reversed(self.layers):
            gradient = l.backward(gradient, learning_rate)


    def save(self, filename):
        data = { "layers": [] }

        for layer in self.layers:
            data["layers"].append(layer.save())

        if self.normalization is not None:
            data["normalization"] = self.normalization

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

        normalization = data.get("normalization")
        if normalization is not None:
            network.normalization = normalization

        return network