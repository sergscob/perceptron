import argparse
import sys
import numpy as np

from training.activations import softmax
from training.network import Network
from training.utils import loadCSV


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    zero = 1e-15
    y_pred = np.clip(y_pred, zero, 1 - zero)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="result/model.json", help="path to saved model")
    parser.add_argument("-d", "--data", default="data/valid.csv", help="path to dataset for prediction")
    args = parser.parse_args()

    try:
        network = Network.fromJSON(args.model)
    except Exception:
        print(f"Error: Could not load model from {args.model}")
        sys.exit(1)

    X, y = loadCSV(args.data)

    normalization = getattr(network, "normalization", None)
    if normalization is None:
        print("Error: model.json does not contain normalization data")
        sys.exit(1)

    try:
        mean = np.asarray(normalization["mean"])
        std = np.asarray(normalization["std"])
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std
    except Exception:
        print("Error: failed to apply normalization")
        sys.exit(1)

    logits = network.forward(X)
    probabilities = softmax(logits)
    positive_class_prob = probabilities[:, 1]

    predictions = (positive_class_prob >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == y))
    loss = binary_cross_entropy(y, positive_class_prob)

    print(f"Loaded model from {args.model}")
    print(f"Loaded dataset from {args.data}")
    print(f"Prediction accuracy: {accuracy:.4f}")
    print(f"Binary cross-entropy: {loss:.4f}")


if __name__ == "__main__":
    main()