import argparse
import sys
from dateutil import parser
import pandas as pd

from layer import DenseLayer
from network import Network
from prepare import normalize 
from split import split
from train import train
from utils import to_hot



def main():
    try:
        df = pd.read_csv("data/data.csv", header=None)
    except:
        print(f"Error: Could not load CSV from data/data.csv")
        sys.exit(1)

    y = df[1]
    y = y.map({ "M": 1, "B": 0 })
    y = y.to_numpy()

    X = df.iloc[:, 2:]
    X = X.to_numpy()
    X_train, X_valid, y_train, y_valid = split(X, y)
    X_train, X_valid = normalize(X_train, X_valid)
    # print (X_train.shape[1])

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", default=10, help="set the batch size")
    parser.add_argument("-n", "--epochs", default=100, help="set the number of epochs")
    parser.add_argument("-a", "--activation", default="sigmoid", help="set the activation function (sigmoid/relu)")
    parser.add_argument("-r", "--learning_rate", default=0.05, help="set the learning rate")
    parser.add_argument("-v", "--verbose", action="store_true", help="set verbose output")
    parser.add_argument("-l", "--layers", default="24 24 24", help="layers (24 24 24)")
    args = parser.parse_args()

    if (args.activation not in ["sigmoid", "relu"]):
        print (f"Invalid activation function: {args.activation}.")
        sys.exit(1)

    if (int(args.batch) <= 0 or int(args.batch) > len(X_train)):
        print (f"Invalid batch size: {args.batch}.")
        sys.exit(1)

    if (int(args.epochs) <= 0):
        print (f"Invalid number of epochs: {args.epochs}.")
        sys.exit(1)

    if (float(args.learning_rate) <= 0.00001 or float(args.learning_rate) > 100 or not isinstance(float(args.learning_rate), float)):
        print (f"Invalid learning rate: {args.learning_rate}. Setting to 0.01.")
        sys.exit(1)

    words = args.layers.split()
    layers = []
    for w in words:
        try:
            n = int(w)
            if n < 2 or n > 100:
                print (f"Size of layer must be 2-100")
                sys.exit(1)
            layers.append(n)
        except ValueError:
            print (f"Layers must be array of numbers 2-100")
            sys.exit(1)

    if len(layers) < 2 or len(layers) > 100:
        print (f"Count of layers must be 2-100")
        sys.exit(1)

    print (f"Using activation function: {args.activation}")
    print (f"Using layers: {layers}.")
    print (f"Using batch size: {args.batch}")
    print (f"Using epochs: {args.epochs}")
    print (f"Using learning rate: {args.learning_rate}")

    network = Network()
    network.add(DenseLayer(X_train.shape[1], 16))
    network.add(DenseLayer(16, 16))
    network.add(DenseLayer(16, 2, activation="none"))

    y_train = to_hot(y_train)
    y_valid = to_hot(y_valid)

    train(network, X_train, y_train, X_valid, y_valid, args)








if __name__ == "__main__":
    main()  