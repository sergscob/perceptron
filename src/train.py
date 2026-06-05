import argparse
import sys
import os
import random
import numpy as np

from training.layer import DenseLayer
from training.network import Network
from training.training import train
from training.utils import to_hot, loadCSV, normalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", default=20, help="set the batch size")
    parser.add_argument("-n", "--epochs", default=100, help="set the number of epochs")
    parser.add_argument("-a", "--activation", default="sigmoid", choices=["sigmoid", "relu"], help="set the activation function (sigmoid/relu)")
    parser.add_argument("-r", "--learning_rate", default=0.05, help="set the learning rate")
    parser.add_argument("-l", "--layers", default="24 24", help="layers (24 24)")
    parser.add_argument("-w", "--w_init", default="heUniform", choices=["heUniform", "xavierUniform", "heNormal", "zero"], help="set the weights initializer")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="set verbose output")
    parser.add_argument("-s", "--seed", type=int, default=None, help="set random seed")
    
    parser.add_argument("-p", "--patience", type=int, default=0, help="stop after this many epochs without validation improvement; 0 disables early stopping")
    parser.add_argument("-d", "--min_delta", type=float, default=0.001, help="minimum validation loss improvement required to reset patience")

    parser.add_argument("-o", "--optimizer", default="sgd", choices=["sgd", "momentum", "nesterov", "rmsprop", "adam"], help="set the optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="set the momentum value for momentum-based optimizers")
    parser.add_argument("--beta2", type=float, default=0.999, help="set beta2 for Adam (and rho for RMSProp if desired)")
    parser.add_argument("--eps", type=float, default=1e-8, help="set epsilon for Adam/RMSProp numeric stability")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    X_train, y_train = loadCSV("data/train.csv")
    X_valid, y_valid = loadCSV("data/valid.csv")

    X_train, X_valid, mean, std = normalize(X_train, X_valid)
    # print (X_train.shape)

    if (args.activation not in ["sigmoid", "relu"]):
        print (f"Invalid activation function: {args.activation}.")
        sys.exit(1)

    if (int(args.batch) <= 0 or int(args.batch) > len(X_train)):
        print (f"Invalid batch size: {args.batch}. max={len(X_train)}")
        sys.exit(1)

    if (int(args.epochs) <= 0 or int(args.epochs) > 100000):
        print (f"Invalid number of epochs: {args.epochs}.")
        sys.exit(1)

    if (int(args.patience) < 0):
        print (f"Invalid patience: {args.patience}.")
        sys.exit(1)

    if (float(args.min_delta) < 0):
        print (f"Invalid min_delta: {args.min_delta}.")
        sys.exit(1)

    if (float(args.learning_rate) <= 0.00001 or float(args.learning_rate) > 100 or not isinstance(float(args.learning_rate), float)):
        print (f"Invalid learning rate: {args.learning_rate}. Setting to 0.01.")
        sys.exit(1)

    words = args.layers.split()
    layers = []
    for w in words:
        try:
            n = int(w)
            if n < 2 or n > 10000:
                print (f"Size of layer must be 2-1000")
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
    if (args.patience < 0):
        print (f"Using early stopping: patience={args.patience}, min_delta={args.min_delta}")
    print (f"Using learning rate: {args.learning_rate}")
    print (f"Using optimizer: {args.optimizer}")
    print (f"Using momentum: {args.momentum}")
    print (f"Using weights init: {args.w_init}")
    os.makedirs("result/charts", exist_ok=True)

    network = Network()
    network.add(DenseLayer(X_train.shape[1], layers[0], activation=args.activation, w_init=args.w_init, optimizer=args.optimizer, learning_rate=float(args.learning_rate), momentum=float(args.momentum), beta2=float(args.beta2), eps=float(args.eps)))
    for i in range(1, len(layers)):
        network.add(DenseLayer(layers[i-1], layers[i], activation=args.activation, w_init=args.w_init, optimizer=args.optimizer, learning_rate=float(args.learning_rate), momentum=float(args.momentum), beta2=float(args.beta2), eps=float(args.eps)))
    network.add(DenseLayer(layers[-1], 2, activation="none", w_init=args.w_init, optimizer=args.optimizer, learning_rate=float(args.learning_rate), momentum=float(args.momentum), beta2=float(args.beta2), eps=float(args.eps)))

    y_train = to_hot(y_train)
    y_valid = to_hot(y_valid)

    train(network, X_train, y_train, X_valid, y_valid, args)

    network.normalization = {"mean": mean.tolist(), "std": std.tolist()}
    network.save("result/model.json")
    print ("Model saved to result/model.json")


if __name__ == "__main__":
    main()  