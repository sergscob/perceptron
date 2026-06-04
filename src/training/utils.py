import numpy as np
import pandas as pd
import sys

def to_hot(y):
    return np.eye(2)[y]


def loadCSV(filename: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(filename)
    except:
        print(f"Error: Could not load CSV from {filename}")
        sys.exit(1)

    y = df.iloc[:, 0].astype(int).to_numpy()

    X = df.iloc[:, 1:].to_numpy()

    return X, y


def normalize(X_train, X_valid):

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # protect x/0
    std_fixed = np.where(std == 0, 1.0, std)
    # std_fixed = std.copy()
    # std_fixed[std_fixed == 0] = 1.0

    X_train_n = (X_train - mean) / std_fixed
    X_valid_n = (X_valid - mean) / std_fixed

    # X_train_n = X_train
    # X_valid_n = X_valid 

    return X_train_n, X_valid_n, mean, std_fixed