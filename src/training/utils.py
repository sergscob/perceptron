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
