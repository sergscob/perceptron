import argparse
import sys
import pandas as pd
import numpy as np

def split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    
    indices = np.random.permutation(len(X))
    split = int(train_ratio * len(X))
    
    train_idx = indices[:split]
    valid_idx = indices[split:]

    X_train = X[train_idx]
    X_valid = X[valid_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]

    return X_train, X_valid, y_train, y_valid   


def saveCSV(X: np.ndarray, y: np.ndarray, filename: str) -> None:
    X_df = pd.DataFrame(X)
    y_series = pd.Series(np.asarray(y).reshape(-1), name="y")
    df = pd.concat([y_series, X_df ], axis=1)
    df.to_csv(filename, index=False)


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42, help="set random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

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
    saveCSV(X_train, y_train, "data/train.csv")
    saveCSV(X_valid, y_valid, "data/valid.csv")


if __name__ == "__main__":
    main()

