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