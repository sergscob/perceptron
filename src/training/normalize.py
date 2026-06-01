import pandas as pd
import numpy as np

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