import pandas as pd
import numpy as np

def normalize(X_train, X_valid):

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std

    return X_train, X_valid