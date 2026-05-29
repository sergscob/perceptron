import numpy as np

def to_hot(y):
    return np.eye(2)[y]