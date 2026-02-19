import numpy as np

def generate_linear_data(n=200000):
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def generate_non_linear_data(n=200000):
    X = np.random.randn(n, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)
    return X, y
