import numpy as np


def step(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(x):
    if x.ndim == 2:
        x_max = np.max(x, axis=1)
        x_exp = np.exp(x - x_max.reshape((x_max.shape[0], 1)))
        x_sum = np.sum(x_exp, axis=1)
        soft = x_exp / x_sum.reshape((x_sum.shape[0], 1))
        return soft
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp)
    soft = x_exp / x_sum
    return soft
