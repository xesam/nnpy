# Activation Function

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
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    sum_x = np.sum(np.exp(x))
    return x_exp / sum_x
