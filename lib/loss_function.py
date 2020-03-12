import numpy as np


def mean_square_error(y, t):
    return np.sum((y - t) ** 2) / 2


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
