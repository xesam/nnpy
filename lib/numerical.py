import numpy as np


def numerical_gradient_1d(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)
    for i in range(x.size):
        x_val = x[i]
        x[i] = x_val - h
        left = f(x)  # f(x-h)
        x[i] = x_val + h
        right = f(x)  # f(x+h)
        gradient[i] = (right - left) / (2 * h)
        x[i] = x_val
    return gradient


def numerical_gradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        x_val = x[idx]
        x[idx] = x_val - h
        left = f(x)  # f(x-h)
        x[idx] = x_val + h
        right = f(x)  # f(x+h)
        gradient[idx] = (right - left) / (2 * h)
        x[idx] = x_val
        it.iternext()
    return gradient
