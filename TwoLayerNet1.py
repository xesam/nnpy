import numpy as np

from lib.activation import softmax, sigmoid
from lib.loss_function import cross_entropy_error
from lib.numerical import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        e = cross_entropy_error(y, t)
        return e

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        lossW = lambda W: self.loss(x, t)
        gradient = {}
        gradient['W1'] = numerical_gradient(lossW, self.params['W1'])
        gradient['b1'] = numerical_gradient(lossW, self.params['b1'])
        gradient['W2'] = numerical_gradient(lossW, self.params['W2'])
        gradient['b2'] = numerical_gradient(lossW, self.params['b2'])
        return gradient
