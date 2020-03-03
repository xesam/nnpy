import numpy as np

from lib import activation


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.w_ih = np.random.normal(0, pow(self.hidden_nodes, 0.5), (self.hidden_nodes, self.input_nodes))
        self.w_ho = np.random.normal(0, pow(self.output_nodes, 0.5), (self.output_nodes, self.hidden_nodes))
        self.activation_function = activation.sigmoid

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        out_inputs = np.dot(self.w_ho, hidden_outputs)
        out_outputs = self.activation_function(out_inputs)

        out_errors = targets - out_outputs
        hidden_errors = np.dot(self.w_ho.T, out_errors)

        w_ho_delta = np.dot(out_errors * out_outputs * (1 - out_outputs), hidden_outputs.T)
        self.w_ho += self.learning_rate * w_ho_delta

        w_ih_delta = np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)
        self.w_ih += self.learning_rate * w_ih_delta

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        out_inputs = np.dot(self.w_ho, hidden_outputs)
        out_outputs = self.activation_function(out_inputs)

        return out_outputs
