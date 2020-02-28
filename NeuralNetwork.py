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

    def train(self):
        pass

    def query(self, inputs):
        inputs = np.array(inputs)
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        out_inputs = np.dot(self.w_ho, hidden_outputs)
        out_outputs = self.activation_function(out_inputs)

        return out_outputs
