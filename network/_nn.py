import numpy as np
from typing import List
import copy


class Sequential:
    def __init__(self, layer_dims: List[int], weights: List[np.ndarray]):
        self.n_layers = len(layer_dims)
        self.n_weights = self.n_layers - 1
        np.random.seed(seed=0)
        self.weight_dims = [
            (layer_dims[i + 1], layer_dims[i] + 1) for i in range(self.n_weights)
        ]
        self.weights = copy.deepcopy(weights)
        self.accumulators = [
            np.zeros(self.weight_dims[i]) for i in range(self.n_weights)
        ]
        self.gradients = [np.zeros(self.weight_dims[i]) for i in range(self.n_weights)]
        self.layers = [
            np.ones(layer_dims[i_lay] + 1) for i_lay in range(self.n_layers - 1)
        ]
        self.layers.append(np.ones(layer_dims[-1]))

        self.layers_b = [np.zeros(layer_dims[i_lay]) for i_lay in range(self.n_layers)]
        self.layer_errors = [
            np.zeros(layer_dims[i_lay]) for i_lay in range(self.n_layers)
        ]

    def propagate_forward(self, input: np.ndarray) -> np.ndarray:
        self.layers[0][1:] = input
        for i_lay in range(1, self.n_layers - 1):
            z = np.matmul(self.weights[i_lay - 1], self.layers[i_lay - 1])
            self.layers[i_lay][1:] = np.array(list(map(sigmoid, z)))
            self.layers_b[i_lay] = np.array(list(map(sigmoid_derivative, z)))
        # Last step explicit
        z = np.matmul(self.weights[-1], self.layers[-2])
        self.layers[-1] = np.array(list(map(sigmoid, z)))
        self.layers_b[-1][:] = 0.0
        return self.layers[-1]

    def propagate_backward(self, error: np.ndarray):

        self.layer_errors[-1] = error
        self.accumulators[-1] += np.outer(self.layer_errors[-1], self.layers[-2])
        for i_lay in reversed(range(1, self.n_layers - 1)):
            aux_vector = np.matmul(self.weights[i_lay].T, self.layer_errors[i_lay + 1])
            self.layer_errors[i_lay] = np.multiply(aux_vector[1:], self.layers_b[i_lay])
            self.accumulators[i_lay - 1] += np.outer(
                self.layer_errors[i_lay], self.layers[i_lay - 1]
            )

    def make_gradients(self, reg, batch_size):
        for i in range(self.n_weights):
            self.gradients[i] = self.accumulators[i] / batch_size
            self.gradients[i][:, 1:] += reg / batch_size * self.weights[i][:, 1:]

    def zero_gradients(self):
        for i in range(self.n_weights):
            self.accumulators[i] = 0.0
            self.gradients[i] = 0.0

    def __call__(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.propagate_forward(X)
        self.propagate_backward((y_pred - y))
        return y_pred


def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float):
    return sigmoid(x) * (1 - sigmoid(x))


def costfunc(y: np.ndarray, y_pred: np.ndarray):
    cost = -np.dot(y, np.log(y_pred)) - np.dot((1 - y), np.log(1 - y_pred))
    return cost


def regularization_term(weights, reg):
    weights_2 = [np.square(weight[:, 1:]) for weight in weights]
    weights_sum = [np.sum(weight_2) for weight_2 in weights_2]
    return reg / 2 * sum(weights_sum)


def output_class(y: np.ndarray):
    y_class = np.zeros(len(y))
    y_class[np.where(y == np.max(y))[0]] = 1
    return y_class

