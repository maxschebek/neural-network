import numpy as np
from typing import List


class Sequential:
    def __init__(self, layer_dims: List[int]):
        self.n_layers = len(layer_dims)
        self.n_weights = self.n_layers - 1
        np.random.seed(seed=0)
        self.weight_dims = [
            (layer_dims[i + 1], layer_dims[i] + 1) for i in range(self.n_weights)
        ]
        self.weights = [
            np.random.uniform(-1, 1, self.weight_dims[i]) for i in range(self.n_weights)
        ]
        self.accumulators = [
            np.zeros(self.weight_dims[i]) for i in range(self.n_weights)
        ]
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

    def train(self, input: np.ndarray) -> np.ndarray:
        self.propagate_forward(input)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def costfunc(y, y_pred):
    n_samples = np.shape(y)[0]
    cost = 0
    for isp in range(n_samples):
        cost += -np.dot(y[isp, :], np.log(y_pred[isp, :])) - np.dot(
            (1 - y[isp, :]), np.log(1 - y_pred[isp, :])
        )
    return 1 / n_samples * cost
