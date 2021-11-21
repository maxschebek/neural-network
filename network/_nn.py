import numpy as np
from typing import List

from numpy.lib.function_base import append


class Sequential:
    def __init__(self, layer_dimensions: List[int]):
        self.n_layers = len(layer_dimensions)
        self.n_matrices = self.n_layers - 1
        np.random.seed(seed=0)
        self.weight_matrices = [
            np.random.uniform(-1, 1, (layer_dimensions[i + 1], layer_dimensions[i] + 1))
            for i in range(self.n_matrices)
        ]

    def propagate_forward(self, input: np.ndarray) -> np.ndarray:
        a_with_bias = np.append(1, input)
        for i_matrix in range(self.n_matrices):
            z = np.matmul(self.weight_matrices[i_matrix], a_with_bias)
            a = np.array(list(map(sigmoid, z)))
            a_with_bias = np.append(1, a)
        return a

    def train(self, input: np.ndarray) -> np.ndarray:
        self.propagate_forward(input)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))
