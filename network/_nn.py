import numpy as np
from typing import List

from numpy.lib.function_base import append


class Sequential:
    def __init__(self, layer_dimensions: List[int]):
        self.n_layers = len(layer_dimensions)
        self.n_matrices = self.n_layers - 1
        np.random.seed(seed=0)
        self.matrix_dimensions = [(layer_dimensions[i + 1], layer_dimensions[i] + 1) for i in range(self.n_matrices)]
        self.weight_matrices = [
            np.random.uniform(-1, 1, self.matrix_dimensions[i])
            for i in range(self.n_matrices)
                                            ]


    def propagate_forward(self, input: np.ndarray) -> np.ndarray:
        self.layers = [np.append(1, input)]
        for i_layer in range(1,self.n_layers ):
            z = np.matmul(self.weight_matrices[i_layer-1], self.layers[i_layer-1])
            a = np.array(list(map(sigmoid, z)))
            if i_layer != self.n_layers - 1:
                self.layers.append(np.append(1, a)) 
            else:
                 self.layers.append(a)                
        return self.layers[-1]

    def propagate_backward(self,y_pred: np.ndarray,y_train):
        diff = y_pred - y_train 
        for i_matrix in reversed(range(self.n_matrices)):
            pass
            

    def train(self, input: np.ndarray) -> np.ndarray:
        self.propagate_forward(input)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))
