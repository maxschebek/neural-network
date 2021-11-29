import numpy as np
from typing import List
from numpy.core.numeric import ones

from numpy.lib.function_base import append


class Sequential:
    def __init__(self, layer_dims: List[int]):
        self.n_layers = len(layer_dims)
        self.n_weights = self.n_layers - 1
        np.random.seed(seed=0)
        self.weight_dims = [
            (layer_dims[i + 1], layer_dims[i] + 1)
            for i in range(self.n_weights)
        ]
        self.weights = [
            np.random.uniform(-1, 1, self.weight_dims[i])
            for i in range(self.n_weights)
        ]
        self.weight_errors = [
            np.zeros(self.weight_dims[i])
            for i in range(self.n_weights)
        ]
        
        
        self.layers = [
            np.ones(layer_dims[i_lay] + 1)
            for i_lay in range(self.n_layers)
        ]

        self.layers_b = [
            np.zeros(layer_dims[i_lay])
            for i_lay in range(self.n_layers)
        ]
        self.layer_errors = [
            np.zeros(layer_dims[i_lay])
            for i_lay in range(self.n_layers)
        ]

    def propagate_forward(self, input: np.ndarray) -> np.ndarray:
        self.layers[0][1:] = input
        for i_lay in range(1, self.n_layers):
            z = np.matmul(self.weights[i_lay - 1], self.layers[i_lay - 1])
            a = np.array(list(map(sigmoid, z)))
            b = np.array(list(map(sigmoid_derivative, z)))
            self.layers[i_lay][1:] = a  
            self.layers_b[i_lay] = b
        return self.layers[-1][1:]

    def propagate_backward(self, error: np.ndarray):

        self.layer_errors[-1] = error
        self.weight_errors[-1] = np.outer(self.layer_errors[-1], self.layers[-2])
        for i_lay in reversed(range(1, self.n_layers - 1)):
            aux_vector = np.matmul(
                self.weights[i_lay][:,1:].T, self.layer_errors[i_lay + 1]
            )
            self.layer_errors[i_lay] = np.multiply(aux_vector, self.layers_b[i_lay])
            self.weight_errors[i_lay - 1 ] = np.outer(self.layer_errors[i_lay], self.layers[i_lay - 1])
            

       
   

    def train(self, input: np.ndarray) -> np.ndarray:
        self.propagate_forward(input)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))
