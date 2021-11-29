from numpy.core.fromnumeric import shape
from network import Sequential, sigmoid
import numpy as np


def test_propagate_forward():
    model = Sequential([2, 5, 1])
    input = np.array([0.7, 0.5])

    np.random.seed(seed=0)
    weights = [np.random.uniform(-1, 1, (5, 3))]
    weights.append(np.random.uniform(-1, 1, (1, 6)))
    layers = [np.append(np.ones(1), input)]
    z1 = np.matmul(weights[0], layers[0])
    a1 = np.array(list(map(sigmoid, z1)))
    layers.append(np.append(1, a1))
    z2 = np.matmul(weights[1], layers[1])
    a2 = np.array(list(map(sigmoid, z2)))
    layers.append(np.append(1, a2))

    model.propagate_forward(input)
    assert model.n_layers == 3
    assert model.n_weights == 2
    assert np.array_equal(model.layers[0], layers[0])
    assert np.array_equal(model.layers[1], layers[1])
    assert np.array_equal(model.layers[2], layers[2])
    assert np.array_equal(model.weights[0], weights[0])
    assert np.array_equal(model.weights[1], weights[1])


def test_propagate_backward():
    model = Sequential([2, 5, 1])
    input = np.array([0.7, 0.5])
    y_pred = model.propagate_forward(input)
    y_train = np.array([0.4])
    layer_errors = 3 * [None]
    weight_errors = 2 * [None]

    layer_errors[2] = y_pred - y_train
    weight_errors[1] = np.outer(layer_errors[2], model.layers[1])
    aux_vector = np.matmul(model.weights[1][:, 1:].T, layer_errors[2])
    layer_errors[1] = np.multiply(aux_vector, model.layers_b[1])
    weight_errors[0] = np.outer(layer_errors[1], model.layers[0])

    model.propagate_backward(y_pred - y_train)
    assert np.shape(model.weight_errors[0]) == np.shape(model.weights[0])
    assert np.shape(model.weight_errors[1]) == np.shape(model.weights[1])
    assert np.array_equal(layer_errors[2], model.layer_errors[2])
    assert np.array_equal(layer_errors[1], model.layer_errors[1])
    assert np.array_equal(weight_errors[1], model.weight_errors[1])
    assert np.array_equal(weight_errors[0], model.weight_errors[0])
