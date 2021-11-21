from network import Sequential, sigmoid
import numpy as np


def test_propagate_forward():
    model = Sequential([2, 5, 1])
    input = np.array([0.7, 0.5])
    prediction_model = model.propagate_forward(input)

    np.random.seed(seed=0)
    weights_a = np.random.uniform(-1, 1, (5, 3))
    weights_b = np.random.uniform(-1, 1, (1, 6))
    layer0 = np.append(1, input)
    z1 = np.matmul(weights_a, layer0)
    a1 = np.array(list(map(sigmoid, z1)))
    layer1 = np.append(1, a1)
    z2 = np.matmul(weights_b, layer1)
    prediction = np.array(list(map(sigmoid, z2)))

    assert all(model.layers[1] == layer1)
    assert all(model.layers[2] == prediction)
    assert prediction_model == prediction

