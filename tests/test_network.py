from network import Sequential, sigmoid
import numpy as np


def test_propagate_forward():
    model = Sequential([2, 5, 1])
    input = np.array([0.7, 0.5])
    prediction_model = model.propagate_forward(input)

    np.random.seed(seed=0)
    weights_a = np.random.uniform(-1, 1, (5, 3))
    weights_b = np.random.uniform(-1, 1, (1, 6))
    a_with_bias1 = np.append(1, input)
    z2 = np.matmul(weights_a, a_with_bias1)
    a2 = np.array(list(map(sigmoid, z2)))
    a_with_bias2 = np.append(1, a2)
    z3 = np.matmul(weights_b, a_with_bias2)
    prediction = np.array(list(map(sigmoid, z3)))

    assert prediction_model == prediction
