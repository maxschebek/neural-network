# Neural network test with reference data from
# https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw
from network import Sequential, costfunc
import numpy as np


def test_propagate():
    params = np.arange(1, 19) / 10
    weights = [ np.reshape(params[0:6], (2, 3), "F"),np.reshape(params[6:18], (4, 3), "F") ]
    model = Sequential([2, 2, 4],weights)
    X = np.cos([[1, 2]])
    y = np.array([[0.0, 0.0, 0.0, 1]])
    y_pred = model.propagate_forward(X)
    model.propagate_backward((y_pred - y)[0])

    # Test forward propagation
    assert np.allclose(model.layers[1], np.array([1.0, 0.51350, 0.54151]), atol=1e-5)
    assert np.allclose(
        model.layers[2], np.array([0.88866, 0.90743, 0.92330, 0.93665]), atol=1e-5
    )
    assert np.allclose(model.layers_b[0], np.zeros(2), atol=1e-5)
    assert np.allclose(model.layers_b[1], np.array([0.24982, 0.24828]), atol=1e-5)
    assert np.allclose(model.layers_b[2], np.zeros(4), atol=1e-5)

    # Test backward propagation
    assert np.allclose(
        model.layer_errors[2],
        np.array([0.888659, 0.907427, 0.923305, -0.063351]),
        atol=1e-5,
    )

    assert np.allclose(model.layer_errors[1], np.array([0.79393, 1.05281]), atol=1e-5)


def test_accumulation():
    params = np.arange(1, 19) / 10
    weights = [ np.reshape(params[0:6], (2, 3), "F"),np.reshape(params[6:18], (4, 3), "F") ]
    model = Sequential([2, 2, 4],weights)
    X = np.cos([[1, 2], [3, 4], [5, 6]])
    y = np.array([[0.0, 0.0, 0.0, 1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    y_pred = np.zeros(np.shape(y))
    for i in range(len(y)):
        y_pred[i] = model.propagate_forward(X[i])
        model.propagate_backward(y_pred[i] - y[i])

    assert np.allclose(
        model.accumulators[1] / len(y),
        np.array(
            [
                [0.88342, 0.45931, 0.47834],
                [0.56876, 0.34462, 0.36892],
                [0.58467, 0.25631, 0.25977],
                [0.59814, 0.31189, 0.32233],
            ]
        ),
        atol=1e-5,
    )

    assert np.allclose(
        model.accumulators[0] / len(y),
        np.array([[0.766138, -0.027540, -0.024929], [0.979897, -0.035844, -0.053862]]),
        atol=1e-5,
    )

    assert np.allclose(costfunc(y, y_pred,weights,4), 19.474, atol=1e-3)
