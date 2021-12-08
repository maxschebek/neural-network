# Neural network test with reference data from
# https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw
from numpy.core.defchararray import mod
from network import Sequential, costfunc, regularization_term
import numpy as np


def test_propagate():
    params = np.arange(1, 19) / 10
    weights = [
        np.reshape(params[0:6], (2, 3), "F"),
        np.reshape(params[6:18], (4, 3), "F"),
    ]
    model = Sequential([2, 2, 4], weights)
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
    weights = [
        np.reshape(params[0:6], (2, 3), "F"),
        np.reshape(params[6:18], (4, 3), "F"),
    ]
    model = Sequential([2, 2, 4], weights)
    alpha = 4
    X = np.cos([[1, 2], [3, 4], [5, 6]])
    y = np.array([[0.0, 0.0, 0.0, 1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    batch_size = len(y)
    y_pred = np.zeros(np.shape(y))
    cost = np.zeros(1)

    for i in range(batch_size):
        y_pred[i] = model.propagate_forward(X[i])
        model.propagate_backward(y_pred[i] - y[i])
        cost += costfunc(y[i], y_pred[i])
    reg = regularization_term(weights, alpha)
    cost += reg
    cost /= batch_size
    
    model.make_gradients(alpha, batch_size)



    assert np.allclose(
        model.accumulators[1] / batch_size,
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
        model.accumulators[0] / batch_size,
        np.array([[0.766138, -0.027540, -0.024929], [0.979897, -0.035844, -0.053862]]),
        atol=1e-5,
    )

    assert np.allclose(cost, 19.474, atol=1e-3)

    assert np.allclose(
        model.gradients[0],
        np.array([[0.76614, 0.37246, 0.64174], [0.9799, 0.49749, 0.74614]]),
        atol=1e-3,
    )

    assert np.allclose(
        model.gradients[1],
        np.array(
            [
                [0.88342, 1.92598, 2.47834],
                [0.56876, 1.94462, 2.50225],
                [0.58467, 1.98965, 2.52644],
                [0.59814, 2.17855, 2.72233],
            ]
        ),
        atol=1e-3,
    )
