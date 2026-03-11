import numpy as np
import AI_stats_lab as A


def test_lasso_regression():

    train_mse, test_mse, train_r2, test_r2, theta = A.lasso_regression_diabetes()

    assert train_mse > 0
    assert test_mse > 0

    assert train_r2 > 0
    assert test_r2 > 0

    assert isinstance(theta, np.ndarray)
    assert theta.ndim == 1


def test_polynomial_overfitting():

    result = A.polynomial_overfitting_experiment()

    degrees = result["degrees"]
    train_mse = result["train_mse"]
    test_mse = result["test_mse"]

    assert len(degrees) == len(train_mse)
    assert len(degrees) == len(test_mse)

    # training error should decrease
    assert train_mse[-1] <= train_mse[0]

    # test error should increase at some point
    assert max(test_mse) >= test_mse[0]
