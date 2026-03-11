"""
AIstats_lab.py

Student starter file for the Regularization & Overfitting lab.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# =========================
# Helper Functions
# =========================

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# =========================
# Q1 Lasso Regression
# =========================

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent.
    """

    # TODO: Load diabetes dataset
    # TODO: Train/test split
    # TODO: Standardize features
    # TODO: Add bias column
    # TODO: Initialize theta
    # TODO: Implement gradient descent with L1 regularization
    # TODO: Compute predictions
    # TODO: Compute metrics

    raise NotImplementedError


# =========================
# Q2 Polynomial Overfitting
# =========================

def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression.
    """

    # TODO: Load dataset
    # TODO: Select BMI feature only
    # TODO: Train/test split

    degrees = []
    train_errors = []
    test_errors = []

    # TODO: Loop through polynomial degrees
    # TODO: Create polynomial features
    # TODO: Fit regression using normal equation
    # TODO: Compute train/test errors

    raise NotImplementedError
