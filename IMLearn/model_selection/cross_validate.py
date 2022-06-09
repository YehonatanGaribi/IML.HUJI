from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd

from IMLearn import BaseEstimator
from ..utils.utils import split_train_test


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_test_errors = np.zeros(cv)
    validation_errors = np.zeros(cv)
    X_split, y_split = np.array_split(X, cv), np.array_split(y, cv)

    for i in range(cv):
        X_train_smaller = np.concatenate(X_split[:i] + X_split[i+1:])
        y_train_smaller = np.concatenate(y_split[:i] + y_split[i + 1:])
        X_valid, y_valid = X_split[i], y_split[i]
        estimator.fit(X_train_smaller, y_train_smaller)

        # Testing over train data.
        y_test_pred = estimator.predict(X_train_smaller)
        train_test_errors[i] = scoring(y_train_smaller, y_test_pred)

        # Testing over validation data.
        y_valid_pred = estimator.predict(X_valid)
        validation_errors[i] = scoring(y_valid, y_valid_pred)

    train_score = sum(train_test_errors) / cv
    validation_score = sum(validation_errors) / cv
    return train_score, validation_score


def split_train_validation(X, y):
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y))
    train_X = train_X.to_numpy().reshape((train_X.shape[0],))
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy().reshape((test_X.shape[0],))
    test_y = test_y.to_numpy()
    return train_X, train_y, test_X, test_y
