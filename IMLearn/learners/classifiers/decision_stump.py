from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.sign_ = 1
        features_num = X.shape[1]
        pos_errors, neg_errors = np.empty(features_num), np.empty(
            features_num)
        pos_thresholds, neg_thresholds = np.empty(features_num), np.empty(
            features_num)

        for i in range(features_num):
            column = X[:, i]
            pos_thresholds[i], pos_errors[i] = self._find_threshold(
                column, y,
                self.sign_)
            neg_thresholds[i], neg_errors[i] = self._find_threshold(
                column, y,
                -self.sign_)

        min_pos_index, min_neg_index = np.argmin(pos_errors), \
                                       np.argmin(neg_errors)
        self.threshold_ = pos_thresholds[min_pos_index]
        self.j_ = min_pos_index

        if neg_errors[min_neg_index] < pos_errors[min_pos_index]:
            self.threshold_ = neg_thresholds[min_neg_index]
            self.j_ = min_neg_index
            self.sign_ = -1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        column = X[:, self.j_]
        return np.where(column < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        y = np.sign(labels)
        abs_labels = np.abs(labels)
        sort_indexes = np.argsort(values)
        X, y, abs_labels = values[sort_indexes], y[sort_indexes], abs_labels[
            sort_indexes]

        thresholds = np.concatenate(
            [[-np.inf], (X[1:] + X[: -1]) / 2, [np.inf]])
        min_error = np.sum(abs_labels[y == sign])
        errors = np.append(min_error,
                           min_error - np.cumsum(abs_labels * (y * sign)))
        min_error_idnex = np.argmin(errors)
        return thresholds[min_error_idnex], errors[min_error_idnex]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        error = np.where(y_pred != np.sign(y), 1, 0)
        return np.sum(error) / error.shape[0]
