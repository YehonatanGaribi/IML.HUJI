from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import misclassification_error
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier
        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`
        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`
        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`
        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        samples_dim = y.shape[0]
        self.classes_, N_k = np.unique(y, return_counts=True)
        self.pi_ = N_k / samples_dim
        classes_num = N_k.shape[0]
        features_dim = 1
        if len(X.shape) == 2:
            features_dim = X.shape[1]
        means = np.ndarray(shape=(classes_num, features_dim))
        vars = np.ndarray(shape=(classes_num, features_dim))
        for i, k in enumerate(self.classes_):
            x_count = X[y == k]
            means[i] = np.mean(x_count)
            vars[i] = np.var(x_count, ddof=1)
        self.mu_, self.vars_ = means, vars

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        maximized = np.argmax(self.likelihood(X), axis=1)
        return self.classes_[maximized]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.
        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        classes_num = self.classes_.shape[0]
        samples_dim = X.shape[0]
        features_dim = X.shape[1]
        x_dist_mu = np.repeat(X, classes_num).reshape(
            (samples_dim, classes_num, features_dim)) - self.mu_
        coeff = 1 / np.sqrt(2 * np.pi * self.vars_)
        features_PDF = np.log(coeff) + (
                    (x_dist_mu ** 2) / (-2 * self.vars_))
        return np.sum(features_PDF, axis=2)

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
        return misclassification_error(y, self._predict(X))