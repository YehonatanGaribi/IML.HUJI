from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import misclassification_error
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        means = np.ndarray(shape=(self.classes_.shape[0], X.shape[1]))
        cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for ind, k in enumerate(self.classes_):
            x_count = X[y==k]
            means[ind] = np.mean(x_count)
            x_dist_mu = x_count - means[ind]
            cov_matrix += x_dist_mu.T @ x_dist_mu
        self.mu_ = means
        self.cov_ = cov_matrix / (samples_dim-1)
        self._cov_inv = inv(self.cov_)


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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        N_k, features_num = self.classes_.shape[0], X.shape[1]
        x_count = np.repeat(X[:, :, np.newaxis], N_k, axis=2)
        a = np.einsum('ijk,jk->ik', x_count, self._cov_inv @ self.mu_.T)
        b = np.einsum('ij,ji->i', self.mu_ @ self._cov_inv, self.mu_.T)
        return np.log(self.pi_) + a - (0.5 * b)

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
