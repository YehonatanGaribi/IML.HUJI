from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import loss_functions
import numpy as np
from math import atan2, pi
from utils import *
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[: ,:2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        X, y = load_dataset(f"..\\datasets\\{f}")
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(model, u, v):
            losses.append(model.loss(X, y))
        model = Perceptron(callback=callback)
        model.fit(X, y)



        # Plot figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=np.arange(len(losses)), y=losses, name="Loss Value",
                       line=dict(color='blue', width=2))
        )
        fig.update_layout(title=f"Loss as function of the number of updates "
                                f"- {n}.",
                          xaxis_title="Updates Number",
                          yaxis_title="Misclassification Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker=dict(
                      color="black"))


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load(f"..\\datasets\\{f}")
        X = data[:, :-1]
        y = data[:, -1].astype(int)

        # Fit models and predict over training set
        lda, gaussian = LDA().fit(X,y), GaussianNaiveBayes().fit(X,y)
        lda_predict, gaussian_predict = lda.predict(X), gaussian.predict(X)

        symbols = np.array(["circle", "star", "square"])
        lda_accuracy = loss_functions.accuracy(y, lda_predict)
        gnb_accuracy = loss_functions.accuracy(y, gaussian_predict)
        titles = [("LDA", lda_accuracy),
                  ("Gaussian Naive Bayes", gnb_accuracy)]
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "%s Model. Accuracy: %.3f. " % t for t in titles],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        fig.update_layout(
            title=f"Comparison Between LDA and GNB. Dataset: {f}",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        for i, pred in enumerate([lda_predict, gaussian_predict]):
            fig.add_traces([
                go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                           showlegend=False,
                           marker=dict(color=pred, symbol=symbols[y],
                                       colorscale=[custom[0], custom[-1]],
                                       line=dict(color="black", width=1)))],
                rows=1, cols=i + 1
            )
        for i, m in enumerate([lda, gaussian]):
            fig.add_traces([
                go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers",
                           showlegend=False,
                           marker=dict(color="black", symbol="x",
                                       line=dict(color="black", width=1)))],
                rows=1, cols=i + 1
            )
        fig.add_traces([get_ellipse(lda.mu_[k, :], lda.cov_) for k in
                        range(lda.mu_.shape[0])], rows=1, cols=1)
        fig.add_traces(
            [get_ellipse(gaussian.mu_[k, :], np.diag(gaussian.vars_[k, :])) for k in
             range(gaussian.mu_.shape[0])], rows=1, cols=2)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

    
