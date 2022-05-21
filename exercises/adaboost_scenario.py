import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_erros = np.empty(n_learners)
    test_errors = np.empty(n_learners)

    for t in range(n_learners):
        train_erros[t] = \
            np.array(adaboost.partial_loss(train_X, train_y, t + 1))
        test_errors[t] = np.array(adaboost.partial_loss(test_X, test_y, t + 1))

    x = list(range(1, n_learners + 1))
    go.Figure([
        go.Scatter(x=x, y=test_errors, mode='lines', name=r'$test loss$'),
        go.Scatter(x=x, y=train_erros, mode='lines', name=r'$train loss$')]) \
        .update_layout(
        title=f"Training and Test errors as a function of the "
              f"number of fitted learners. Noise ratio is: {noise}").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    symbols = np.array(["circle", "x"])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"Decision stump, {t} iterations." for
                                        t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda X: adaboost.partial_predict(X, t),
                              lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y,
                                    symbol=symbols[test_y.astype(int)],
                                    colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black",
                                              width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries Of Decision Stump Models Dataset}}$"
        , margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    min_error_index = int(np.argmin(test_errors)) + 1
    fig1 = go.Figure().add_traces(
        [decision_surface(lambda X: adaboost.partial_predict(X,
                                                             min_error_index),
                          lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=test_y,
                                symbol=symbols[test_y.astype(int)],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black",
                                          width=1)))])

    fig1.update_layout(
        title=f"Best size of ensemble is {min_error_index} and accuracy is "
              f"{accuracy(test_y, adaboost.partial_predict(test_X, min_error_index))}"
        , margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)
    fig1.show()

    # Question 4: Decision surface with weighted samples

    normalized_D = adaboost.D_ / np.max(adaboost.D_) * 5
    fig2 = go.Figure().add_traces(
        [decision_surface(adaboost.predict,
                          lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(size=normalized_D, color=train_y,
                                symbol=symbols[train_y.astype(int)],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black",
                                          width=1)))])

    fig2.update_layout(
        title=f"Decision surface of ensemble at full size. Samples size "
              f"represent their weights. Noise factor is {noise}"
        , margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)
    fig2.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
