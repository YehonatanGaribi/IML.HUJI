from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    func = lambda X: (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    X = np.linspace(-1.2, 2, n_samples)  # X values.
    true_f = func(X)  # Noiseless responses.
    epsilon = np.random.normal(size=len(true_f), scale=noise)
    y = true_f + epsilon  # Noisy response.

    # Splitting the data to train-test, and converting back to numpy.
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y),
                                                        train_proportion=(
                                                                2 / 3))
    train_X = train_X.to_numpy().reshape((train_X.shape[0],))
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy().reshape((test_X.shape[0],))
    test_y = test_y.to_numpy()

    # Creating the graph.

    plt.plot(X, true_f, label="f(x)", color="darkgreen", linewidth='3.0')
    plt.scatter(train_X, train_y, label="Train data")
    plt.scatter(test_X, test_y, label="Test data")
    plt.title(f"Noiseless function, test and train data with noise level of "
              f"{noise}.")
    plt.xlabel("X values (uniformly ranged in [-1.2, 2]")
    plt.ylabel("Noisy response values")
    plt.legend()
    plt.savefig(f"Q1,4,5 - Noise {noise}.png")
    plt.close()
    # plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    train_scores = np.zeros(11)
    validation_scores = np.zeros(11)

    for k in range(11):
        model = PolynomialFitting(k)
        train_scores[k], validation_scores[k] = cross_validate(model, train_X,
                                                               train_y,
                                                               mean_square_error)
    # Creating the graph.
    x = np.linspace(0, 10, 11)

    plt.plot(x, validation_scores, label="Validation error")
    plt.plot(x, train_scores, label="Train error")
    plt.title(f"5-Fold Cross-Validation for polynomial degrees k = 0,1,...,"
              f"10.\nNoise level: {noise}")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Error Score")
    plt.legend()
    plt.savefig(f"Q2,4,5 - Noise {noise}.png")
    plt.close()
    # plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_scores).astype(int)
    best_model = PolynomialFitting(best_k).fit(train_X, train_y)
    test_error = mean_square_error(test_y, best_model.predict(
        test_X)).__round__(2)
    print(f"Best polynomial degree is: {best_k},"
          f" and the achieved test error is: {test_error}.")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X = X.sample(n=n_samples)
    test_X = X.loc[X.index.difference(train_X.index)]
    test_y, train_y = y.loc[test_X.index], y.loc[train_X.index]
    test_X, test_y, train_X, train_y = test_X.to_numpy(), test_y.to_numpy(), \
                                       train_X.to_numpy(), train_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lasso_lambdas = np.linspace(0.02, 5, n_evaluations)
    ridge_lambdas = np.linspace(0.01, 50, n_evaluations)
    lasso_train_score, lasso_validation_score = np.zeros(n_evaluations), \
                                                np.zeros(n_evaluations)
    ridge_train_score, ridge_validation_score = np.zeros(n_evaluations), \
                                                np.zeros(n_evaluations)
    lambdas = [lasso_lambdas, ridge_lambdas]
    train_scores = [lasso_train_score, ridge_train_score]
    validation_scores = [lasso_validation_score, ridge_validation_score]
    models = ["Lasso", "Ridge"]

    for ind, params in enumerate(zip(lasso_lambdas, ridge_lambdas)):
        lasso_model = Lasso(alpha=params[0])
        ridge_model = RidgeRegression(lam=params[1])
        lasso_train_score[ind], lasso_validation_score[ind] = \
            cross_validate(lasso_model, train_X, train_y, mean_square_error)
        ridge_train_score[ind], ridge_validation_score[ind] = \
            cross_validate(ridge_model, train_X, train_y, mean_square_error)

    #     Create graphs.
    for ind, model in enumerate(models):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lambdas[ind],
                                 y=train_scores[ind],
                                 mode='lines',
                                 name=f'Train Error',
                                 marker=dict(color="darkred")))
        fig.add_trace(go.Scatter(x=lambdas[ind],
                                 y=validation_scores[ind],
                                 mode='lines',
                                 name=f'Validation Error',
                                 marker=dict(color="darkgreen")))
        fig.update_layout(
            title=f"{model} regression CV train and validation "
                  f"errors as function of parametrization term",
            xaxis_title=r'$\lambda$ \text{ Values}',
            yaxis_title='Error Score')
        # fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lasso_ind = np.argmin(lasso_validation_score)
    best_lasso_param = lasso_lambdas[best_lasso_ind]
    best_lasso_model = Lasso(alpha=best_lasso_param)
    best_lasso_model.fit(train_X, train_y)
    lasso_pred = best_lasso_model.predict(test_X)
    best_lasso_err = mean_square_error(test_y, lasso_pred)
    print(f"Best Lasso Param: {best_lasso_param}\n"
          f"Best Lasso Test Error: {best_lasso_err}")

    best_ridge_ind = np.argmin(ridge_validation_score)
    best_ridge_param = ridge_lambdas[best_ridge_ind]
    best_ridge_model = RidgeRegression(lam=best_ridge_param)
    best_ridge_model.fit(train_X, train_y)
    ridge_pred = best_ridge_model.predict(test_X)
    best_ridge_err = mean_square_error(test_y, ridge_pred)
    print(f"Best Ridge Param: {best_ridge_param}\n"
          f"Best Ridge Test Error: {best_ridge_err}")

    linear_model = LinearRegression().fit(train_X, train_y)
    linear_err = linear_model.loss(test_X, test_y)
    print(f"Linear Regression Test Error: {linear_err}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
