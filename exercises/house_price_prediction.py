import math
import trace

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.learners.regressors import PolynomialFitting

from typing import NoReturn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = pd.get_dummies(df, columns=['zipcode'])
    df['house_age'] = df[['yr_built', 'yr_renovated']].max(axis=1)
    df.drop(['yr_built', 'yr_renovated', 'long', 'date', 'lat', 'sqft_lot15',
             'sqft_lot', 'condition', 'id', 'bedrooms'], axis=1,
            inplace=True)
    df.dropna(inplace=True)
    y = df['price']
    X = df.drop('price', axis=1)
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        if 'zipcode' in feature:
            continue
        X_vals = X[feature].values
        covariance_matrix = np.cov(X_vals, y)
        sigma_feature = math.sqrt(covariance_matrix[0][0])
        sigma_y = math.sqrt(covariance_matrix[1][1])
        covariance = covariance_matrix[0][1]
        correlation = covariance / (sigma_y * sigma_feature)
        fig = go.Figure(
            [go.Scatter(x=y, y=X_vals, mode="markers",
                        marker=dict(color="blue", opacity=.3))],
            layout=go.Layout(title=f"Correlation between "
                                   f"house price and {feature} is: "
                                   f"{correlation.__round__(4)}.",
                             xaxis={"title": "House price"},
                             yaxis={"title": f"{feature}"},
                             height=400))
        fig.write_image(f"{output_path}/{feature}.jpg")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    path = "..\\datasets\\house_prices.csv"
    X, y_true = load_data(path)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y_true)

    # Question 3 - Split samples into training- and testing sets.

    X_train, y_train, X_test, y_test = split_train_test(X, y_true)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_loss, std_loss = np.zeros(91), np.zeros(91)
    index = 0
    for p in range(10, 101):
        loss = []
        for i in range(10):
            indexes = X_train.sample(frac=p / 100).index
            curX_sample = X_train.filter(items=indexes, axis=0)
            cury_sample = y_train.filter(items=indexes, axis=0)
            cur_model = LinearRegression().fit(curX_sample.to_numpy(),
                                               cury_sample.to_numpy())
            loss.append(cur_model._loss(X_test.to_numpy(), y_test.to_numpy()))
        mean_loss[index] = np.mean(loss)
        std_loss[index] = np.std(loss)
        index += 1
    plt.plot(range(10, 101, 1), mean_loss)
    plt.fill_between(range(10, 101, 1), (mean_loss - 2 * std_loss),
                     (mean_loss + 2 * std_loss), color='b', alpha=.1)
    plt.title("Mean loss as function of p%")
    plt.xlabel("p%")
    plt.ylabel("Mean loss")
    plt.show()
