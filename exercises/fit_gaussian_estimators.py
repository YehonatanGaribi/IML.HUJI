from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = 1000
    X = np.random.normal(mu, sigma, samples)
    uni_gauss = UnivariateGaussian().fit(X)
    print('(', uni_gauss.mu_, ', ', uni_gauss.var_, ')')

    # Question 2 - Empirically showing sample mean is consistent
    samples_sizes = []
    plots = []
    for i in range(1, 101):
        model = UnivariateGaussian().fit(X[:1 + 10 * i])
        samples_sizes.append(10 * i)
        plots.append(abs(model.mu_ - mu))
    go.Figure([go.Scatter(x=samples_sizes, y=plots, mode='markers+lines')],
              layout=go.Layout(
                  title=r"$\text{Estimation of Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$\\text{Number of samples}$",
                  yaxis_title="r$\\text{Absolute distance}$",
                  height=300, width=1400)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uni_gauss.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf, mode="markers")],
              layout=go.Layout(
                  title=r"$\text{Empirical PDF Under Fitted Model}$",
                  xaxis_title="$\\text{Samples values}$",
                  yaxis_title="$\\text{PDF values}$ $\mu$",
                  width=1400)).show()

    log1 = uni_gauss.log_likelihood(1, 1, np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2]))
    log2 = uni_gauss.log_likelihood(10, 1, np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2]))
    print(log1)
    print(log2)

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    Sigma = np.array([[1, 0.2, 0, 0.5],
                        [0.2, 2, 0, 0],
                        [0, 0, 1, 0],
                        [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, Sigma, 1000)
    multi_gauss = MultivariateGaussian().fit(X)
    print(multi_gauss.mu_)
    print(multi_gauss.cov_)

    # Question 5 - Likelihood evaluation
    f1, f3 = np.linspace(-10, 10, 200), np.linspace(-10, 10, 200)
    log_likelihood = np.zeros((200, 200))
    for cur_f1 in range(200):
        for cur_f3 in range(200):
            cur_mu = np.array([f1[cur_f1], 0, f3[cur_f3], 0])
            log_likelihood[cur_f1, cur_f3] = \
                MultivariateGaussian.log_likelihood(cur_mu, Sigma, X)
    go.Figure(go.Heatmap(x=f1, y=f3, z=log_likelihood), layout=go.Layout(
        title=r"$\text{Log-likelihood of } \hat\mu "
              r"\text{=[f1, 0, f3, 0], according to the real } "
              r"\mu \text{=[0, 0, 4, 0].}$",
        xaxis_title="$\\text{f1's values}$",
        yaxis_title="$\\text{f3's values}$",
        height=500, width=1000)).show()

    # Question 6 - Maximum likelihood
    argmax = np.argmax(log_likelihood)
    row = (argmax // 200)
    col = (argmax % 200)
    print(f1[row], ',', f3[col])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
