import matplotlib
import numpy as np
#import os.path
import scipy.io
from scipy.spatial.distance import pdist, squareform
from sklearn import linear_model
import time


# DATA
def load_CPU():
    mat = scipy.io.loadmat('cpu.mat')
    X = np.transpose(mat['Xtrain']).todense()               # 6554 x 21
    X_test = np.transpose(mat['Xtest']).todense()           # 819  x 21
    y = np.transpose(mat['ytrain']).astype('float')         # 6554 x 1
    y_test = np.transpose(mat['ytest']).astype('float')     # 819  x 1

    X = np.array(X)
    X_test = np.array(X_test)

    y = y.flatten()
    y_test = y_test.flatten()

    return X, y, X_test, y_test


def construct_data_synthetic_Laplacian(D, lifetime, noise_var, N_train, N_test):
    # pick datapoint locations uniformly at random
    N = N_train + N_test
    X = np.random.rand(N, D)

    # construct kernel matrix
    K = scipy.exp(- lifetime * squareform(pdist(X, 'cityblock')))

    # sample the function at picked locations x
    y = np.linalg.cholesky(K).dot(np.random.randn(N)) + np.sqrt(noise_var) * np.random.randn(N)

    # pick training indices sequentially
    indices_train = range(0, N_train)
    indices_test  = range(N_train, N)

    # split the data into train and test
    X_train = X[indices_train]
    X_test  = X[indices_test ]
    y_train = y[indices_train]
    y_test  = y[indices_test ]

    return X_train, y_train, X_test, y_test


# SAMPLING
def sample_discrete(weights):
    cumsums = np.cumsum(weights)
    cut = cumsums[-1] * np.random.rand()
    return np.searchsorted(cumsums, cut)


def sample_cut(lX, uX, birth_time):
    rate = np.sum(uX - lX)
    if rate > 0:
        E = np.random.exponential(scale=1.0/rate)
        cut_time = birth_time + E
        dim = sample_discrete(uX - lX)
        loc = lX[dim] + (uX[dim] - lX[dim]) * np.random.rand()
        return cut_time, dim, loc
    else:
        return np.Infinity, None, None


# FOURIER FEATURES
def Fourier_features(X, lifetime, R):
    D = X.shape[1]
    omega = lifetime * np.random.standard_cauchy(size=(D, R))
    Z = np.c_[np.cos(X.dot(omega)), np.sin(X.dot(omega))]
    feature_from_repetition = np.tile(np.arange(R), 2)
    return Z, feature_from_repetition


# REGRESSION
def errors_regression(y, y_test, y_hat_train, y_hat_test):
    relative_error_train = 100.0 * np.linalg.norm(y_hat_train - y) / np.linalg.norm(y)
    relative_error_test = 100.0 * np.linalg.norm(y_hat_test - y_test) / np.linalg.norm(y_test)
    return relative_error_train, relative_error_test


def exact_regression_test_error(X, y, X_test, y_test, delta):
    # center training targets
    y_mean = np.mean(y)
    y_train = y - y_mean

    w_MAP = np.linalg.solve(np.transpose(X).dot(X) + delta * np.identity(np.shape(X)[1]),
                            np.transpose(X).dot(y_train))
    y_hat_test = y_mean + X_test.dot(w_MAP)
    return 100.0 * np.linalg.norm(y_hat_test - y_test) / np.linalg.norm(y_test)


def SGD_regression_test_error(X, y, X_test, y_test, delta, SGD_epochs):
    # center training targets
    y_mean = np.mean(y)
    y_train = y - y_mean

    # solve primal problem
    clf = linear_model.SGDRegressor(alpha=delta, fit_intercept=False, n_iter=SGD_epochs)
    clf.fit(X, y_train)
    y_hat_test = y_mean + X_test.dot(clf.coef_)
    return 100.0 * np.linalg.norm(y_hat_test - y_test) / np.linalg.norm(y_test)


# BINARY SEARCH KERNEL WIDTH
def select_width(X, y, X_test, y_test, M, delta, eval_function, num_iters):
    time_start = time.clock()

    # compute initial performance
    lg_lifetime = 0.0
    error_test = eval_function(X, y, X_test, y_test, M, np.power(2, lg_lifetime), delta)
    list_runtime = [time.clock() - time_start]
    list_error_test = [error_test]
    ll_errors = [(lg_lifetime, error_test)]

    for i in range(num_iters):
        am = np.argmin(zip(*ll_errors)[1])
        if am == 0 or (ll_errors[am][1] > 20.0 and i % 2 == 0):
            lg_lifetime = ll_errors[0][0] - 1
        elif am + 1 == len(ll_errors) or (ll_errors[am][1] > 20.0 and i % 2 == 1):
            lg_lifetime = ll_errors[-1][0] + 1
        elif ll_errors[am + 1][1] < ll_errors[am - 1][1]:
            lg_lifetime = 0.5 * (ll_errors[am + 1][0] + ll_errors[am][0])
        else:
            lg_lifetime = 0.5 * (ll_errors[am - 1][0] + ll_errors[am][0])

        error_test = eval_function(X, y, X_test, y_test, M, np.power(2, lg_lifetime), delta)
        list_runtime.append(time.clock() - time_start)
        list_error_test.append(error_test)
        ll_errors.append((lg_lifetime, error_test))
        ll_errors.sort()

    return list_runtime, list_error_test


# PLOTTING UTILITIES
def initialize_plotting():
    font = {'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14}
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)


def remove_chartjunk(ax):
    # remove top and side borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # change bottom border to dotted line
    ax.spines["bottom"].set_linestyle('dotted')
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color('black')
    ax.spines["bottom"].set_alpha(0.3)

    # remove ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")


def tableau20(k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20[k]
