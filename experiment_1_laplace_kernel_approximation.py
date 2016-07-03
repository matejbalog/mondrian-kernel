import matplotlib.pylab as plt
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
import sys

from mondrian_kernel import Mondrian_kernel_features
from random_binning import random_binning_features
from utils import load_CPU, Fourier_features, initialize_plotting, remove_chartjunk, exact_regression_test_error, tableau20


def run_and_plot_experiment_convergence(X, lifetime, M_max, num_sweeps):
    """ Compares kernel approximation error of different random feature schemes (Fourier features,
    	random binning, Mondrian features) for the Laplace kernel with the given lifetime.
    """

    # precompute NxN kernel (Gram) matrix
    K = scipy.exp(- lifetime * squareform(pdist(X, 'cityblock')))

    # compute maximum kernel approximation error as number of features increases
    def get_errors(get_features, R_max, scheme_name):
        errors = [[] for _ in range(R_max)]
        for repeat in range(num_sweeps):
            # obtain fresh features
            Z, feature_from_repetition = get_features(X, lifetime, R_max)
            for R in range(1, R_max+1):
                # save maximum kernel approximation error
                fs = feature_from_repetition < R
                E = abs(Z[:, fs].dot(np.transpose(Z[:, fs])) / R - K)
                errors[R-1].append(np.max(E))
            sys.stdout.write("\r%s %d / %d" % (scheme_name, repeat+1, num_sweeps))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return map(np.mean, errors), map(np.std, errors)

    RB_error_avg, RB_error_std = get_errors(random_binning_features, M_max, 'random binning')
    MK_error_avg, MK_error_std = get_errors(Mondrian_kernel_features, M_max, 'Mondrian kernel')
    FF_error_avg, FF_error_std = get_errors(Fourier_features, M_max/2, 'Fourier features')

    # plot
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)
    ax.yaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('$M$ (non-zero features per data point)')
    ax.set_ylabel('maximum absolute error')
    ax.set_ylim((0, 1.2))
    ax.errorbar(range(1, M_max+1, 2), FF_error_avg, yerr=FF_error_std, marker='^', linestyle='',
                markeredgecolor=tableau20(0), color=tableau20(0), label='Fourier features')
    ax.errorbar(range(1, M_max+1, 1), RB_error_avg, yerr=RB_error_std, marker='o', linestyle='',
                markeredgecolor=tableau20(6), color=tableau20(6), label='random binning')
    ax.errorbar(range(1, M_max+1, 1), MK_error_avg, yerr=MK_error_std, marker='v', linestyle='',
                markeredgecolor=tableau20(4), color=tableau20(4), label='Mondrian kernel')
    ax.legend(bbox_to_anchor=(1.5, 1.1), frameon=False)


def run_and_plot_experiment_convergence_testerror(X, y, X_test, y_test, M_max, lifetime, delta, num_sweeps, exact_Laplace=False):
    """ Compares approximation error of different random feature schemes (Fourier features,
        random binning, Mondrian features) for the Laplace kernel indirectly via test set error.
    """
    N, D = np.shape(X)
    X_all = np.array(np.r_[X, X_test])

    if exact_Laplace:
        K_all = scipy.exp(- lifetime * squareform(pdist(X_all, 'cityblock')))
        h = np.linalg.solve(K_all[:N, :N] + delta * np.identity(N), y - np.mean(y))
        y_test_hat = np.mean(y) + np.transpose(K_all[:N, N:]).dot(h)
        Laplace_error_test = 100.0 * np.linalg.norm(y_test_hat - y_test) / np.linalg.norm(y_test)
        print Laplace_error_test

    # compute RMSE as the number of features increases
    def get_errors(get_features, R_max, scheme_name):
        errors = [[] for _ in range(R_max)]
        for sweep in range(num_sweeps):
            # obtain fresh features
            Z, feature_from_repetition = get_features(X_all, lifetime, R_max)
            for R in range(1, R_max+1):
                # save maximum kernel approximation error
                fs = feature_from_repetition < R
                Z_train = Z[:N, fs] / np.sqrt(R)
                Z_test = Z[N:, fs] / np.sqrt(R)
                error_test = exact_regression_test_error(Z_train, y, Z_test, y_test, delta)
                errors[R-1].append(error_test)
            sys.stdout.write("\r%s %d / %d" % (scheme_name, sweep + 1, num_sweeps))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return map(np.mean, errors), map(np.std, errors)

    RB_error_avg, RB_error_std = get_errors(random_binning_features, M_max, 'random binning')
    MK_error_avg, MK_error_std = get_errors(Mondrian_kernel_features, M_max, 'Mondrian kernel')
    FF_error_avg, FF_error_std = get_errors(Fourier_features, M_max/2, 'Fourier features')

    # plot error against # M of non-zero features
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)
    ax.yaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('$M$ (non-zero features per data point)')
    ax.set_ylabel('relative test set error [\%]')
    ax.errorbar(range(1, M_max+1, 2), FF_error_avg, yerr=FF_error_std, marker='^',
                markeredgecolor=tableau20(0), ls='', color=tableau20(0), label='Fourier features')
    ax.errorbar(range(1, M_max+1, 1), RB_error_avg, yerr=RB_error_std, marker='o',
                markeredgecolor=tableau20(6), ls='', color=tableau20(6), label='random binning')
    ax.errorbar(range(1, M_max+1, 1), MK_error_avg, yerr=MK_error_std, marker='v',
                markeredgecolor=tableau20(4), ls='', color=tableau20(4), label='Mondrian kernel')
    # add exact Laplace kernel regression test set RMSE (on CPU this is 3.12% with lifetime 1.0)
    ax.axhline(y=3.12, color='black', lw=2)
    ax.legend(bbox_to_anchor=(1.5, 1.1), frameon=False)


def experiment_convergence_kernelerror():
    # obtain data
    D = 2       # input dimension
    N = 100     # number of sampling locations
    X = np.random.rand(N, D)    # sample N datapoints uniformly at random from unit interval/square/cube/hypercube

    # run experiment and plot results
    lifetime = 10.0
    M_max = 50          # maximum value of M to sweep until
    num_sweeps = 5      # sweeps through M (repetitions of the experiment)
    run_and_plot_experiment_convergence(X, lifetime, M_max, num_sweeps)


def experiment_convergence_testerror():
    np.random.seed(0)

    # load CPU data
    X, y, X_test, y_test = load_CPU()

    # run experiment and plot results
    M_max = 50
    lifetime = 1e-6     # value used by Rahimi & Recht
    delta = 0.0001      # value used by Rahimi & Recht
    num_repeats = 5     # number of experiment repetitions to get error bars
    run_and_plot_experiment_convergence_testerror(X, y, X_test, y_test, M_max, lifetime, delta, num_repeats)


def main():
    initialize_plotting()
    experiment_convergence_kernelerror()
    experiment_convergence_testerror()
    plt.show()

if __name__ == "__main__":
    main()
