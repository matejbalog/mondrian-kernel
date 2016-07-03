import matplotlib.pylab as plt
import numpy as np

from mondrian_kernel import evaluate_all_lifetimes
from random_binning import evaluate_random_binning
from utils import construct_data_synthetic_Laplacian, load_CPU, select_width, SGD_regression_test_error, initialize_plotting, remove_chartjunk, tableau20


def evaluate_fourier_features(X, y, X_test, y_test, M, lifetime, delta):
    # get Fourier features
    D = X.shape[1]
    omega = lifetime * np.random.standard_cauchy(size=(D, M))
    Z = np.c_[np.cos(X.dot(omega)), np.sin(X.dot(omega))]
    Z_test = np.c_[np.cos(X_test.dot(omega)), np.sin(X_test.dot(omega))]

    SGD_epochs = 10
    error_test = SGD_regression_test_error(Z, y, Z_test, y_test, delta, SGD_epochs)
    print 'lg_lifetime = %.3f; error_test = %.2f%%' % (np.log2(lifetime), error_test)
    return error_test


def experiment_synthetic():
    """ Simulates data from a Gaussian process prior with a Laplace kernel of known lifetime (inverse width). Then the
        Mondrian kernel procedure for evaluating all lifetimes from 0 up to a terminal lifetime is run on this simulated
        dataset and the results are plotted, showing how accurately the ground truth inverse kernel width could be recovered.
    """

    # synthetize data from Laplace kernel
    D = 2
    lifetime = 10.00
    noise_var = 0.01 ** 2
    N_train = 1000
    N_validation = 1000
    N_test = 1000
    X, y, X_test, y_test = construct_data_synthetic_Laplacian(D, lifetime, noise_var, N_train, N_validation + N_test)

    # Mondrian kernel lifetime sweep parameters
    M = 50
    lifetime_max = lifetime * 2
    delta = noise_var   # prior variance 1

    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta, mondrian_kernel=True, validation=True)
    lifetimes = res['times']
    error_train = res['kernel_train']
    error_validation = res['kernel_validation']
    error_test = res['kernel_test']

    # set up plot
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot('111')
    remove_chartjunk(ax)

    ax.set_title('$M = %d$, $\mathcal{D}$ = synthetic ($D = 2$, $N = N_{val} = N_{test}=%d$)' % (M, 1000))
    ax.set_xlabel('lifetime $\lambda$')
    ax.set_ylabel('relative error [\%]')
    ax.set_xscale('log')
    ax.yaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)

    ax.plot(lifetimes, error_train, drawstyle="steps-post", ls='-', color=tableau20(15), label='train')
    ax.plot(lifetimes, error_validation, drawstyle="steps-post", ls='-', lw=2, color=tableau20(2), label='validation')
    ax.plot(lifetimes, error_test, drawstyle="steps-post", ls='-', color=tableau20(4), label='test')

    # plot ground truth and estimate
    ax.axvline(x=10, ls=':', color='black')
    i = np.argmin(error_validation)
    lifetime_hat = lifetimes[i]
    print 'lifetime_hat = %.3f' % lifetime_hat
    ax.plot([lifetime_hat, lifetime_hat], [0, error_validation[i]], ls='dashed', lw=2, color=tableau20(2))

    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[5] = '$\lambda_0$'
    labels.append('\hat{\lambda}')
    ax.set_xticks(list(ax.get_xticks()) + [lifetime_hat])
    ax.set_xticklabels(labels)

    ax.set_xlim((1e-2, lifetime_max))

    ax.legend(bbox_to_anchor=[0.35, 0.35], frameon=False)


def experiment_CPU():
    """ Plots test error performance as a function of computational time on the CPU data set. For the Mondrian kernel,
        all lifetime values from 0 up to a terminal lifetime are swept through. For Fourier features and random binning
        a binary search procedure is employed to find good lifetime parameter values, with an initial expansion phase.
    """

    # fix random seed
    np.random.seed(9879846)

    # load data
    X, y, X_test, y_test = load_CPU()

    # set parameters
    M = 350                     # number of Mondrian trees to use
    lifetime_max = 1*1e-6       # terminal lifetime
    delta = 0.0001              # ridge regression delta

    # set up plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)
    ax.yaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('computational time [s]')
    ax.set_ylabel('validation set relative error [\%]')
    ax.set_xscale('log')
    ax.set_ylim((0, 25))

    # Fourier features
    runtimes, errors = select_width(X, y, X_test, y_test, M, delta, evaluate_fourier_features, 100)
    ax.scatter(runtimes, errors, marker='^', color=tableau20(0), label='Fourier features')
    # random binning
    runtimes, errors = select_width(X, y, X_test, y_test, M, delta, evaluate_random_binning, 50)
    ax.scatter(runtimes, errors, marker='o', color=tableau20(6), label='random binning')
    # Mondrian kernel
    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta, mondrian_kernel=True)
    ax.scatter(res['runtimes'], res['kernel_test'], marker='.', color=tableau20(4), label='Mondrian kernel')

    ax.legend(bbox_to_anchor=[0.42, 0.35], frameon=False, ncol=1)


if __name__ == "__main__":
    initialize_plotting()
    experiment_synthetic()
    experiment_CPU()
    plt.show()
