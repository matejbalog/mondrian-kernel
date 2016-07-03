import matplotlib.pylab as plt
import numpy as np

from mondrian_kernel import evaluate_all_lifetimes
from utils import load_CPU, initialize_plotting, remove_chartjunk, tableau20


def plot_mondrian_kernel_vs_mondrian_forest(lifetime_max, res):
    """ Plots training and test set error of Mondrian kernel and Mondrian forest based on the same set of M Mondrian samples.
        This procedure takes as input a dictionary res, returned by the evaluate_all_lifetimes procedure in mondrian_kernel.py.
    """

    times = res['times']
    forest_train = res['forest_train']
    forest_test = res['forest_test']
    kernel_train = res['kernel_train']
    kernel_test = res['kernel_test']

    # set up test error plot
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot('111')
    remove_chartjunk(ax)

    ax.set_xlabel('lifetime $\lambda$')
    ax.set_ylabel('relative error [\%]')
    ax.yaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)

    ax.set_xscale('log')
    ax.set_xlim((1e-8, lifetime_max))
    ax.set_ylim((0, 25))

    rasterized = False
    ax.plot(times, forest_test, drawstyle="steps-post", ls='-', lw=2, color=tableau20(6), label='"M. forest" (test)', rasterized=rasterized)
    ax.plot(times, forest_train, drawstyle="steps-post", ls='-', color=tableau20(7), label='"M. forest" (train)', rasterized=rasterized)
    ax.plot(times, kernel_test, drawstyle="steps-post", ls='-', lw=2, color=tableau20(4), label='M. kernel (test)', rasterized=rasterized)
    ax.plot(times, kernel_train, drawstyle="steps-post", ls='-', color=tableau20(5), label='M. kernel (train)', rasterized=rasterized)

    ax.legend(bbox_to_anchor=[1.15, 1.05], frameon=False)


def plot_kernel_vs_forest_weights(y, res):
    """ Plots the weights learned by Mondrian kernel and Mondrian forest based on the same set of M Mondrian samples.
        This procedure takes as input a dictionary res, returned by the evaluate_all_lifetimes procedure in mondrian_kernel.py.
    """

    w_forest = res['w_forest']
    w_kernel = res['w_kernel']

    # plot weights against each other
    fig1 = plt.figure(figsize=(8, 4))
    ax1 = fig1.add_subplot('121')
    ax1.set_xlabel('weights learned by "Mondrian forest"')
    ax1.set_ylabel('weights learned by Mondrian kernel')
    ax1.scatter(w_forest, w_kernel, marker='.', color=tableau20(16))
    xl = ax1.get_xlim()
    yl = ax1.get_ylim()
    lims = [
        np.min([xl, yl]),  # min of both axes
        np.max([xl, yl]),  # max of both axes
    ]
    ax1.plot(lims, lims, '--', color='black', alpha=0.75, zorder=0)
    ax1.set_xlim(xl)
    #ax1.set_ylim(yl)
    ax1.set_ylim((-60, 60))

    # plot histogram of weight values (and training targets)
    ax2 = fig1.add_subplot('122')
    ax2.set_xlabel('values')
    ax2.set_ylabel('value frequency')
    bins = np.linspace(-100, 20, 50)
    ax2.hist(w_forest, bins=bins, histtype='stepfilled', normed=True, color=tableau20(6), alpha=0.5,
             label='M. forest weights $\mathbf{w}$')
    ax2.hist(w_kernel, bins=bins, histtype='stepfilled', normed=True, color=tableau20(4), alpha=0.5,
             label='M. kernel weights $\mathbf{w}$')
    ax2.hist(y - np.mean(y), bins=bins, histtype='stepfilled', normed=True, color=tableau20(8), alpha=0.5,
             label='training targets $\mathbf{y}$')
    ax2.set_ylim((0.0, 0.16))
    ax2.legend(frameon=False, loc='upper left')

    fig1.tight_layout()


def main():
    # randomness seed
    np.random.seed(9879846)

    # load data
    X, y, X_test, y_test = load_CPU()

    # set experiment parameters
    M = 50                      # number of Mondrian trees to use
    lifetime_max = 1*1e-5       # terminal lifetime
    weights_lifetime = 2*1e-6   # lifetime for which weights should be plotted
    delta = 0.0001              # ridge regression delta

    # run both Mondrian kernel and Mondrian forest from lifetime 0 up to lifetime_max
    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta,
                                 mondrian_kernel=True, mondrian_forest=True, weights_from_lifetime=weights_lifetime)

    # plot
    plot_mondrian_kernel_vs_mondrian_forest(lifetime_max, res)
    plot_kernel_vs_forest_weights(y, res)


if __name__ == "__main__":
    initialize_plotting()
    main()
    plt.show()
