import numpy as np
import numpy.random as npr
npr.seed(0)
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
colors = ['r', 'b', 'y', 'g']
colors = [ColorConverter().to_rgb(c) for c in colors]

from pypolyagamma import MixtureOfMultinomialRegressions
from pypolyagamma.utils import compute_psi_cmoments, gradient_cmap


def _plot_mult_probs(reg,
                     xlim=(-4,4), ylim=(-3,3), n_pts=100,
                     fig=None):
    XX,YY = np.meshgrid(np.linspace(*xlim,n_pts),
                        np.linspace(*ylim,n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))

    D_reg = reg.D_in
    inputs = np.hstack((np.zeros((n_pts**2, D_reg-2)), XY))
    test_prs = reg.pi(inputs)

    if fig is None:
        fig = plt.figure(figsize=(10,6))

    for k in range(reg.K):
        ax = fig.add_subplot(1,reg.K,k+1)
        cmap = gradient_cmap([np.ones(3), colors[k]])
        ax.imshow(test_prs[:,k].reshape(*XX.shape),
                  extent=xlim + tuple(reversed(ylim)),
                  vmin=0, vmax=1, cmap=cmap)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout()
    return ax

if __name__ == "__main__":
    ### Construct multinomial regression to divvy up the space #
    M, N, K, D_in = 3, 1000, 4, 2

    # Account for stick breaking asymmetry
    mu_b, _ = compute_psi_cmoments(np.ones(K))

    # Ps = [np.eye(K) for _ in range(M)]
    Ps = None
    true_reg = MixtureOfMultinomialRegressions(
        M=M, N=1, D_out=K, D_in=D_in, Ps=Ps,
        sigmasq_A=1000,  sigmasq_b=1000, mu_b=mu_b)

    # Sample data from the model
    X = np.random.randn(N,2).dot(np.diag([2,1]))
    y_oh = true_reg.rvs(x=X).astype(np.float)
    y = np.argmax(y_oh, axis=1)
    usage = y_oh.sum(0)
    print("Label usage: ", usage)

    # Apply a random permutation
    # perm = np.random.permutation(K)
    # perm = np.argsort(np.argsort(-usage))
    perm = np.arange(K)
    y_oh_perm = y_oh[:, perm]
    y_perm = np.argmax(y_oh_perm, axis=1)

    ### Create a test model for fitting
    test_reg = MixtureOfMultinomialRegressions(
        M=M, N=1, D_out=K, D_in=D_in,
        sigmasq_A=1000., sigmasq_b=1000.)

    # test_reg.Ps = true_reg.Ps

    lls = []
    for itr in range(1000):
        if itr % 10 == 0:
            print("Iter: {}".format(itr))
        test_reg.resample(data=[(X, y_oh[:, :-1])])
        lls.append(test_reg.log_likelihood((X, y_oh_perm[:, :-1])).sum())
    #
    # np.set_printoptions(precision=3)
    # print("True A:\n{}".format(true_reg.A))
    # print("True b:\n{}".format(true_reg.b))
    # print("Test A:\n{}".format(test_reg.A))
    # print("Test b:\n{}".format(test_reg.b))

    ### Plot the results
    fig = plt.figure(figsize=(5,5))
    plt.plot(lls)
    plt.xlabel("Iteration")
    plt.xlabel("Log Likelihood")

    fig = plt.figure(figsize=(10,3))
    _plot_mult_probs(true_reg, fig=fig)
    for k in range(K):
        ax = fig.add_subplot(1, K, k+1)
        ax.plot(X[y==k, 0], X[y==k, 1], 'o', color=colors[k], markeredgecolor="none")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("$\Pr(z={} \mid x)$".format(k+1))
    # plt.savefig("mixture_multinomial_regression.png")

    fig = plt.figure(figsize=(10, 3))
    _plot_mult_probs(test_reg, fig=fig)
    for k in range(K):
        ax = fig.add_subplot(1, K, k + 1)
        ax.plot(X[y_perm == k, 0], X[y_perm == k, 1], 'o', color=colors[k], markeredgecolor="none")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("$\Pr(z={} \mid x)$".format(k + 1))
    plt.show()

