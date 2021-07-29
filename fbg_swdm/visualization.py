import numpy as np
import matplotlib.pyplot as plt

from fbg_swdm.variables import figsize, n, p, λ, Δλ, A, λ0, Δ, Q
import fbg_swdm.variables as vars
from fbg_swdm.simulation import R, X


def mae(a, b):
    return np.mean(np.abs(a-b))


# Plot distribution
def plot_dist(y, y_label=None, mean=False):
    plt.figure(figsize=figsize)
    plt.hist(y, bins=100, stacked=True, density=True)
    plt.xlabel(y_label)
    if mean:
        print("mean("+y_label+") =", np.mean(y))


# Plot sweep of one FBG with the other static
def plot_sweep(model, d=0.6*n, normalize=True, rec_error=False, N=300):
    
    y = np.zeros([N, vars.Q])
    # Static
    y[:, 0] = vars.λ0
    # Sweep
    y[:, 1] = np.linspace(vars.λ0-d, vars.λ0+d, N)

    # broadcast shape: N, M, FBGN
    X_sweep = X(y, vars.λ, vars.A, vars.Δλ)

    if normalize:
        X_sweep = X_sweep/np.sum(vars.A)  # scale input
        y_hat = vars.λ0+vars.Δ*model.predict(X_sweep)
        X_sweep = X_sweep*np.sum(vars.A)

    else:
        y_hat = model.predict(X_sweep)

    error = np.abs(y - y_hat)

    plot_dist(error/p, "Absolute Error [pm]", mean=True)

    fig = plt.figure(figsize=figsize)

    a1 = plt.gca()
    a1.plot(y[:, 1]/n, y_hat/n, linewidth=2)
    a1.set_xlabel("$\lambda_{B_2}$ [nm]")
    a1.set_ylabel('$\lambda_{B_i}$ [nm]')

    a2 = a1.twinx()
    a2.plot(y[:, 1]/n, np.sum(error, axis=1)/p, "--r")
    a2.set_ylabel('Absolute Error [pm]')
    fig.legend(labels=("FBG1", "FBG2", "Absolute Error"))

    if rec_error:
        plt.figure(figsize=figsize)
        X_hat = X(y_hat, vars.λ, vars.A, vars.Δλ)
        plt.plot(y[:, 1]/n, np.sum(np.abs(X_sweep - X_hat), axis=1))
        plt.xlabel("$\lambda_{B_2}$ [nm]")
        plt.ylabel("$Reconstruction Error$")
