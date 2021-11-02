import numpy as np
import matplotlib.pyplot as plt

from fbg_swdm.variables import figsize, n, p, λ, Δλ, A, λ0, Δ, Q
import fbg_swdm.variables as vars
from fbg_swdm.simulation import R, X

import torch

def mae(a, b):
    return np.mean(np.abs(a-b))


# Plot distribution
def plot_dist(y, y_label=None, mean=False):
    plt.figure(figsize=figsize)
    plt.hist(y, bins=100, stacked=True, density=True)
    plt.xlabel(y_label)
    if mean:
        print("mean("+y_label+") =", np.mean(y))


def _gen_sweep(d=0.6*n, N=300, noise=False, invert=False):
    y = np.zeros([N, vars.Q])
    # Static
    y[:, 0] = vars.λ0
    # Sweep
    y[:, 1] = np.linspace(vars.λ0-d, vars.λ0+d, N)

    if invert:
        y = y[:,::-1]

    # broadcast shape: N, M, FBGN
    x = X(y, vars.λ, vars.A, vars.Δλ)
    if noise:
        x += np.random.randn(*x.shape)*1e-3*noise
    return x, y 


# Plot sweep of one FBG with the other static
def plot_sweep(model, normalize=True, rec_error=False, **kwargs):
    
    n_sweep = int(not kwargs['invert']) # select y row that sweeps

    x, y = _gen_sweep(**kwargs)

    if normalize:
        x = x/np.sum(vars.A)  # scale input
        y_hat = vars.λ0+vars.Δ*model.predict(x)
        x = x*np.sum(vars.A)

    else:
        y_hat = model.predict(x)

    error = np.abs(y - y_hat)

    plot_dist(error/p, "Absolute Error [pm]", mean=True)

    fig = plt.figure(figsize=figsize)

    a1 = plt.gca()
    a1.plot(y[:, n_sweep]/n, y_hat/n, linewidth=2)
    a1.set_xlabel("$\lambda_{B_2}$ [nm]")
    a1.set_ylabel('$\lambda_{B_i}$ [nm]')

    a2 = a1.twinx()
    a2.plot(y[:, n_sweep]/n, np.sum(error, axis=1)/p, "--r")
    a2.set_ylabel('Absolute Error [pm]')
    fig.legend(labels=("FBG1", "FBG2", "Absolute Error"))

    if rec_error:
        plt.figure(figsize=figsize)
        X_hat = X(y_hat, vars.λ, vars.A, vars.Δλ)
        plt.plot(y[:, n_sweep]/n, np.sum(np.abs(x - X_hat), axis=1))
        plt.xlabel("$\lambda_{B_2}$ [nm]")
        plt.ylabel("$Reconstruction Error$")

def check_latent(model, K=10,  **kwargs):

    x, y = _gen_sweep(**kwargs)

    y = (y - vars.λ0)/vars.Δ
    input = torch.tensor(x, dtype=torch.get_default_dtype(), device=model.device)
    
    x = x/np.sum(vars.A)  # scale input
    y_hat, latent = model(input)
    y_hat = vars.λ0+vars.Δ*y_hat
    x = x*np.sum(vars.A)
    y_hat = y_hat.detach().numpy()

    AE = np.abs(y-y_hat)
    AE = np.sum(AE, axis=1)/vars.p
    top_n = AE.argsort()[-K:][::-1]

    for n in top_n:
        test = x[n]
        input = torch.tensor(test, dtype=torch.get_default_dtype(), device=model.device)
        input = input.unsqueeze(0) # add batch dim
        y_hat, latent = model(input)
        y_hat = np.squeeze(y_hat.detach().numpy())
        latent = latent.detach().numpy()
        latent = latent.T
        latent = np.squeeze(latent)
        plt.figure()
        plt.plot(test)
        plt.plot(latent)
        plt.title('y='+str(y[n])+' y_hat='+str(y_hat))