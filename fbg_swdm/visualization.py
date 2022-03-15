import numpy as np
import matplotlib.pyplot as plt

from fbg_swdm.variables import figsize, n, p, λ, Δλ, A, λ0, Δ, Q
import fbg_swdm.variables as vars
from fbg_swdm.simulation import X, normalize, denormalize, get_max_R
from fbg_swdm.deep_regresors import autoencoder_model

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
    x = X(y, vars.λ, vars.A, vars.Δλ, vars.S)

    if noise:
        x += np.random.randn(*x.shape)*noise
    return x, y 


# Plot sweep of one FBG with the other static
def plot_sweep(model, norm=True, rec_error=False, **kwargs):
    
    n_sweep = int(not kwargs['invert']) # select y row that sweeps

    x, y = _gen_sweep(**kwargs)

    autoencoder = isinstance(model, autoencoder_model)

    if norm:
        x = normalize(x)
        if autoencoder and rec_error:
            x_hat, y_hat, _ = model.batch_forward(x)
            x, y_hat = denormalize(x, y_hat)
            x_hat = denormalize(x_hat)
        else:   
            y_hat = model.predict(x)
            x, y_hat = denormalize(x, y_hat)
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
        if not autoencoder:
            x_hat = X(y_hat, vars.λ, vars.A, vars.Δλ)
        plt.figure(figsize=figsize)
        plt.plot(y[:, n_sweep]/n, np.sum(np.abs(x - x_hat), axis=1))
        plt.xlabel("$\lambda_{B_2}$ [nm]")
        plt.ylabel("$Reconstruction Error$")

def check_latent(model, K=10, add_center=True, add_border=False, **kwargs):

    autoencoder = isinstance(model, autoencoder_model)

    x, y = _gen_sweep(**kwargs)

    x, y = normalize(x, y)
    if autoencoder:
        x_hat, y_hat, latent = model.batch_forward(x)
    else:
        y_hat, latent = model.batch_forward(x)
    
    y_hat = denormalize(y=y_hat)
    y = denormalize(y=y)
    

    if add_center:
        i = len(y_hat)//2
        plt.figure(figsize=vars.figsize)
        plt.title('y_hat='+str(y_hat[i]))
        a1 = plt.gca()
        a1.plot(vars.λ, x[i])
        a2 = a1.twinx()
        a2._get_lines.prop_cycler = a1._get_lines.prop_cycler # set same color cycler
        a2.plot(vars.λ, latent[i].T)
        if autoencoder:
            a1.plot(vars.λ, x_hat[i], linestyle='-')
    if add_border:
        for i in [0, len(y_hat)-1]:
            plt.figure(figsize=vars.figsize)
            plt.title('y_hat='+str(y_hat[i]))
            a1 = plt.gca()
            a1.plot(vars.λ, x[i])
            a2 = a1.twinx()
            a2._get_lines.prop_cycler = a1._get_lines.prop_cycler # set same color cycler
            a2.plot(vars.λ, latent[i].T)
            if autoencoder:
                a1.plot(vars.λ, x_hat[i], linestyle='-')

    MAE = np.sum(np.abs(y-y_hat), axis=1) # mean absolute error
    top_n = MAE.argsort()[-K:][::-1]

    for i in top_n:
        plt.figure(figsize=vars.figsize)
        plt.title('y_hat='+str(y_hat[i]))
        a1 = plt.gca()
        a1.plot(vars.λ, x[i])
        a2 = a1.twinx()
        a2._get_lines.prop_cycler = a1._get_lines.prop_cycler # set same color cycler
        a2.plot(vars.λ, latent[i].T)
        if autoencoder:
            a1.plot(vars.λ, x_hat[i], linestyle='-')

def error_snr(model, norm=True, min_snr=0, max_snr = 40, M=10, **kwargs):
    db_vect = np.linspace(min_snr, max_snr, M)
    error_vect = np.empty(M)
    noise_vect = 10.0**(-db_vect/10.0)
    if vars.topology == 'parallel':
        noise_vect *= np.max(vars.A*get_max_R(vars.S))  
    else:
        noise_vect *= np.max(np.cumprod(vars.A)*get_max_R(vars.S))
    for i, noise in enumerate(noise_vect):
        x, y = _gen_sweep(noise=noise, **kwargs)

        if norm:
            x = normalize(x)
            y_hat = model.predict(x)
            x, y_hat = denormalize(x, y_hat)
        else:
            y_hat = model.predict(x)

        error_vect[i] = np.mean(np.abs(y - y_hat))
    
    error_vect /= vars.p # to pm
    
    plt.figure(figsize=vars.figsize)
    plt.title('Mean Absolute error vs SNR')
    plt.ylabel('Mean Absolute_error [pm]')
    plt.xlabel('SNR [dB]')
    plt.plot(db_vect, error_vect)
    plt.yscale('log')