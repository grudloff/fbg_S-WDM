import numpy as np
import matplotlib.pyplot as plt

from fbg_swdm.variables import figsize, n, p, λ, Δλ, A, λ0, Δ, Q
import fbg_swdm.variables as vars
plt.rcParams['figure.figsize'] = vars.figsize
plt.rcParams['figure.dpi'] = vars.dpi
from fbg_swdm.simulation import X, R, normalize, denormalize, get_max_R
from fbg_swdm.deep_regresors import autoencoder_model
from scipy.signal import sawtooth
from pandas import DataFrame, concat
from seaborn import pairplot

def mae(a, b):
    return np.mean(np.abs(a-b))

def plot_datapoint(X, Y, N_datapoint = 1):
    plt.figure(figsize=(20, 10))
    plt.title("Datapoint visualization")
    plt.plot(vars.λ/vars.n, X[N_datapoint, :], label="$\sum FBGi$")
    for i in range(vars.Q):
        plt.plot(vars.λ/vars.n, R(vars.λ[:, None], Y[N_datapoint, None, i],
                 vars.A[None, i], vars.Δλ[None, i], vars.S[None, i]),
                 linestyle='dashed',
                 label="FBG"+str(i))
    plt.stem(Y[N_datapoint, :]/vars.n, np.full(vars.Q, 1), linefmt='r-.', markerfmt="None",
             basefmt="None", use_line_collection=True)
    plt.xlabel("Reflection spectrum")
    plt.xlabel("[nm]")
    plt.legend()

def plot(X_train, y_train, X_test, y_test):
    df_train = DataFrame(data=y_train/vars.n, 
                        columns=['$y_'+str(i+1)+"$" for i in range(vars.Q)])\
                       .assign(label='Train')
    df_test = DataFrame(data=y_test/vars.n, 
                            columns=['$y_'+str(i+1)+"$" for i in range(vars.Q)])\
                        .assign(label='Test')
    df = concat([df_train, df_test], ignore_index=True)

    g = pairplot(df, hue='set', diag_kind='hist', markers='.')
    g._legend.set_title(None) # remove legend title


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

def _gen_triang_sweep(d=0.6*n, N=300, noise=False):
    t =  np.linspace(0, 1, N)
    y = np.column_stack([d/vars.Δ*sawtooth(2*vars.π*i*t+vars.π/2, width=0.5) \
                         for i in range(vars.Q)])
    y = denormalize(y=y)
    x = X(y, vars.λ, vars.A, vars.Δλ, vars.S)
    if noise:
        x += np.random.randn(*x.shape)*noise
    return x, y 


# Plot sweep of one FBG with the other static
def plot_sweep(model, norm=True, rec_error=False, invert=False, **kwargs):
    

    if vars.Q == 2:
        x, y = _gen_sweep(invert=invert, **kwargs)
    else:
        x, y = _gen_triang_sweep(**kwargs)

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
    a1.plot(y/n, linewidth=2,
            label=["$y_{i}$" for i in range(1, vars.Q+1)])
    a1.plot(y_hat/n, linewidth=2, linestyle="-.",
            label=["$\hat{y}_{i}$" for i in range(1, vars.Q+1)])
    a1.set_ylabel('$\lambda_{B}$ [nm]')

    a2 = a1.twinx()
    a2.plot(np.sum(error, axis=1)/p, ":r", label='Absolute Error')
    a2.set_ylabel('Absolute Error [pm]')
    fig.legend()

    if rec_error:
        if not autoencoder:
            x_hat = X(y_hat, vars.λ, vars.A, vars.Δλ)
        plt.figure(figsize=figsize)
        plt.plot(np.mean(np.abs(x - x_hat), axis=1))
        plt.ylabel("Mean Absolute Reconstruction Error")

def check_latent(model, K=10, add_center=True, add_border=False, **kwargs):

    autoencoder = isinstance(model, autoencoder_model)

    
    if vars.Q == 2:
        x, y = _gen_sweep(**kwargs)
    else:
        x, y = _gen_triang_sweep(**kwargs)

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
        plt.title('$\hat{y} = '+str(y_hat[i])+"$")
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
            plt.title('$\hat{y} =$'+str(y_hat[i])+"$")
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
        plt.title('$\hat{y}='+str(y_hat[i])+"$")
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
    noise_vect *= np.max(vars.A*get_max_R(vars.S))  
    for i, noise in enumerate(noise_vect):
        if vars.Q == 2:
            x, y = _gen_sweep(noise=noise, **kwargs)
        else:
            x, y = _gen_triang_sweep(noise=noise, **kwargs)

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