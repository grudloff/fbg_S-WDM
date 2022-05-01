import numpy as np
import matplotlib.pyplot as plt

from fbg_swdm.variables import figsize, n, p, λ, Δλ, A, λ0, Δ, Q
import fbg_swdm.variables as vars
plt.rcParams['figure.figsize'] = vars.figsize
plt.rcParams['figure.dpi'] = vars.dpi
plt.style.context('seaborn-paper')
from fbg_swdm.simulation import X, R, normalize, denormalize, get_max_R
from fbg_swdm.deep_regresors import autoencoder_model
from scipy.signal import sawtooth
from pandas import DataFrame, concat
from seaborn import pairplot, color_palette, displot, boxplot
import os

def mae(a, b):
    return np.mean(np.abs(a-b))

def plot_datapoint(x, y, N_datapoint = None):
    if isinstance(N_datapoint, int):
        x = x[N_datapoint]
        y = y[N_datapoint]

    fig, ax = plt.subplots()
    plt.title("Datapoint visualization")
    ax.plot(vars.λ/vars.n, x, label="$x$")
    if vars.topology == 'serial':
        A = np.cumprod(vars.A)
    else:
        A = vars.A
    with color_palette(n_colors=vars.Q):
        ax.plot(vars.λ/vars.n, R(vars.λ[:, None], y[None, :],
                    A[None, :], vars.Δλ[None, :], vars.S[None, :]),
                    linestyle='--',
                    label=["$x_"+str(i+1)+"$" for i in range(vars.Q)])
        for i in range(vars.Q):
            ax.stem(y[i, None]/vars.n, np.array([1]), linefmt=':', markerfmt="None",
                basefmt="None", use_line_collection=False,
                label="$y_"+str(i+1)+"$")
        plt.xlabel("Reflection spectrum")
        plt.xlabel("$\lambda [nm]$")
        plt.legend(loc='right', bbox_to_anchor=(1.15, 0.5), frameon=False)

def plot_sweep_datapoint(N_datapoint = 0, N=300, **kwargs):
    x, y = _gen_sweep(N=N, N_datapoint=N_datapoint, **kwargs)
    plot_datapoint(x, y)

def _train_test_dataframe(y_train, y_test, column_names):
    df_train = DataFrame(data=y_train/vars.n, columns=column_names).assign(label='Train')
    df_test = DataFrame(data=y_test/vars.n, columns=column_names).assign(label='Test')
    df = concat([df_train, df_test], ignore_index=True)
    return df

def _comb_func(y):
    diff_y = np.repeat(y, range(vars.Q-1, -1, -1), axis=-1)\
        - np.concatenate(tuple(y[...,i::] for i in range(1, vars.Q)), axis=-1)
    abs_diff_y = np.abs(diff_y)
    return abs_diff_y


def plot(X_train, y_train, X_test, y_test, plot_diff=False):
    # Plot distribution of y with pairplot
    df = _train_test_dataframe(y_train, y_test, ['$y_'+str(i+1)+"[nm]$" for i in range(vars.Q)])

    g = pairplot(df, hue='label', diag_kind='hist', height=3.0, plot_kws={"s": 1},
                 diag_kws=dict(element='poly'))
    # g.fig.set_size_inches(15,15)
    g._legend.set_title(None) # remove legend title
    # _stack_diag_hist(g)

    if plot_diff:
        diff_y_train = _comb_func(y_train)
        diff_y_test = _comb_func(y_test)
        indices = range(1, vars.Q+1)
        df = _train_test_dataframe(diff_y_train, diff_y_test,
                                   ['$|y_'+str(i)+"-y_"+str(j)+"|[nm]$"
                                    for i,j in zip(np.repeat(indices, range(vars.Q-1, -1, -1)),
                                    np.concatenate(tuple(indices[i::] for i in range(1, vars.Q)), axis=-1))])
        g = pairplot(df, hue='label', diag_kind='hist', height=3.0, plot_kws={"s": 1},
                     diag_kws=dict(element='poly'))
        g._legend.set_title(None) # remove legend title

# Plot distribution
def plot_dist(y, label='Absolute Error ', short_label='AE', unit='[pm]' ,mean=True, figname=None):
    df = DataFrame(data=y, columns=['$FBG_'+str(i+1)+"$" for i in range(vars.Q)])
    g = displot(data=df, element='poly', log_scale=(True, True), stat='probability', kind="hist")
    g.set(xlabel=label+unit)
    if mean:
        g.fig.text(0.8, 0.7, "$\overline{"+short_label+'}'+"= {:.2e}".format(np.mean(y))+unit+"$")
    if figname:
        if not isinstance(figname, str):
            figname = vars.exp_dir+'\\'+vars.exp_name+'_error_dist'
        g.fig.savefig(figname+'.pdf', bbox_inches='tight')

def _gen_sweep_pair(d=0.6*n, N=300, noise=False, invert=False, N_datapoint=None):
    y = np.zeros([N, vars.Q])
    # Static
    y[:, 0] = vars.λ0
    # Sweep
    y[:, 1] = np.linspace(vars.λ0-d, vars.λ0+d, N)

    if invert:
        y = y[:,::-1]
    
    if isinstance(N_datapoint, int):
        y = y[N_datapoint]

    # broadcast shape: N, M, FBGN
    x = X(y, vars.λ, vars.A, vars.Δλ, vars.S)

    if noise:
        x += np.random.randn(*x.shape)*noise
    return x, y

def _gen_sweep_multi(d=0.6*n, N=300, noise=False, invert=None, N_datapoint=None):
    t =  np.linspace(0, 1, N)
    y = np.column_stack([d/vars.Δ*sawtooth(2*vars.π*i*t+vars.π/2, width=0.5) \
                         for i in range(vars.Q)])
    y = denormalize(y=y)
    if isinstance(N_datapoint, int):
        y = y[N_datapoint]
    x = X(y, vars.λ, vars.A, vars.Δλ, vars.S)
    if noise:
        x += np.random.randn(*x.shape)*noise
    return x, y 

def _gen_sweep(**kwargs):
    if vars.Q == 2:
        return _gen_sweep_pair(**kwargs)
    else:
        return _gen_sweep_multi(**kwargs)


# Plot sweep of one FBG with the other static
def plot_sweep(model, norm=True, rec_error=False, noise=None, **kwargs):
    
    x, y = _gen_sweep(noise=noise, **kwargs)

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

    if noise:
        noise_tag = "_noise{:.0e}".format(noise)
    else:
        noise_tag = ''
    if vars.pre_test:
        pretest_tag = '_pretest'
    else:
        pretest_tag = ''
    figname = vars.exp_dir+'\\'+vars.exp_name+'_sweep_error_dist'+noise_tag+pretest_tag
        
    plot_dist(error/p, mean=True, figname=figname)

    fig = plt.figure()

    a1 = plt.gca()
    a1.plot(y/n, linewidth=2,
            label=["$y_{"+str(i)+"}$" for i in range(1, vars.Q+1)])
    a1.plot(y_hat/n, linewidth=2, linestyle="-.",
            label=["$\hat{y}_{"+str(i)+"}$" for i in range(1, vars.Q+1)])
    a1.set_ylabel('$\lambda_{B}$ [nm]')

    a2 = a1.twinx()
    a2.plot(np.sum(error, axis=1)/p, ":r", label='MAE')
    a2.set_ylabel('Mean Absolute Error [pm]')

    fig.legend(loc='right', bbox_to_anchor=(1.085, 0.5), frameon=False)

    figname = vars.exp_dir+'\\'+vars.exp_name+'_sweep'+noise_tag+pretest_tag
    fig.savefig(figname+'.pdf', bbox_inches='tight')

    if rec_error:
        if not autoencoder:
            x_hat = X(y_hat, vars.λ, vars.A, vars.Δλ)
        fig = plt.figure()
        plt.plot(np.mean(np.abs(x - x_hat), axis=1))
        plt.ylabel("Mean Absolute Reconstruction Error")
        figname = vars.exp_dir+'\\'+vars.exp_name+'_sweep_rec_error'+noise_tag+pretest_tag
        fig.savefig(figname+'.pdf', bbox_inches='tight')

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

    MAE = np.sum(np.abs(y-y_hat), axis=1) # mean absolute error
    top_n = list(MAE.argsort()[-K:][::-1])

    if add_center:
        top_n.append(len(y_hat)//2)
    
    if add_border and vars.Q==2:
        top_n.extend([0, len(y_hat)-1])

    for i in top_n:
        plt.figure()
        plt.title('$\hat{y} = ${'+' , '.join([ '%.2f' % elem for elem in y_hat[i]/vars.n])+"}$[nm]$")
        plt.xlabel('$\lambda [nm]$')
        a1 = plt.gca()
        a1.plot(vars.λ/n, x[i])
        a1.set_ylabel('$x$')
        a2 = a1.twinx()
        a2._get_lines.prop_cycler = a1._get_lines.prop_cycler # set same color cycler
        a2.plot(vars.λ/n, latent[i].T)
        a2.set_ylabel(r'$\tilde{y}$')


def error_snr(model, norm=True, min_snr=0, max_snr = 40, M=10, split=False, N=300 ,**kwargs):
    db_vect = np.linspace(min_snr, max_snr, M)
    noise_vect = 10.0**(-db_vect/10.0)
    if not split:
        Q = 1
        noise_vect *= np.max(vars.A*get_max_R(vars.S))
        noise_vect = noise_vect[None, :]
    else:
        Q = vars.Q
        noise_vect = np.outer(vars.A*get_max_R(vars.S), noise_vect)
    error_vect = np.empty((Q, M, N))
    for i in range(Q):
        for j, noise in enumerate(noise_vect[i]):

            x, y = _gen_sweep(noise=noise, N=N, **kwargs)

            if norm:
                x = normalize(x)
                y_hat = model.predict(x)
                x, y_hat = denormalize(x, y_hat)
            else:
                y_hat = model.predict(x)

            error_vect[i, j] = np.mean(np.abs(y - y_hat), axis=-1)

    error_vect = error_vect.squeeze()
    error_vect /= vars.p # to pm

    pretest_tag = '_pretest' if vars.pre_test else ''
    split_tag = '_split' if split else ''
    tag = pretest_tag + split_tag
    save_file = vars.exp_dir+'\\'+vars.exp_name+'_error_snr'+tag+'.npz'
    with open(vars.exp_dir+'\\log.txt','a') as file:
        file.write("error_snr: "+save_file+'\n')
    with open(save_file, 'wb') as f:
        np.savez(f, db_vect=db_vect, error_vect=error_vect)
    
    if not split:
        db_vect = np.trunc(db_vect*10)/10
        boxplot(x=np.repeat(db_vect, N), y=error_vect.flatten(), color='#1f77b4')
    else:
        db_vect = np.trunc(db_vect*10)/10
        group_box_plot(db_vect, error_vect, labels=["$FBG_{}$".format(i+1) for i in range(Q)],
                       title = 'SNR reference')
    plt.ylabel('Absolute Error [pm]')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')

def group_box_plot(x, y, labels, title=None):
    Q, M, N = y.shape
    data = [np.stack((np.repeat(x, N), y[i].flatten()), axis=-1) for i in range(Q)]
    df = concat([DataFrame(data=data[i], columns=['x', 'y']).assign(label=labels[i]) 
                for i in range(vars.Q)], ignore_index=True)
    boxplot(data=df, x='x', y='y', hue='label')
    plt.legend(title=title, loc='upper right', frameon=False)