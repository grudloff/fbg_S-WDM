import numpy as np
import matplotlib.pyplot as plt
import warnings
from fbg_swdm.variables import n, p # prefixes
import fbg_swdm.variables as config
import fbg_swdm.simulation as sim
from fbg_swdm.simulation import X, R, normalize, denormalize, get_I, prep_dims
from fbg_swdm.deep_regresors import autoencoder_model
import fbg_swdm.evolutionary as ev
from scipy.signal import sawtooth
from pandas import DataFrame, concat
from seaborn import pairplot, color_palette, displot, boxplot
import os
from os.path import join
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['figure.figsize'] = config.figsize
plt.rcParams['figure.dpi'] = config.dpi
plt.style.context('seaborn-paper')

def mae(a, b):
    """ Mean absolute values """
    return np.mean(np.abs(a-b))

def plot_datapoint(x, y, i=None, normalized=False, **kwargs):
    """ Plot one datapoint and compare to per sensor simulated spectra.

        Parameters
    ----------
    X : array-like, shape (n_samples, n_features) or (n_features)
        Inputs.
    y : array-like, shape (n_samples, n_targets) or (n_targets)
        Targets.
    i : int (optional) 
        Instance to select. If None, the instance must be passed as directly as input.
    normalized : Bool
        Wether the input data is normalized
    """
    if "N_datapoint" in kwargs.keys(): 
        warnings.warn("N_datapoint is deprecated, use 'i' instead.")
        i = kwargs.pop("N_datapoint") 

    if isinstance(i, int):
        x = x[i]
        y = y[i]

    if config.topology == 'parallel':
        A = config.A
    else:
        A = np.cumprod(config.A)
    
    if normalized:
        y = sim.denormalize(y=y)
    A_b, λ, A, Δλ, I, Δn_dc = prep_dims(y, config.λ, A, config.Δλ, config.I, config.Δn_dc)
    r = R(A_b, λ, A, Δλ, I, Δn_dc)
    λ = λ/config.n
    y = y/config.n
        
    fig, ax = plt.subplots()
    plt.title("Datapoint visualization")    
    ax.plot(λ, x, label="$x$")

    with color_palette(n_colors=config.Q):
        ax.plot(λ, r,
                linestyle='--',
                label=["$x_"+str(i+1)+"$" for i in range(config.Q)])
        for i in range(config.Q):
            ax.stem(y[i, None], np.array([1]), linefmt='--', markerfmt="None",
                basefmt="None", use_line_collection=False,
                label="$y_"+str(i+1)+"$")
        plt.xlabel("Reflection spectrum")
        plt.xlabel("$\lambda [nm]$")
        plt.legend(loc='right', bbox_to_anchor=(1.15, 0.5), frameon=False)

def plot_sweep_datapoint(i=0, N=300, **kwargs):
    # simulate sweep and plot n datapoint
    x, y = _gen_sweep(N=N, i=i, **kwargs)
    plot_datapoint(x, y)

def _train_test_dataframe(y_train, y_test, column_names):
    # Create dataframe with column specifying if sample is train or test
    df_train = DataFrame(data=y_train/config.n, columns=column_names).assign(label='Train')
    df_test = DataFrame(data=y_test/config.n, columns=column_names).assign(label='Test')
    df = concat([df_train, df_test], ignore_index=True)
    return df

def _comb_func(y):
    # Absolute differences between features for every sample
    diff_y = np.repeat(y, range(config.Q-1, -1, -1), axis=-1)\
        - np.concatenate(tuple(y[...,i::] for i in range(1, config.Q)), axis=-1)
    abs_diff_y = np.abs(diff_y)
    return abs_diff_y


def plot(X_train, y_train, X_test=None, y_test=None, plot_diff=False):
    """ Plot pairplot of training targets, and optionally pairplot of differences.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training inputs.
    y_train : array-like, shape (n_samples, n_targets)
        Training targets.
    X_test : array-like, shape (n_samples, n_features) (optional)
     Testing inputs.
    y_test : array-like, shape (n_samples, n_targets) (optional)
        Testing targets.
    """

    
    labels = ['$y_'+str(i+1)+"[nm]$" for i in range(config.Q)]
    if y_test is None:
        df = DataFrame(data=y_train/config.n, 
                       columns=labels).assign(label='Train')
    else:
        df = _train_test_dataframe(y_train, y_test, labels)

    g = pairplot(df, hue='label', diag_kind='hist', height=3.0, plot_kws={"s": 1},
                 diag_kws=dict(element='poly'))
    # g.fig.set_size_inches(15,15)
    g._legend.set_title(None) # remove legend title
    # _stack_diag_hist(g)

    if plot_diff:
        diff_y_train = _comb_func(y_train)
        indices = range(1, config.Q+1)
        labels = ['$|y_'+str(i)+"-y_"+str(j)+"|[nm]$"
                  for i,j in zip(np.repeat(indices, range(config.Q-1, -1, -1)),
                  np.concatenate(tuple(indices[i::] for i in range(1, config.Q)), axis=-1))]
        if X_test is None:
            df = DataFrame(data=diff_y_train/config.n, 
                        columns=labels).assign(label='Train')
        else:
            diff_y_test = _comb_func(y_test)
            df = _train_test_dataframe(diff_y_train, diff_y_test, labels)
        g = pairplot(df, hue='label', diag_kind='hist', height=3.0, plot_kws={"s": 1},
                     diag_kws=dict(element='poly'))
        g._legend.set_title(None) # remove legend title

# Plot distribution
def plot_dist(y, label='Absolute Error ', short_label='AE', unit='[pm]' ,mean=True, figname=None):
    """ Plot distribution of y.
    
    Parameters
    ----------
    y : array-like, shape (n_samples, n_targets)
        Targets.
    label: str
        Long label.
    short_label: str
        Short label.
    unit: str
        Unit of targets
    mean: Bool
        Wether mean distribution is added.
    figname: str (optional)
        Name of figure for saving. If None, no figure is saved.
    """
    df = DataFrame(data=y, columns=['$FBG_'+str(i+1)+"$" for i in range(config.Q)])
    try:
        g = displot(data=df, element='poly', log_scale=(True, False), stat='probability', kind="hist")
    except ValueError:
        # If data has zero previous call will fail with log scale
        g = displot(data=df, element='poly', log_scale=(False, False), stat='probability', kind="hist")
        g.set(xscale='symlog')

    g.set(xlabel=label+unit)
    if mean:
        g.fig.text(0.8, 0.7, "$\overline{"+short_label+'}'+"= {:.2e}".format(np.mean(y))+unit+"$")
    if figname:
        if not isinstance(figname, str):
            figname = join(config.exp_dir, 
                           "_".join(filter(None, (config.exp_name, config.tag, 'error_dist'))))
        g.fig.savefig(figname+'.pdf', bbox_inches='tight')

def _gen_sweep_pair(d=0.6*n, N=300, noise=False, invert=False, N_datapoint=None, **kwargs):
    """ Generate simulated data of one fixed sensor and another in a sweep.

    Parameters
    ----------
    d : float
        Half sweep range in [m].
    N : int
        Number of points.
    noise: Bool of float
        Wether to add noise or not. If float, set noise value.
    invert: Bool
        Wether sweep sensor is sensor 0 or 1.
    N_datapoint: int (optional)
        If provided, only this sample is returned.
    
    """
    y = np.zeros([N, config.Q])
    # Static
    y[:, 0] = config.λ0
    # Sweep
    y[:, 1] = np.linspace(config.λ0-d, config.λ0+d, N)

    if invert:
        y = y[:,::-1]
    
    if isinstance(N_datapoint, int):
        y = y[N_datapoint]

    # broadcast shape: N, M, FBGN
    x = X(y)

    if noise:
        x = x + np.random.randn(*x.shape)*noise
    return x, y

def _gen_sweep_multi(d=0.6*n, N=300, N_datapoint=None, **kwargs):
    """ Generate simulated data of sweeping pattern that provides 
        roughly uniform differences between targets.

    Parameters
    ----------
    d : float
        Half sweep range in [m].
    N : int
        Number of points.
    invert: Bool
        Wether sweep sensor is sensor 0 or 1.
    N_datapoint: int (optional)
        If provided, only this sample is returned.
    """

    t =  np.linspace(0, 1, N)
    y = np.column_stack([d/config.Δ*sawtooth(2*config.π*i*t+config.π/2, width=0.5) \
                         for i in range(config.Q)])
    y = denormalize(y=y)
    if isinstance(N_datapoint, int):
        y = y[N_datapoint]
    x = X(y)
    return x, y 

def _gen_sweep(**kwargs):
    # Generate simulated data of sweeping depending on Q
    if config.Q == 2:
        return _gen_sweep_pair(**kwargs)
    else:
        return _gen_sweep_multi(**kwargs)

def predict_plot(model, x, y, norm=True, rec_error=False, noise=None, **kwargs):
    """ Predict from data, plot prediction and target together with error per sample.
        Plot error distribution per target and optionally plot the reconstruction error.

        Parameters
        ----------
        model : object with predict method
            Predictor.
        X : array-like, shape (n_samples, n_features) or (n_features)
            Inputs.
        y : array-like, shape (n_samples, n_targets) or (n_targets)
            Targets.
        norm : bool
            Wether data should be normalized.
        rec_error: bool
            Wether to plot reconstruction error.
        noise: bool or float
            Wether to add noise. If float, sets noise amplitude.
    """

    autoencoder = isinstance(model, autoencoder_model)

    if noise:
        x = x + np.random.randn(*x.shape)*noise

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

    noise_tag = "noise{:.0e}".format(noise) if noise else None
    pretest_tag = 'pretest' if config.pre_test else None
    tag = (noise_tag, pretest_tag)
    figname = join(config.exp_dir, 
                "_".join(filter(None, (config.exp_name, config.tag, 'sweep_error_dist',*tag)))) 
    plot_dist(error/p, mean=True, figname=figname)

    fig = plt.figure()

    a1 = plt.gca()
    a1.plot(y/n, linewidth=2,
            label=["$y_{"+str(i)+"}$" for i in range(1, config.Q+1)])
    a1.plot(y_hat/n, linewidth=2, linestyle="-.",
            label=["$\hat{y}_{"+str(i)+"}$" for i in range(1, config.Q+1)])
    a1.set_ylabel('$\lambda_{B}$ [nm]')

    a2 = a1.twinx()
    a2.plot(np.sum(error, axis=1)/p, ":k", label='MAE')
    a2.set_ylabel('Mean Absolute Error [pm]')
    a2.set_ylim(bottom=0)
    # a2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.legend(loc='right', bbox_to_anchor=(1.085, 0.5), frameon=False)

    figname = join(config.exp_dir, 
            "_".join(filter(None, (config.exp_name, config.tag, 'sweep',*tag))))
    fig.savefig(figname+'.pdf', bbox_inches='tight')

    if rec_error:
        if not autoencoder:
            x_hat = X(y_hat)
        if isinstance(model, ev.GeneticAlgo):
            x_hat = X(y_hat, ev.config.λ, ev.config.A, ev.config.Δλ, ev.config.I, 
                      ev.config.Δn_dc, simulation=ev.config.simulation)
        fig = plt.figure()
        plt.plot(np.mean(np.abs(x - x_hat), axis=1))
        plt.ylabel("Mean Absolute Reconstruction Error")
        plt.ylim(bottom=0)
        
        figname = join(config.exp_dir, 
                     "_".join(filter(None, (config.exp_name, config.tag, 'sweep_rec_error',*tag))))
        fig.savefig(figname+'.pdf', bbox_inches='tight')

def plot_sweep(model, **kwargs):
    """ Simulate sweep and call predict_plot.

        Parameters
        ----------
        model : object with predict method
            Predictor.  
    """

    x, y = _gen_sweep(**kwargs)
    predict_plot(model, x, y, **kwargs)

def check_reconstruction(model, x, y, K=10, add_center=True, add_border=False, **kwargs):
    """ Predict from data. Plot worst performing reconstructions and latent representation if model is autoencoder

        Parameters
        ----------
        model : object with predict method
            Predictor.
        X : array-like, shape (n_samples, n_features) or (n_features)
            Inputs.
        y : array-like, shape (n_samples, n_targets) or (n_targets)
            Targets.
        K : int
            Number of instances to plot.
        add_center: bool
            Wether to plot center sample.
        add_border: bool
            Wether to plot border samples.
    """

    autoencoder = isinstance(model, autoencoder_model)
    evolutionary = isinstance(model, ev.GeneticAlgo)

    if evolutionary: 
        y_hat = model.predict(x)
        # simulate with evolutionary instance parameters
        x_hat = X(y_hat, ev.config.λ, ev.config.A, ev.config.Δλ, ev.config.I, 
                  ev.config.Δn_dc, simulation=ev.config.simulation)
    else:
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
    
    if add_border and config.Q==2:
        top_n.extend([0, len(y_hat)-1])

    for i in top_n:
        plt.figure()
        plt.xlabel('$\lambda [nm]$')
        a1 = plt.gca()
        a1.plot(config.λ/n, x[i], label='$x$')
        a1.set_ylabel('$x$')
        for j in range(config.Q):
            a1.stem(y[i, j, None]/config.n, np.max(x[i], keepdims=True), linefmt='-', markerfmt="None",
                basefmt="None", use_line_collection=False,
                label="$y_"+str(j+1)+"$")

        if not evolutionary:
            a2 = a1.twinx()
            a2._get_lines.prop_cycler = a1._get_lines.prop_cycler # set same color cycler
            a2.plot(config.λ/n, latent[i].T, label=[r"$\tilde{y}_{"+str(i+1)+"}$" for i in range(config.Q)])
            a2.set_ylabel(r'$\tilde{y}$')
        else:
            a1.plot(config.λ/n, x_hat[i], label="$\hat{x}$")
            for j in range(config.Q):
                a1.stem(y_hat[i, j, None]/config.n, np.max(x[i], keepdims=True), linefmt='--', markerfmt="None",
                    basefmt="None", use_line_collection=False,
                    label="$\hat{y}_"+str(j+1)+"$")
        plt.legend()

def check_sweep_reconstruction(model, **kwargs):
    """ Simulate sweep and call check_reconstruction.

        Parameters
        ----------
        model : object with predict method
            Predictor.  
    """

    x, y = _gen_sweep(**kwargs)
    check_reconstruction(model, x, y, **kwargs)

def error_snr(model, norm=None, min_snr=0, max_snr = 40, M=10, split=True, N=300 ,**kwargs):
    db_vect = np.linspace(min_snr, max_snr, M)
    noise_vect = 10.0**(-db_vect/10.0)
    noise_vect *= np.max(config.A*config.I)
    error_vect = np.empty((M, N, config.Q))
    if norm is None:
        # default to False only for evolutionary algorithms
        norm = False if isinstance(model, ev.GeneticAlgo) else True
    for i, noise in enumerate(noise_vect):

        x, y = _gen_sweep(noise=noise, N=N, **kwargs)

        if norm:
            x = normalize(x)
            y_hat = model.predict(x)
            x, y_hat = denormalize(x, y_hat)
        else:
            y_hat = model.predict(x)

        error = np.abs(y - y_hat)
        error_vect[i] = error

    error_vect /= config.p # to pm

    pretest_tag = 'pretest' if config.pre_test else None
    save_file = join(config.exp_dir, 
                     "_".join(filter(None, (config.exp_name, config.tag, 'error_snr', pretest_tag))))
    with open(config.exp_dir+'\\log.txt','a') as file:
        file.write("error_snr: "+save_file+'.npz'+'\n')
    with open(save_file+'.npz', 'wb') as f:
        np.savez(f, db_vect=db_vect, error_vect=error_vect)
    
    db_vect = np.trunc(db_vect*10)/10

    error_vect_total = np.mean(error_vect, axis=-1)
    plt.figure()
    boxplot(x=np.repeat(db_vect, N), y=error_vect_total.flatten(), color='#1f77b4')
    plt.ylabel('Absolute Error [pm]')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')
    plt.savefig(save_file+'.pdf', bbox_inches='tight')

    if split:
        save_file = "_".join((save_file, 'split'))
        group_box_plot(db_vect, error_vect, labels=["$FBG_{}$".format(i+1) for i in range(config.Q)])
        plt.ylabel('Absolute Error [pm]')
        plt.xlabel('SNR [dB]')
        plt.yscale('log')
        plt.savefig(save_file+'.pdf', bbox_inches='tight')




def group_box_plot(x, y, labels, title=None):
    plt.figure()
    M, N, Q = y.shape
    data = [np.stack((np.repeat(x, N), y[:,:,i].flatten()), axis=-1) for i in range(Q)]
    df = concat([DataFrame(data=data[i], columns=['x', 'y']).assign(label=labels[i]) 
                for i in range(Q)], ignore_index=True)
    boxplot(data=df, x='x', y='y', hue='label')
    plt.legend(title=title, loc='upper right', frameon=False)

def compare_finetune_snr(exp_name=None, finetune=None, db_id=None):

    # Train
    if exp_name is not None:
        exp_name = exp_name
    else:
        exp_name = config.exp_name.replace('_finetune', '')
        exp_name = exp_name.replace('auto', '')
        if finetune is not None:
            exp_name = exp_name.replace("_"+finetune, '')
    exp_dir = join(config.base_dir, exp_name)
    pretest_tag = None
    save_file = join(exp_dir, 
                    "_".join(filter(None, (exp_name, 'error_snr', pretest_tag))))
    with np.load(save_file+'.npz') as f:
        db_vect = f['db_vect']
        error_vect = f['error_vect']
        if db_id is not None:
            db_vect = db_vect[db_id]
            error_vect = error_vect[db_id]
    x = db_vect
    y = np.mean(error_vect, axis=-1, keepdims=True)

    tag = None if finetune is not None else config.tag

    # Pre-Finetune
    pretest_tag = 'pretest'
    save_file = join(config.exp_dir, 
                    "_".join(filter(None, (config.exp_name, tag, 'error_snr', pretest_tag))))
    with np.load(save_file+'.npz') as f:
        db_vect = f['db_vect']
        error_vect = f['error_vect']
        if db_id is not None:
            db_vect = db_vect[db_id]
            error_vect = error_vect[db_id]
    y = np.concatenate((y, np.mean(error_vect, axis=-1, keepdims=True)), axis=-1) 

    # Finetune
    pretest_tag = None
    save_file = join(config.exp_dir, 
                    "_".join(filter(None, (config.exp_name, tag, 'error_snr', pretest_tag))))
    with np.load(save_file+'.npz') as f:
        db_vect = f['db_vect']
        error_vect = f['error_vect']
        if db_id is not None:
            db_vect = db_vect[db_id]
            error_vect = error_vect[db_id]
    y = np.concatenate((y, np.mean(error_vect, axis=-1, keepdims=True)), axis=-1) 

    labels = ["Train", "Pre-Finetune", "Finetune"]
    title = 'Model'
    group_box_plot(x, y, labels, title)
    plt.ylabel('Absolute Error [pm]')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')
    save_file = join(config.exp_dir, "_".join(filter(None, (config.exp_name, tag, 'compare'))))
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')

def compare_snr(tags=None , db_id=[4,-1], list_id=None, labels=None, compare=False, compare_label=None, filename=None):

    tags = config.baseline_tags if tags is None else tags
    labels = config.baseline_labels if labels is None else labels
    filename = "_".join(filter(None, (config.exp_name, 'compare'))) if filename is None else filename

    if list_id is not None:
        tags = [tags[id] for id in list_id]
        labels = [labels[id] for id in list_id]

    if compare:
        exp_dir = join(config.base_dir, compare)
        save_file = join(exp_dir, 
                "_".join(filter(None, (compare, config.tag, 'error_snr'))))
        with np.load(save_file+'.npz') as f:
            x = f['db_vect']
            error_vect = f['error_vect']
            error_vect = error_vect[db_id]
            error_vect = np.mean(error_vect, axis=-1, keepdims=True)
            try:
                y = np.concatenate((y, error_vect), axis=-1) 
            except UnboundLocalError:
                y = error_vect
        labels = [compare_label] + labels


    for tag in tags:
        save_file = join(config.exp_dir, 
                     "_".join(filter(None, (config.exp_name, tag, 'error_snr'))))
        with np.load(save_file+'.npz') as f:
            x = f['db_vect']
            error_vect = f['error_vect']
            error_vect = error_vect[db_id]
            error_vect = np.mean(error_vect, axis=-1, keepdims=True)
            try:
                y = np.concatenate((y, error_vect), axis=-1) 
            except UnboundLocalError:
                y = error_vect

    x = x[db_id]
    title = 'Model'
    group_box_plot(x, y, labels, title)
    plt.ylabel('Absolute Error [pm]')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')
    save_file = join(config.base_dir, filename)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, frameon=False)
    plt.savefig(save_file+'.pdf', bbox_inches='tight')

def print_snr(tags=None , list_id=None, labels=None):

    tags = config.baseline_tags if tags is None else tags

    if list_id is not None:
        tags = [tags[id] for id in list_id]

    for tag in tags:
        save_file = join(config.exp_dir, 
                     "_".join(filter(None, (config.exp_name, tag, 'error_snr'))))
        with np.load(save_file+'.npz') as f:
            db_vect = f['db_vect']
            error_vect = f['error_vect']

            plt.figure()
            group_box_plot(db_vect, error_vect, labels=["$FBG_{}$".format(i+1) for i in range(config.Q)])
            
            plt.ylabel('Absolute Error [pm]')
            plt.xlabel('SNR [dB]')
            plt.yscale('log')
            plt.savefig(save_file+'_split.pdf', bbox_inches='tight')

            error_vect = np.mean(error_vect, axis=-1, keepdims=True)
            plt.figure()
            N = error_vect.shape[1]
            boxplot(x=np.repeat(db_vect, N), y=error_vect.flatten(), color='#1f77b4')
            plt.ylabel('Absolute Error [pm]')
            plt.xlabel('SNR [dB]')
            plt.yscale('log')
            plt.savefig(save_file+'.pdf', bbox_inches='tight')