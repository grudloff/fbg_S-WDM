import numpy as np
import matplotlib.pyplot as plt
import warnings
import fbg_swdm.variables as config
from fbg_swdm.variables import constant
import fbg_swdm.simulation as sim
from fbg_swdm.simulation import X, R, normalize, denormalize, get_I, prep_dims
from fbg_swdm.deep_regresors import autoencoder_model, encoder_model
import fbg_swdm.evolutionary as ev
from scipy.signal import sawtooth
from pandas import DataFrame, concat, melt
import seaborn as sns
from seaborn import pairplot, color_palette, displot, boxplot
import os
from os.path import join
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['figure.figsize'] = config.figsize
plt.rcParams['figure.dpi'] = config.dpi
plt.style.context('seaborn-paper')

def latex_float(f):
    float_str = "{0:.0e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def mae(a, b):
    """ Mean absolute values """
    return np.mean(np.abs(a-b))

def plot_datapoint(x, y, i=None, normalized=False, figname=None, dashed_individual=True, plot_y=True, legend="right", **kwargs):
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
    legend : str
        Location of legend. Default is 'right'.
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
    λ = λ/constant.n
    y = y/constant.n
        
    fig, ax = plt.subplots()
    plt.title("Datapoint visualization")    
    ax.plot(λ, x, label="$x$")
    ax.set_xticks(ax.get_xticks()[1::2])

    with color_palette(n_colors=config.Q):
        linefmt = '--' if dashed_individual else '-'
        ax.plot(λ, r,
                linestyle=linefmt,
                label=["$x_"+str(i+1)+"$" for i in range(config.Q)])
        if plot_y:
            for i in range(config.Q):
                ax.stem(y[i, None], np.array([1]), linefmt='--', markerfmt="None",
                    basefmt="None", use_line_collection=False,
                    label="$y_"+str(i+1)+"$")
        plt.ylabel("Reflection spectrum [-]")
        plt.xlabel("Wavelength [nm]")
        if legend == "right":
            bbox_to_anchor = (1.15, 0.5) if plt.rcParams['figure.figsize']==(8,6) else (1.2, 0.5)
            plt.legend(loc='right', bbox_to_anchor=bbox_to_anchor, frameon=False)
        elif legend == "inside":
            plt.legend(loc='upper right', frameon=False)
    
    if figname:
        plt.title("")
        if not isinstance(figname, str):
            figname = join(config.exp_dir, 
                           "_".join(filter(None, (config.exp_name, config.tag, 'datapoint'))))
        fig.savefig(figname+'.pdf', bbox_inches='tight')



def plot_sweep_datapoint(i=0, N=300, **kwargs):
    # simulate sweep and plot n datapoint
    x, y = _gen_sweep(N=N, i=i, **kwargs)
    plot_datapoint(x, y)

def _train_test_dataframe(y_train, y_test, column_names):
    # Create dataframe with column specifying if sample is train or test
    df_train = DataFrame(data=y_train/constant.n, columns=column_names).assign(label='Train')
    df_test = DataFrame(data=y_test/constant.n, columns=column_names).assign(label='Test')
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
        df = DataFrame(data=y_train/constant.n, 
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
            df = DataFrame(data=diff_y_train/constant.n, 
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
        g.fig.text(0.8, 0.7, r"$\overline{{{}}}=$".format(short_label) + latex_float(np.mean(y))+r"${}$".format(unit))
    if figname:
        if not isinstance(figname, str):
            figname = join(config.exp_dir, 
                           "_".join(filter(None, (config.exp_name, config.tag, 'error_dist'))))
        with open(figname+'.npz', 'wb') as f:
            np.savez(f, y=y)
        g.fig.savefig(figname+'.pdf', bbox_inches='tight')

def _gen_sweep_pair(d=0.6*constant.n, N=300, noise=False, invert=False, N_datapoint=None, **kwargs):
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

def _gen_sweep_multi(d=0.6*constant.n, N=300, N_datapoint=None, **kwargs):
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

def predict_plot(model, x, y, norm=True, rec_error=False, noise=None, subplot=False,
                 delta_lambda=False, legend="upper", save=True, **kwargs):
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
        subplot: bool
            Wether to separate sweep and mae plots.
        delta_lambda: bool
            Wether to use difference with respect to lambda_0 in sweep.
        legend: str
            Location of legend. Can be 'upper', 'right' or False.
        save: bool
            Wether to save figure.
    """

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

    tag = predict_plot_partial(y, y_hat, noise, subplot, save, delta_lambda, legend)

    if rec_error:
        if not autoencoder:
            x_hat = X(y_hat)
        if isinstance(model, ev.GeneticAlgo):
            x_hat = X(y_hat, ev.config.λ, ev.config.A, ev.config.Δλ, ev.config.I, 
                      ev.config.Δn_dc, simulation=ev.config.simulation)
        fig = plt.figure()
        mare = np.mean(np.abs(x - x_hat), axis=1)
        plt.plot(mare)
        plt.ylabel("Mean Absolute Reconstruction Error")
        plt.ylim(bottom=0)

        figname = join(config.exp_dir, 
                     "_".join(filter(None, (config.exp_name, config.tag, 'sweep_rec_error',*tag))))
        if save:
            with open(figname+'.npz', 'wb') as f:
                np.savez(f, mare=mare, y=y)
        fig.savefig(figname+'.pdf', bbox_inches='tight')

def predict_plot_partial(y, y_hat, noise=False, subplot=False, save=True, delta_lambda=False, legend="upper"):
    error = np.abs(y - y_hat)
    if delta_lambda:
        y -= config.λ0
        y_hat -= config.λ0
    if isinstance(noise, str):
        noise_tag = "noise"+noise+"db"
    else:
        noise_tag = "noise{:.0e}".format(noise) if noise else None
    pretest_tag = 'pretest' if config.pre_test else None
    tag = (noise_tag, pretest_tag)
    figname = join(config.exp_dir, 
                "_".join(filter(None, (config.exp_name, config.tag, 'sweep_error_dist',*tag)))) 
    plot_dist(error/constant.p, mean=True, figname=figname)

    if subplot:
        fig, (a1, a2) = plt.subplots(2, 1)
        # a1.set_xticks([]) 
        plt.subplots_adjust(hspace=0)
    else:
        fig = plt.figure()
        a1 = plt.gca()
        a2 = a1.twinx()

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    a1.set_prop_cycle(None) # reset color cycle
    a1.plot(y/n, linewidth=2,
            label=["$y_{"+str(i)+"}$" for i in range(1, config.Q+1)])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2.colors)
    a1.set_prop_cycle(None) # reset color cycle
    a1.plot(y_hat/n, linewidth=2, linestyle="-.",
            label=["$\hat{y}_{"+str(i)+"}$" for i in range(1, config.Q+1)]
            )
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
    a1.set_prop_cycle(None) # reset color cycle

    if delta_lambda:
        a1.set_ylabel('$\Delta\lambda_{B}$ [nm]')
    else:
        a1.set_ylabel('$\lambda_{B}$ [nm]')

    a2.plot(np.mean(error, axis=1)/constant.p, ":k", label='MAE')
    a2.set_ylabel('MAE [pm]', labelpad=27 if subplot else 10)

    a2.set_ylim(bottom=0)
    # a2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a1.set_xlabel('Instance [-]')
    a2.set_xlabel('Instance [-]')

    if legend=="right":
        fig.legend(loc='right', bbox_to_anchor=(1.2 if subplot else 1.35, 0.5), frameon=False)
    elif legend=="upper":
        bbox_to_anchor=(0.5, 1.2) if config.Q==2 else (0.5, 1.3)
        fig.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True, ncol=3)
    elif legend==False:
        pass
    else:
        raise ValueError("legend must be 'upper', 'right' or False")

    figname = join(config.exp_dir, 
            "_".join(filter(None, (config.exp_name, config.tag, 'sweep',*tag))))
    if save:
        with open(figname+'.npz', 'wb') as f:
            np.savez(f, y=y, y_hat=y_hat)
    fig.savefig(figname+'.pdf', bbox_inches='tight')
    print("Figure saved at: " + figname+'.pdf')
    return tag

def plot_sweep(model, **kwargs):
    """ Simulate sweep and call predict_plot.

        Parameters
        ----------
        model : object with predict method
            Predictor.  
    """
    noise = kwargs.pop("noise", False)
    # TODO: This should be np.max(np.cumprod(config.A)*config.I) for serial
    # Scale noise to be in relation to greatest peak value
    scaled_noise = noise*np.max(config.A*config.I)
    x, y = _gen_sweep(noise=scaled_noise, **kwargs)
    predict_plot(model, x, y, noise=noise, **kwargs)

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
            a1.stem(y[i, j, None]/constant.n, np.max(x[i], keepdims=True), linefmt='-', markerfmt="None",
                basefmt="None", use_line_collection=False,
                label="$y_"+str(j+1)+"$")
        if evolutionary or autoencoder:
            a1.plot(config.λ/n, x_hat[i], label="$\hat{x}$")
        for j in range(config.Q):
            a1.stem(y_hat[i, j, None]/constant.n, np.max(x[i], keepdims=True), linefmt='--', markerfmt="None",
                basefmt="None", use_line_collection=False,
                label="$\hat{y}_"+str(j+1)+"$")
        if not evolutionary:
            a2 = a1.twinx()
            a2._get_lines.prop_cycler = a1._get_lines.prop_cycler # set same color cycler
            a2.plot(config.λ/n, latent[i].T, label=[r"$\tilde{y}_{"+str(i+1)+"}$" for i in range(config.Q)])
            a2.set_ylabel(r'$\tilde{y}$')

        lines, labels = a1.get_legend_handles_labels()
        lines2, labels2 = a2.get_legend_handles_labels()
        a2.legend(lines + lines2, labels + labels2, loc=0)


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
    # TODO: This should be np.max(np.cumprod(config.A)*config.I) for serial
    # Won't change this now because would change all error_snr plots
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
        group_boxplot(db_vect, error_vect, labels=["$FBG_{}$".format(i+1) for i in range(config.Q)])
        plt.ylabel('Absolute Error [pm]')
        plt.xlabel('SNR [dB]')
        plt.yscale('log')
        plt.savefig(save_file+'.pdf', bbox_inches='tight')


def group_boxplot(x, y, labels, title=None):
    plt.figure()
    M, N, Q = y.shape
    data = [np.stack((np.repeat(x, N).astype(int), y[:,:,i].flatten()), axis=-1) for i in range(Q)]
    df = concat([DataFrame(data=data[i], columns=['x', 'y']).assign(label=labels[i]) 
                for i in range(Q)], ignore_index=True)
    df["x"] = df['x'].astype(int)
    boxplot(data=df, x='x', y='y', hue='label')
    plt.legend(title=title, loc='upper right', frameon=False)

def group_lineplot(x, y, labels, title=None):
    plt.figure()
    M, N, Q = y.shape
    data = [np.stack((np.repeat(x, N).astype(int), y[:,:,i].flatten()), axis=-1) for i in range(Q)]
    df = concat([DataFrame(data=data[i], columns=['x', 'y']).assign(label=labels[i]) 
                for i in range(Q)], ignore_index=True)
    df["x"] = df['x'].astype(int)
    sns.lineplot(data=df, x="x", y="y", hue="label", estimator=np.median,
                  errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)))
    plt.legend(title=title, loc='upper right', frameon=False)

def load_sweep_data(save_file):
    with np.load(save_file+'.npz') as f:
        y = f['y']
        y_hat = f['y_hat']
    return np.mean(np.abs(y-y_hat), axis=-1)/config.p

def compare_finetune(pretrain_exp_name=None, noise_compare=False):
    """
    Compare the performance of a model before and after finetuning, and compare it to a pretrained model.
    
    Parameters:
    - pretrain_exp_name (str, optional): the name of the experiment for the pretrained model. If not provided, 
      it is set to the `exp_name` attribute of the `config` object, with certain substrings removed.
    """

 
    if pretrain_exp_name is None:
        pretrain_exp_name = config.exp_name.replace('_finetune', '')
        pretrain_exp_name = pretrain_exp_name.replace('auto', '')
        # if finetune is not None:
        #     pretrain_exp_name = pretrain_exp_name.replace("_"+finetune, '')

    load_file_list = []
    exp_dir = join(config.base_dir, pretrain_exp_name)
    pretest_tag = None
    # Train
    load_file = join(exp_dir, 
                    "_".join(filter(None, (pretrain_exp_name, 'sweep', pretest_tag))))
    load_file_list.append(load_file)

    exp_dir = "_".join(filter(None, (config.exp_dir, config.tag)))

    # Pre-Finetune
    pretest_tag = 'pretest'
    load_file = join(exp_dir, 
                    "_".join(filter(None, (config.exp_name, config.tag, 'sweep', pretest_tag))))
    load_file_list.append(load_file)

    # Finetune
    pretest_tag = None
    load_file = join(exp_dir, 
                    "_".join(filter(None, (config.exp_name, config.tag, 'sweep', pretest_tag))))
    load_file_list.append(load_file)

    vector_list = [load_sweep_data(load_file) for load_file in load_file_list]

    labels = ["Source Domain", "Target Domain\n (Pretrained)", "Target Domain\n  (Finetuned)"]

    noise_tag = None
    if noise_compare:
        pretest_tag = None
        exp_dir = "_".join((exp_dir, 'noise'))
        noise_tag = "noise"
        load_file = join(exp_dir, 
                        "_".join(filter(None, (config.exp_name, config.tag, "noise", 'sweep', noise_compare, pretest_tag))))
        load_file_list.append(load_file)
        vector_list.append(load_sweep_data(load_file))
        labels.append("Target Domain\n with noise\n(Finetuned)")

    # Create the dataframe
    df_list = [DataFrame({label: vector}) for label, vector in zip(labels, vector_list)]
    df = concat(df_list, axis=1)
    boxplot(data=df)
    # plt.legend(title='Model', loc='upper right', frameon=False)
    plt.ylabel('MAE [pm]')
    # plt.xlabel('SNR [dB]')
    plt.yscale('log')
    save_file = join(exp_dir, "_".join(filter(None, (config.exp_name, config.tag, noise_tag, 'compare_finetune'))))
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')


def compare_finetune_snr(exp_name=None, finetune=None, db_id=None, lineplot=False):

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
    if lineplot:
        group_lineplot(x, y, labels, title)
    else:
        group_boxplot(x, y, labels, title)
    plt.ylabel('MAE [pm]')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')
    # plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, frameon=False)
    # put fancy legend on top right corner
    plt.legend(loc='upper right', frameon=False)
    lineplot_tag = 'lineplot' if lineplot else None
    save_file = join(config.exp_dir, "_".join(filter(None, (config.exp_name, tag,
                                                            lineplot_tag ,'compare'))))
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')

def compare_snr(tags=None , db_id=[4,-1], list_id=None, labels=None, compare=False, compare_label=None, filename=None, legend="right"):

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

    x = x[db_id].astype(int)
    title = 'Model'
    group_boxplot(x, y, labels, title)
    plt.ylabel('MAE [pm]')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')
    save_file = join(config.base_dir, filename)
    if legend=="right":
        plt.legend(loc='right', bbox_to_anchor=(1.2 , 0.5), frameon=False)	
    else:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=3)
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')

def print_snr(tags=None , list_id=None, labels=None):

    tags = config.baseline_tags if tags is None else tags

    if list_id is not None:
        tags = [tags[id] for id in list_id]

    for tag in tags:
        save_file = join(config.exp_dir, 
                     "_".join(filter(None, (config.exp_name, tag, 'error_snr'))))
        with np.load(save_file+'.npz') as f:
            db_vect = f['db_vect'].astype(int)
            error_vect = f['error_vect']

            plt.figure()
            group_boxplot(db_vect, error_vect, labels=["$FBG_{}$".format(i+1) for i in range(config.Q)])
            
            plt.ylabel('Absolute Error [pm]')
            plt.xlabel('SNR [dB]')
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, frameon=False)
            plt.savefig(save_file+'_split.pdf', bbox_inches='tight')

            error_vect = np.mean(error_vect, axis=-1, keepdims=True)
            plt.figure()
            N = error_vect.shape[1]
            boxplot(x=np.repeat(db_vect, N), y=error_vect.flatten(), color='#1f77b4')
            plt.ylabel('Absolute Error [pm]')
            plt.xlabel('SNR [dB]')
            plt.yscale('log')
            plt.savefig(save_file+'.pdf', bbox_inches='tight')

def print_sweep(noise=False, subplot=False, delta_lambda=False, legend="upper"):

    noise_tag = "noise{:.0e}".format(noise) if noise else None
    pretest_tag = "pretest" if config.pre_test else ""
    save_file = join(config.exp_dir, 
                    "_".join(filter(None, (config.exp_name, config.tag, 'sweep',
                                           pretest_tag, noise_tag))))
    print("Loading from file at: " + save_file + ".npz" )
    with np.load(save_file+'.npz') as f:
        y = f["y"]
        y_hat = f["y_hat"]

        predict_plot_partial(y, y_hat, noise, subplot, save=False,
                             delta_lambda=delta_lambda, legend=legend)

def reprint_sweep(noise=False, subplot=False, delta_lambda=False):
    noise_tag = "noise{:.0e}".format(noise) if noise else None
    load_file = join(config.exp_dir, "checkpoint", "last.ckpt")
    if "autoencoder" in config.exp_dir:
        model = autoencoder_model.load_from_checkpoint(load_file, strict=False)
    else:
        model = encoder_model.load_from_checkpoint(load_file, strict=False)

    predict_plot(model, noise=noise, subplot=subplot, delta_lambda=delta_lambda)


def load_snr_data(load_file, db_id=4):
    with np.load(load_file+'.npz') as f:
        error_vect = f['error_vect'][db_id]
    error_vect = np.mean(error_vect, axis=-1)
    return error_vect


def compare_simulated_finetune(pretrain_exp_name=None, legend="right"):

    finetune_options = ["attenuation", "spectral",  None]
    finetune_labels = ["Attenuation", "Spectral Profile", "Attenuation and\n Spectral Profile "]

    topology = "serial"
    exp_name_no_topology = config.exp_name.replace('_'+topology, '')
 
    if pretrain_exp_name is None:
        pretrain_exp_name = config.exp_name.replace('_'+topology, '')
        pretrain_exp_name = config.exp_name.replace('_finetune', '')
        pretrain_exp_name = pretrain_exp_name.replace('auto', '')
        # if finetune is not None:
        #     pretrain_exp_name = pretrain_exp_name.replace("_"+finetune, '')
    
    df_partial_list = []
    for finetune_option, finetune_label in zip(finetune_options, finetune_labels):
        tag = finetune_option

        load_file_list = []
        exp_dir = join(config.base_dir, pretrain_exp_name)
        pretest_tag = None
        # Train
        load_file = join(exp_dir, 
                        "_".join(filter(None, (pretrain_exp_name, 'error_snr', pretest_tag))))
        load_file_list.append(load_file)

        exp_dir = join(config.base_dir, exp_name_no_topology)
        exp_dir = "_".join(filter(None, (exp_dir, tag, topology)))

        # Pre-Finetune
        pretest_tag = 'pretest'
        load_file = join(exp_dir, 
                        "_".join(filter(None, (exp_name_no_topology, tag, topology, 'error_snr', pretest_tag))))
        load_file_list.append(load_file)

        # Finetune
        pretest_tag = None
        load_file = join(exp_dir, 
                        "_".join(filter(None, (exp_name_no_topology, tag, topology, 'error_snr', pretest_tag))))
        load_file_list.append(load_file)

        vector_list = [load_snr_data(load_file) for load_file in load_file_list]

        labels = ["Source Domain", "Target Domain\n (Pretrained)", "Target Domain\n  (Finetuned)"]

        # Create the dataframe
        df_list = [DataFrame({"absolute_error" :vector}).assign(finetune=finetune_label, label=label) 
                    for label, vector in zip(labels, vector_list)]
        # df_partial = concat(df_list, axis=1)
        for df_instance in df_list:
            df_partial_list.append(df_instance)
    df = concat(df_partial_list)
    boxplot(data=df, y='absolute_error', x="finetune", hue="label")
    # plt.legend(title='Model', loc='upper right', frameon=False)
    plt.ylabel('MAE [pm]')
    plt.xlabel('Simulation Innacuracy Scenario')
    plt.yscale('log')
    if legend == "right":
        plt.legend(bbox_to_anchor=(1.45, 0.5), title=None, loc='center right', frameon=False)
    elif legend == "upper":
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=4)
    save_file = join(exp_dir, "_".join(filter(None, (config.exp_name, config.tag, 'compare_finetune'))))
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')

def compare_simulated_finetune_baseline(pretrain_exp_name=None, baseline_exp_name = "baseline",
                                        reg_tag = "ls_svm", ea_tag = 'swap_diff',
                                        finetune_option_dict=None, x_axis_label=True, 
                                        noise=False, legend="right"):

    reg_label = config.reg_labels[config.reg_tags.index(reg_tag)]
    ea_label = config.ea_labels[config.ea_tags.index(ea_tag)]

    if finetune_option_dict is None:
        finetune_options = [None, "attenuation", "spectral",  "multi"]
        finetune_labels = ["Reference","  Attenuation", "Profile", "Attenuation \n & Profile "]
    else:
        finetune_labels ,finetune_options = zip(*finetune_option_dict.items())
    
    if config.Q==2: 
        topology = "serial"
        exp_name_no_topology = config.exp_name.replace('_'+topology, '')
    else:
        topology=None
        exp_name_no_topology = config.exp_name
 
    if pretrain_exp_name is None:
        pretrain_exp_name = config.exp_name
        if config.Q==2:
            pretrain_exp_name = pretrain_exp_name.replace('_'+topology, '')
        pretrain_exp_name = pretrain_exp_name.replace('_finetune', '') # TODO: this was wrong?
        pretrain_exp_name = pretrain_exp_name.replace('auto', '')
        # if finetune is not None:
        #     pretrain_exp_name = pretrain_exp_name.replace("_"+finetune, '')
    
    df_partial_list = []
    for finetune_option, finetune_label in zip(finetune_options, finetune_labels):
        tag = finetune_option

        vector_list = []
        exp_dir = join(config.base_dir, "_".join(filter(None, (baseline_exp_name, tag))))
        # regression
        try:
            noise_tag = "noise{:.0e}".format(noise) if noise else None
            load_file = join(exp_dir, 
                "_".join(filter(None, (baseline_exp_name, tag, reg_tag, 'sweep', noise_tag))))
            vector_list.append(load_sweep_data(load_file))
        except:
            load_file = join(exp_dir, 
                            "_".join(filter(None, (baseline_exp_name, tag, reg_tag, 'error_snr'))))
            vector_list.append(load_snr_data(load_file))

        # evolutionary algorithm
        load_file = join(exp_dir, 
                        "_".join(filter(None, (baseline_exp_name, tag, ea_tag, 'error_snr'))))
        vector_list.append(load_snr_data(load_file))

        if finetune_label == finetune_labels[0]:
            exp_name = pretrain_exp_name
        else:
            exp_name = exp_name_no_topology

        # if config.Q==2:
        #     tag = None if tag == "multi" else finetune_option
        #     exp_dir = join(config.base_dir, exp_name)
        #     exp_dir = "_".join(filter(None, (exp_dir, tag, topology)))
        # else:
        #     exp_dir = join(config.base_dir, exp_name)
        #     exp_dir = "_".join(filter(None, (exp_dir, topology)))

        exp_dir_tag = None if tag == "multi" or config.Q!=2 else tag
        exp_dir = join(config.base_dir, exp_name)
        exp_dir = "_".join(filter(None, (exp_dir, exp_dir_tag, topology)))

        # Pretrained and/or Finetune
        exp_name_tag = None if tag == "multi" and config.Q==2 else tag
        # load_file = join(exp_dir, 
        #                 "_".join(filter(None, (exp_name, exp_name_tag, topology, 'error_snr'))))
        # vector_list.append(load_snr_data(load_file))

        if finetune_label == finetune_labels[0]:
            load_file = join(exp_dir, 
                "_".join(filter(None, (exp_name, exp_name_tag, topology, 'error_snr'))))
            vector_list.append(load_snr_data(load_file))
            labels = [reg_label, ea_label, "Pretrained\n CNN-AE"]
        else:
            load_file = join(exp_dir, 
                "_".join(filter(None, (exp_name, exp_name_tag, topology, 'error_snr', 'pretest'))))
            vector_list.append(load_snr_data(load_file))
            load_file = join(exp_dir, 
                "_".join(filter(None, (exp_name, exp_name_tag, topology, 'error_snr'))))
            vector_list.append(load_snr_data(load_file))
            labels = [reg_label, ea_label, "Pretrained\n CNN-AE", "Finetuned\n CNN-AE"]

        # Create the dataframe
        df_list = [DataFrame({"absolute_error" :vector}).assign(finetune=finetune_label, label=label) 
                    for label, vector in zip(labels, vector_list)]
        # df_partial = concat(df_list, axis=1)
        for df_instance in df_list:
            df_partial_list.append(df_instance)
    df = concat(df_partial_list)
    boxplot(data=df, y='absolute_error', x="finetune", hue="label")
    # plt.legend(title='Model', loc='upper right', frameon=False)
    plt.ylabel('MAE [pm]')
    if x_axis_label:
        plt.xlabel('Simulation Innacuracy Scenario')
    else:
        plt.xlabel("")
    plt.yscale('log')
    if legend=="right":
        plt.legend(bbox_to_anchor=(1.45, 0.5), title=None, loc='center right', frameon=False)
    elif legend=="upper":
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=4)
    save_file = join(exp_dir, "_".join(filter(None, (config.exp_name, config.tag, 'compare_finetune_baseline'))))
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')


def compare_simulated_finetune_new(pretrain_exp_name=None, legend="right"):

    finetune_options = ["attenuation", "spectral", "multi"]
    finetune_labels = ["Attenuation", "Profile", "Attenuation \n & Profile "]
 
    if pretrain_exp_name is None:
        pretrain_exp_name = config.exp_name.replace('_finetune', '')
        pretrain_exp_name = pretrain_exp_name.replace('auto', '')
        # if finetune is not None:
        #     pretrain_exp_name = pretrain_exp_name.replace("_"+finetune, '')
    
    df_partial_list = []
    for finetune_option, finetune_label in zip(finetune_options, finetune_labels):
        tag = finetune_option

        load_file_list = []
        exp_dir = join(config.base_dir, pretrain_exp_name)
        pretest_tag = None
        # Train
        load_file = join(exp_dir, 
                        "_".join(filter(None, (pretrain_exp_name, 'error_snr', pretest_tag))))
        load_file_list.append(load_file)

        exp_dir = join(config.base_dir, config.exp_name)

        # Pre-Finetune
        pretest_tag = 'pretest'
        load_file = join(exp_dir, 
                        "_".join(filter(None, (config.exp_name, tag, 'error_snr', pretest_tag))))
        load_file_list.append(load_file)

        # Finetune
        pretest_tag = None
        load_file = join(exp_dir, 
                        "_".join(filter(None, (config.exp_name, tag, 'error_snr', pretest_tag))))
        load_file_list.append(load_file)

        vector_list = [load_snr_data(load_file) for load_file in load_file_list]

        labels = ["Source Domain", "Target Domain\n (Pretrained)", "Target Domain\n  (Finetuned)"]

        # Create the dataframe
        df_list = [DataFrame({"absolute_error" :vector}).assign(finetune=finetune_label, label=label) 
                    for label, vector in zip(labels, vector_list)]
        # df_partial = concat(df_list, axis=1)
        for df_instance in df_list:
            df_partial_list.append(df_instance)
    df = concat(df_partial_list)
    boxplot(data=df, y='absolute_error', x="finetune", hue="label")
    # plt.legend(title='Model', loc='upper right', frameon=False)
    plt.ylabel('Absolute Error [pm]')
    plt.xlabel('Simulation Innacuracy Scenario')
    plt.yscale('log')
    if legend == "right":
        plt.legend(bbox_to_anchor=(1.6, 0.5), title=None, loc='center right', frameon=False)
    elif legend == "upper":
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=3)
    save_file = join(exp_dir, "_".join(filter(None, (config.exp_name, config.tag, 'compare_finetune'))))
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')


def compare_simulation_sweep(pretrain_exp_name=None, baseline_exp_name = "baseline",
                             reg_tag = "ls_svm", ea_tag = 'swap_diff',
                             simulation_noise_tag=None, legend="right"):

    reg_label = config.reg_labels[config.reg_tags.index(reg_tag)]
    ea_label = config.ea_labels[config.ea_tags.index(ea_tag)]

    finetune_options = [None, "experimental"]
    finetune_labels = ['Simulation',"Experimental\n"]
 
    if pretrain_exp_name is None:
        pretrain_exp_name = config.exp_name
        # if config.Q==2:
        #     pretrain_exp_name = pretrain_exp_name.replace('_'+topology, '')
        pretrain_exp_name = pretrain_exp_name.replace('_finetune', '')
        pretrain_exp_name = pretrain_exp_name.replace('auto', '')
        # if finetune is not None:
        #     pretrain_exp_name = pretrain_exp_name.replace("_"+finetune, '')
    
    df_partial_list = []
    for finetune_option, finetune_label in zip(finetune_options, finetune_labels):
        tag = finetune_option
        noise_tag = simulation_noise_tag if finetune_label == 'Simulation' else None

        load_file_list = []
        exp_dir = join(config.base_dir, "_".join(filter(None, (baseline_exp_name, tag))))
        # regression
        load_file = join(exp_dir, 
                        "_".join(filter(None, (baseline_exp_name, tag, reg_tag, 'sweep', noise_tag))))
        load_file_list.append(load_file)

        # evolutionary algorithm
        load_file = join(exp_dir, 
                        "_".join(filter(None, (baseline_exp_name, tag, ea_tag, 'sweep', noise_tag))))
        load_file_list.append(load_file)

        # Pretrained or Finetune
        if finetune_label == "Simulation":
            exp_dir = join(config.base_dir, pretrain_exp_name)
            load_file = join(exp_dir, 
                            "_".join(filter(None, (pretrain_exp_name, 'sweep', noise_tag))))
            load_file_list.append(load_file)
            labels = [reg_label, ea_label, "Pretrained\n CNN-AE"]
        else:
            exp_dir = join(config.base_dir, 
                           "_".join(filter(None, (config.exp_name, config.tag))))
            load_file = join(exp_dir, 
                             "_".join(filter(None, (config.exp_name, config.tag, 'sweep', "pretest"))))
            load_file_list.append(load_file)
            load_file = join(exp_dir, 
                             "_".join(filter(None, (config.exp_name, config.tag, 'sweep'))))
            load_file_list.append(load_file)
            labels = [reg_label, ea_label, "Pretrained\n CNN-AE", "Finetuned\n CNN-AE"]
        # load_file_list.append(load_file)

        vector_list = [load_sweep_data(load_file) for load_file in load_file_list]

        # Create the dataframe
        df_list = [DataFrame({"absolute_error" :vector}).assign(finetune=finetune_label, label=label) 
                    for label, vector in zip(labels, vector_list)]
        # df_partial = concat(df_list, axis=1)
        for df_instance in df_list:
            df_partial_list.append(df_instance)
    df = concat(df_partial_list)
    boxplot(data=df, y='absolute_error', x="finetune", hue="label")
    # plt.legend(title='Model', loc='upper right', frameon=False)
    plt.ylabel('MAE [pm]')
    plt.xlabel(None)
    plt.yscale('log')
    if legend=="right":
        plt.legend(bbox_to_anchor=(1.45, 0.5), title=None, loc='center right', frameon=False)
    elif legend=="upper":
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=4)
    save_file = join(exp_dir, "_".join(filter(None, (config.exp_name, config.tag, 'compare_simulation_sweep'))))
    plt.savefig(save_file+'.pdf', bbox_inches='tight')
    print("Saved at:", save_file+'.pdf')