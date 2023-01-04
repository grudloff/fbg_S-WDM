import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks, peak_widths
import copy

import fbg_swdm.variables as config

# ---------------------------------------------------------------------------- #
#                               General Utilities                              #
# ---------------------------------------------------------------------------- #

def dbm2mag(x):
    """dBm to magnitude transformation """
    return 10.0**(x/10.0)

def ma_interp(a):
    """Interpolates the masked values in a masked array.
    
    Args:
    a (ndarray): 1D or 2D masked array to interpolate.
    """
    if len(a.shape)>1:
        return np.array([ma_interp(alpha) for alpha in a])
    else:
        lin = np.arange(a.shape[0])
        return np.interp(lin, lin[~a.mask], a[~a.mask])

def wav2points(gamma, delta, proportion=1.1):
    """Convert a wavelength plus a proportion to the bandwidth to a point value on the x-axis.
    
    Parameters:
        gamma (float): Wavelength in nanometers.
        delta (float): Bandwidth in nanometers.
    """
    gamma_relative = gamma-config.λ0+config.Δ
    gamma_relative += delta*proportion
    deltaN = int(gamma_relative*config.N/(2*config.Δ))
    return deltaN

def get_mask_from_wavelength(x, y, proportion=1.1):
    """Mask x according to the y positions and their corresponding bandwidths.
    
    Parameters:
        x (ndarray): A 2D NumPy array.
        y (ndarray): A 2D NumPy array.
        
    Returns:
        ndarray: A 2D masked array
    """
    y = y*config.n # to [nm]
    mask = np.zeros_like(x, dtype='bool')
    for i in np.arange(len(x)):
        for gamma, delta in zip(y[i], config.Δλ):
            start = wav2points(gamma, -delta, proportion)
            stop = wav2points(gamma, delta, proportion)
            mask[i, start:stop] = True
    return np.ma.masked_array(x, mask)


# ---------------------------------------------------------------------------- #
#                                 Plane Fitting                                #
# ---------------------------------------------------------------------------- #

# ------------------------ Profile & intercept Fitting ----------------------- #

class transmision_profile():
    def __init__(self, L, N):
        """
        Initializes the transmission profile object with the given dimensions.

        Args:
        L (int): Number of samples in the data set.
        N (int): Number of spectral points in the data set.
        """
        self.intercept = np.zeros(L) # per sample intercept
        self.profile = np.empty(N) # per spectral point relative value

    def fit(self, x, n=10):
        """
        Fits the transmission profile to the given data set.

        Args:
            x (ndarray): 2D array of data to fit the profile to, where each row is
            an individual data set.
            n (int, optional): Number of iterations to run the fitting process for.
            Defaults to 10.
        """
        for i in range(n):
            self.profile = np.mean(ma_interp(x) - self.intercept[:, None], axis=0)
            self.intercept = np.ma.mean(self.profile - x, axis=-1)
    
    def predict(self, x=None):
        """
        Transmission profile prediction.

        Args:
            x (ndarray): 2D array of data to make predictions for, where each row is an individual data set.
        """
        return self.profile[None, :] - self.intercept[:, None]

    def trim(self, idx):
        """
        Trims the transmission profile to only include the given indices.

        Args:
            idx (ndarray): 1D array of indices to include in the trimmed profile.

        Returns:
            new: Trimmed transmission profile object.
        """
        new = copy.copy(self)
        new.intercept = self.intercept[idx]
        return new

# ------------------------------- Line Fitting ------------------------------- #

def fit_line(x):
    """Fits a linear regression model to the input data.

    Args:
        x (ndarray): 1D array of data to fit the model to.

    Returns:
        model: Trained linear regression model.
    """
    model = LinearRegression()
    y = np.arange(x.shape[-1])
    if isinstance(x, np.ma.masked_array):
        mask = np.logical_not(x.mask)
        x = np.array(x)[mask]
        y = y[mask]
    model.fit(y.reshape(-1, 1), x.reshape(-1, 1))
    return model

def fit_multi_lines(x):
    """Fits a linear regression model to each subarray in the input data.

    Args:
        x (ndarray): 2D array of data to fit the model to, where each row is an individual data set.

    Returns:
        model_vect: List of trained linear regression models, one for each data set.
    """
    model_vect = [fit_line(v) for v in x]
    return model_vect

def predict(x, model):
    """Predicts the values of x using the given model(s).

    Args:
        x (ndarray): 1D or 2D array of data to make predictions for.
        model: Linear regression model or list of models.

    Returns:
        x_pred: Predicted values of x.
    """
    if isinstance(model, transmision_profile):
        x_pred = model.predict(x)
    else:
        if len(x.shape)==1:
            x = x.reshape(1, -1)
            model = [model]
        assert len(x)==len(model)
        y = np.arange(x.shape[-1])
        x_pred = np.array([m.predict(y.reshape(-1, 1)) for m in model])
        x_pred = x_pred.squeeze()
    return x_pred


def find_transmision_profile(x, cuttout_factor=0.2, iter=10):
    """Determines the transmission profile of the input data.

    Args:
        x (ndarray): 2D array of data to determine the transmission profile for, where each row is an individual data set.
        cuttout_factor (float, optional): Factor to determine how much of the data is masked. Defaults to 0.2.
        iter (int, optional): Number of iterations to run the transmission profile determination for. Defaults to 10.

    Returns:
        model_vect: List of trained linear regression models, one for each data set.
        M: Maximum difference between the masked input data and the predicted data.
    """
    model_vect = fit_multi_lines(x)
    x = np.ma.masked_array(x, np.zeros_like(x, dtype='bool')) # turn into masked array
    q = int(x.shape[-1]*cuttout_factor)
    M=[]
    for i, v in enumerate(x):
        for j in range(iter):
            v.mask[:] = False # remove mask on every iteration
            v_pred = predict(v, model_vect[i])
            diff = np.abs(v-v_pred)
            idx = np.argsort(diff)[-q:]
            v.mask[idx] = True
            model_vect[i] = fit_line(v)
        v_pred = predict(v, model_vect[i])
        M.append(np.max(np.abs(v-v_pred)))
    M = np.max(M)

    return model_vect, M

# ---------------------------------------------------------------------------- #
#                                 Visualization                                #
# ---------------------------------------------------------------------------- #

def plot_instance(x, x_pred, x_masked, i=0):
    """Plots a single instance of raw, predicted, and masked data.

    Parameters:
    x (numpy array): Raw data to plot
    x_pred (numpy array): Predicted data to plot
    x_masked (numpy array): Masked data to plot
    i (int, optional): Index of instance to plot (default is 0)

    """
    plt.figure()
    plt.plot(x[i], label='raw')
    plt.plot(x_pred[i], label='prediction')
    plt.plot(x_masked[i], label='masked')
    plt.legend()

def plot3d(x, ax=None, **kwargs):
    """Plots 3D data.

    Parameters:
    x (numpy array): Data to plot in 3D
    ax (matplotlib axis, optional): Axis to plot on (default is None)
    kwargs: Additional keyword arguments to pass to matplotlib's plot_surface function

    Returns:
    matplotlib axis: Axis that the data was plotted on
    """
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    X = np.arange(x.shape[1])
    Y = np.arange(x.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = x
    ax.plot_surface(X, Y, Z, **kwargs)
    return ax

def plot(x, model):
    """Plots raw and predicted 3D data.

    Parameters:
    x (numpy array): Raw data to plot
    model: Model to use for prediction
    """
    ax = plot3d(x, cmap="viridis", alpha=0.5, label='raw')
    x_pred = predict(x, model)
    plot3d(x_pred, ax, alpha=0.5, label='predicted')
    #plt.legend() #TODO: This should work on matplotlib 3.6.2

def plot_masked(x, model, M):
    """Plots raw and predicted 3D data with mask applied.

    Parameters:
    x (numpy array): Raw data to plot
    model: Model to use for prediction
    M (float): Threshold for masking data

    Returns:
    None: Function only plots the data
    """
    # plot surface and plane
    x_pred = predict(x, model)
    mask = np.abs(x-x_pred) > M
    x_masked = np.ma.masked_where(mask, x)
    ax = plot3d(x_masked, cmap="viridis", alpha=0.5, label= "masked")
    x_pred = predict(x, model)
    plot3d(x_pred, ax, alpha=0.5, label='predicted')
    #plt.legend() #TODO: This should work on matplotlib 3.6.2

def plot_image(x, model, aspect=1):
    """Plots difference between raw and predicted data as an image.

    Parameters:
    x (numpy array): Raw data to plot
    model: Model to use for prediction
    aspect (float, optional): Aspect ratio for image plot (default is 1)
    """
    x_pred = predict(x, model)
    diff = x-x_pred
    plt.figure()
    plt.imshow(diff, cmap='viridis', aspect=aspect)
    plt.colorbar()

def plot_masked_image(x, model, M, aspect=1):
    """Plots difference between raw and predicted data as an image with mask applied
    Parameters:
    x (numpy array): Raw data to plot
    model: Model to use for prediction
    M (float): Threshold for masking data
    aspect (float, optional): Aspect ratio for image plot (default is 1)
    """
    x_pred = predict(x, model)
    diff = x-x_pred
    mask = np.abs(diff) > M
    diff[mask] = 0
    plot_image(diff, model, aspect)

def plot_hist(x, model, M):
    """Plots histogram of difference between raw and predicted data with mask applied.

    Parameters:
    x (numpy array): Raw data to plot.
    model: Model to use for prediction..
    M (float): Threshold for masking data.
    """
    plt.figure()
    x_pred = predict(x, model)
    diff = x-x_pred
    mask = np.abs(diff) > M
    diff_masked = np.ma.masked_where(mask, diff)
    plt.hist(diff_masked.flatten())

# ---------------------------------------------------------------------------- #
#                                 Peak Tracking                                #
# ---------------------------------------------------------------------------- #

const_val = config.N
def calculatePeaks(x):
    """Calculate peaks in a given array
    Parameters:
    x (numpy array): Array to find peaks in

    Returns:
    numpy array: Array of peak indices
    """
    shape = x.shape
    if len(shape)>1:
        peak_ind = [calculatePeaks(x[i]) for i in range(shape[0])]
        return np.stack(peak_ind).squeeze()
    else:
        peaks, properties = find_peaks(x, height=0.2, distance=150, width=1)
        # Sort
        indx = np.argsort(properties['peak_heights'])[::-1]
        peaks = peaks[indx]
        # Extend to length 2
        if len(peaks)>2:
            peaks = peaks[:2]
        else:
            peaks = np.pad(peaks, pad_width=(0, 2-len(peaks)), constant_values=const_val) 
        peaks = peaks[::-1]
        return peaks


def refine_with_centroid(x, peak_ind, λ, rel_height=0.5):
    """Refine peak prediction using the centroid.

    Parameters:
    x (numpy array): Array to find peaks in
    peak_ind (numpy array): Array of peak indices
    λ (numpy array): Array of wavelengths
    rel_height (float, optional): Relative height for peak width calculation (default is 0.5)

    Returns:
    numpy array: Refined peak indices
    """
    shape = x.shape
    mask = np.zeros((*shape, config.Q))
    M = len(peak_ind)
    for i in range(M):
        for j in range(config.Q):
            center = peak_ind[i,j]
            if center != const_val:
                start, stop = peak_widths(x[i], [center], rel_height=rel_height)[2:]
                start = int(start)
                stop = int(stop)
                mask[i, start:stop, j] = 1
    mask = x[..., None]*mask
    mask = mask/np.linalg.norm(mask, ord=1, axis=1, keepdims=True)
    y_hat = np.sum(λ[None, :, None]*mask, axis=1)
    return y_hat

# ---------------------------------------------------------------------------- #
#                          Wavelength Twin Regression                          #
# ---------------------------------------------------------------------------- #

def plot_fit_w(model, X, y, label_a, label_b, suffix='', plot_error=False):
    """Plots data and model fit, and optionally plots error between data and
        prediction.

    Parameters:
    model: Model to use for prediction
    X (numpy array): Data to use as input for model
    y (numpy array): Data to compare to model prediction
    label_a (str): Label for X data
    label_b (str): Label for y data
    suffix (str): Suffix to append to label_a and label_b
    plot_error (bool, optional): Whether to plot error between y and model prediction (default is False)
    """
    y_pred = model.predict(X)
    fig, ax1 = plt.subplots()
    c = np.arange(len(X))
    ax1.scatter(X, y, c = c)
    ax1.plot(X, y_pred, color = 'blue')
    plt.title(label_a+suffix+' vs '+label_b+suffix)
    plt.xlabel(label_a+suffix)
    ax1.set_ylabel(label_b+suffix)

    if plot_error:
        ax2 = ax1.twinx()
        ax2.plot(X, y_pred-y, color = 'red')
        ax2.set_ylabel('error')

    print("MAE:", np.mean(np.abs(y_pred-y)))
    print("std:", np.std(y_pred-y))

def fit_linear_regression(df, pairs, **kwargs):
    """Fits a linear regression model for given pairs of data in a dataframe.

    Parameters:
    df (pandas dataframe): Dataframe containing data to use for model fitting
    pairs (list of tuple): List of tuples containing columns in df to use for model fitting
    kwargs: Additional keyword arguments to pass to plot_fit_w function

    Returns:
    numpy array: Array of model coefficients and intercepts
    """
    coefs =[]
    for i in range(len(pairs)):
        x = np.array(df[pairs[i][0]].values).reshape(-1, 1)
        y = np.array(df[pairs[i][1]].values).reshape(-1, 1)
        label_a, label_b = pairs[i]
        model = LinearRegression()
        model.fit(x, y)

        plot_fit_w(model, x, y, label_a, label_b, **kwargs)

        print("coefs:")
        print('np.array([')
        print(repr(model.coef_[0]),',')
        print(repr(model.intercept_))
        print('])')
        coefs.append([model.coef_[0], model.intercept_])
    coefs = np.array(coefs)
    return coefs
    

# ---------------------------------------------------------------------------- #
#                                     Save                                     #
# ---------------------------------------------------------------------------- #

def save(x, y, i=None):
    """Saves data to a file, concatenating with previous data if previous file already exists.
    File naming conventions:
    - "data_" + i + ".npz" for the i-th data file
    - "data_experimental.npz" for a copy of the latest data

    Parameters:
    x (numpy array): Data to save
    y (numpy array): Data to save
    i (int, optional): Iteration number to append to file name (default is None, 
                               in which case it uses the value from config.iteration)
    """
    if i is None:
        i = config.iteration
    try:
        with np.load("data_"+str(i-1)+'.npz', allow_pickle=True) as f:
            x_aux = f["X"]
            y_aux = f["y"]
            x = np.concatenate([x_aux, x])
            y = np.concatenate([y_aux, y])
    except FileNotFoundError:
        pass
    with open("data_"+str(i)+".npz", 'wb') as f:
        np.savez(f, X=x, y=y, iteration=i)
    # copy latest
    with open("data_experimental.npz", 'wb') as f:
        np.savez(f, X=x, y=y, iteration=i)