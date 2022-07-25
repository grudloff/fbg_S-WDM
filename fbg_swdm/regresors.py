import numpy as np

# Already implemented regressions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.base import BaseEstimator, RegressorMixin

from lssvr import LSSVR
from scipy.linalg import lstsq
from scipy.special import expit as sigmoid
from sklearn.utils import check_X_y, check_array


class ELM(BaseEstimator, RegressorMixin):
    """Extreme Learning Machine (ELM)

    'ELM' trains a single hidden layer neural network by randomly
    initializing the first layer and then computing the output 
    weight matrix as W = pinv(Φ)Y, where Φ is the hidden state matrix
    and Y is the target matrix.

    Parameters 
    ----------
    hidden_size : int, default=10
        Size of the hidden representation.

    Attributes
    ----------
    input_weights_: array-like, shape (n_features, hidden_size)
        Random normal input weight matrix.
    biases_ : array-like, shape(hidden_size,)
        Random normal input bias vector.
    output_weights_ : array-like, shape (hidden_size, n_targets)
        Output weight matrix.
    """
    
    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size

    def _hidden_nodes(self, X):
        h = np.dot(X, self.input_weights_) + self.biases_
        H = sigmoid(h)
        return H

    def predict(self, X):
        """
        Predict using the ELM model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        y : array-like, shape (n_samples, n_targets)
            Predicted targets.
        """
        X = check_array(X, ensure_2d=False)
        H = self._hidden_nodes(X)
        y = np.dot(H, self.output_weights_)
        return y

    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training Samples.
        y : array-like, shape (n_samples, n_targets)
            Training Targets.
        Returns
        -------
        self : object
            An instance of the ELM model.
        """

        X, y = check_X_y(X, y, multi_output=True, dtype='float')

        n_features = X.shape[1]

        self.input_weights_ = np.random.rand(n_features, self.hidden_size)*2-1
        self.biases_ = np.random.rand(self.hidden_size)

        H = self._hidden_nodes(X)
        try:
            self.output_weights_ = lstsq(H, y)[0]
        except:
            self.output_weights_ = lstsq(H, y, lapack_driver='gelss')[0]
        return self


class lookuptable(BaseEstimator, RegressorMixin):
    """Lookup Table

    This is a baseline regression model that is essentially
    a lookup table where the training data constitutes the
    keys and values and a prediction is made by finding the
    value that corresponds to the minimum distance between
    the query and the corresponding key.

    Parameters 
    ----------
    distance : {'euclidean', 'cosine'}, default='euclidean'
        Specifies the distance function to be used in the algorithm.
        

    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        Training Samples.
    y_ : array-like, shape (n_samples, n_targets)
        Training Targets.
    """

    def __init__(self, distance = 'euclidean'):
        self.distance = distance

    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training Samples.
        y : array-like, shape (n_samples, n_targets)
            Training Targets.
        Returns
        -------
        self : object
            An instance of the model.
        """
        X, y = check_X_y(X, y, multi_output=True, dtype='float')
        self.X_ = X
        self.y_ = y

        if self.distance == 'euclidean':
            def dist_func(X):
                alpha = np.linalg.norm(self.X_ - X[None, ...], axis=-1)
                return -alpha
        elif self.distance == 'cosine':
            def dist_func(X):
                alpha = np.sum(self.X_*X[None, ...], axis=-1)
                alpha = alpha/np.linalg.norm(X)/np.linalg.norm(self.X_, axis=-1)
                return alpha
        self.dist_func = dist_func
             
        return self

    def predict(self, X):
        """
        Predict using the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        y : array-like, shape (n_samples, n_targets)
            Predicted targets.
        """
        if len(X.shape)==1:
            alpha = self.dist_func(X)
            indx = np.argmax(alpha)
            y = self.y_[indx]
        else:
            y = np.stack([self.predict(x) for x in X])
            
        return y