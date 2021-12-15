import numpy as np

# Already implemented regressions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.base import BaseEstimator, RegressorMixin

from lssvr import LSSVR
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

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _hidden_nodes(self, X):
        G = np.dot(X, self.input_weights_)
        G = G + self.biases_
        H = self._sigmoid(G)
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

        self.input_weights_ = np.random.normal(size=[n_features,
                                                    self.hidden_size])

        self.biases_ = np.random.normal(size=[self.hidden_size])

        H = self._hidden_nodes(X)
        H_moore_penrose = np.linalg.pinv(H)
        self.output_weights_ = np.dot(H_moore_penrose, y)
        return self


        return self
