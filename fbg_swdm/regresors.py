import numpy as np

# Already implemented regressions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.base import BaseEstimator, RegressorMixin

from lssvr import LSSVR
from sklearn.utils import check_X_y, check_array


class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def hidden_nodes(self, X):
        G = np.dot(X, self.input_weights)
        G = G + self.biases
        H = self.sigmoid(G)
        return H

    def predict(self, X):
        X = check_array(X)
        H = self.hidden_nodes(X)
        out = np.dot(H, self.output_weights)
        return out

    def fit(self, X, y):

        X, y = check_X_y(X, y, dtype='float')

        input_size = X.shape[1]

        self.input_weights = np.random.normal(size=[input_size,
                                                    self.hidden_size])

        self.biases = np.random.normal(size=[self.hidden_size])

        H = self.hidden_nodes(X)
        H_moore_penrose = np.linalg.pinv(H)
        self.output_weights = np.dot(H_moore_penrose, y)
        return self
