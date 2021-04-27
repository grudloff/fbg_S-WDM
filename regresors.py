import numpy as np

#already implemented regressions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from lssvr import LSSVR


class ELM:
    def __init__(self, input_size, hidden_size):
    
        self.input_weights = np.matrix(np.random.normal(size=[input_size,hidden_size]))
    
        self.biases = np.matrix(np.random.normal(size=[hidden_size]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def hidden_nodes(self, X):
        G = X * self.input_weights
        G = G + self.biases
        H = self.sigmoid(G)
        return H

    def predict(self, X):
        H = self.hidden_nodes(X)
        out = H * self.output_weights
        return out

    def fit(self, X, y, hidden_size=1000):
        input_size = X.shape[1]
        H = self.hidden_nodes(X)
        H_moore_penrose = np.linalg.inv(H.T * H) * H.T
        self.output_weights = H_moore_penrose * y
        return
