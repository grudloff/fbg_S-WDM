
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
from numpy import exp as exp

from fbg_swdm.variables import *
#A = np.linspace(-2, 2, N);\

#reflection spectrum
def R(A, A_b, I = 1, dA=0.4*n):
  return I*exp(-4*ln(2)*((A - A_b)/dA)**2)

def X(A_b, I=I):
    x = np.sum(R(A[:, None], A_b[None, :], I[None, :], dA[None, :]) ,axis=-1)
    return x

# gen M datapoints on an N-sized spectrum
def gen_data(train_dist="mesh"):

    if train_dist == "mesh": 
        y_train = np.linspace(A0-0.7*D, A0+0.7*D, np.int(np.sqrt(M))) #1d array
        y_train = np.meshgrid(y_train, y_train) #2d mesh from that array
        y_train = np.reshape(y_train, (FBGN, M)).T
        
    elif train_dist == "uniform":
        y_train = np.random.uniform(A0-0.7*D, A0+0.7*D, [M, FBGN])

    y_test = np.random.uniform(A0-0.7*D, A0+0.7*D, [np.int(M*test_ratio), FBGN])

    # broadcast shape: N, M, FBGN
    X_train = np.sum(R(A[None, :, None], y_train[:, None, :], I[None, None, :], dA[None, None, :]) ,axis=-1)
    X_test = np.sum(R(A[None, :, None], y_test[:, None, :], I[None, None, :], dA[None, None, :]) ,axis=-1)

    return X_train, y_train, X_test, y_test

def plot_datapoint(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(20, 10))
    plt.title("Datapoint visualization")
    plt.plot(A/n, X_test[1,:], label = "FBG1+FBG2")
    N_datapoint = 1
    plt.plot(A/n, R(A[:,None], y_test[N_datapoint, None, 0], I[None, 0], dA[None, 0]), linestyle='dashed', label = "FBG1")
    plt.plot(A/n, R(A[:,None], y_test[N_datapoint, None, 1], I[None, 1], dA[None, 1]), linestyle='dashed', label = "FBG2")
    plt.stem(y_test[1,:]/n, np.full(2, 1), linefmt = 'r-.', markerfmt = "None", basefmt = "None", use_line_collection=True)
    plt.xlabel("Reflection spectrum")
    plt.xlabel("[nm]")
    plt.legend()

def plot_dist(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 10))
    plt.title("Distribution of samples")
    plt.scatter(y_train[:,0]/n,y_train[:,1]/n, s=2, label = "train")
    plt.scatter(y_test[:,0]/n,y_test[:,1]/n, s=2, label = "test")
    plt.ylabel("FBG1[nm]")
    plt.xlabel("FBG2[nm]")
    plt.legend()

def plot_freq_dist(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 5))
    plt.title("Train Histogram")
    plt.xlabel("[nm]")
    plt.hist(y_train/n, bins=100, stacked=True, density=True, label=["FBG1", "FBG2"])
    plt.legend()
    plt.figure(figsize=(10, 5))
    plt.title("Test Histogram")
    plt.xlabel("[nm]")
    plt.hist(y_test/n, bins=100, stacked=True, density=True, label=["FBG1", "FBG2"])
    plt.legend()

def normalize(X_train, y_train, X_test, y_test):
    #A = (A-A0)/d
    y_train = (y_train - A0)/D
    y_test = (y_test - A0)/D

    X_train = X_train/np.sum(I)
    X_test = X_test/np.sum(I)

    return X_train, y_train, X_test, y_test

def denormalize(X_train, y_train, X_test, y_test):
    y_train = y_train*D + A0
    y_test = y_test*D + A0

    X_train = X_train*np.sum(I)
    X_test = X_test*np.sum(I)

    return X_train, y_train, X_test, y_test    