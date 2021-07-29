
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
from numpy import exp as exp
import fbg_swdm.variables as vars
from math import sqrt


# reflection spectrum
def gaussian_R(λb, λ, A=1, Δλ=0.4*vars.n):
    return A*exp(-4*ln(2)*((λ - λb)/Δλ)**2)


def R(λb, λ, A=vars.A[0], Δλ=vars.Δλ[0]):

    # Δn in relation to Δλ assuming L=S/κ
    Δn = Δλ*2*vars.n_eff/λb/np.sqrt(1 + (λb*vars.π*vars.M_p/vars.λ0/vars.S))

    κ0 = vars.π*Δn/vars.λ0*vars.M_p
    L = vars.S/κ0  # length
    κ = vars.π*Δn/λ*vars.M_p
    
    Δβ = 2*vars.π*vars.n_eff*(1/λ - 1/λb)  # Δβ = β - π/Λ
    s_2 = κ**2 - Δβ**2  # s**2
    s = np.lib.scimath.sqrt(s_2)

    # auxiliary variables
    sL = s*L
    sinh_sL_2 = np.sinh(sL)**2
    cosh_sL_2 = np.cosh(sL)**2

    R = κ**2*sinh_sL_2/(Δβ**2*sinh_sL_2 + s_2*cosh_sL_2)
    R = R/np.tanh(κ0*L)**2  # normalization
    R = np.abs(R)  # amplitude
    R = R*A

    return R


def X(A_b, λ=vars.λ, A=vars.A, Δλ=vars.Δλ):
    if len(A_b.shape) > 1:
        A_b = A_b[:, None, :]
        λ = λ[None, :, None]
        A = A[None, None, :]
        Δλ = Δλ[None, None, :]
    else:
        A_b = A_b[None, :]
        λ = λ[:, None]
        A = A[None, :]
        Δλ = Δλ[None, :]

    if vars.topology == 'parallel':
        x = np.sum(R(A_b, λ, A, Δλ), axis=-1)

    elif vars.topology == 'serial':
        x = 0
        for i, j, k in zip(A_b.T, A.T, Δλ.T):
            x_next = R(i.T, np.squeeze(λ), j.T, k.T)
            x = x + (1-x)**2*x_next/(1-x_next*x)
    return x


# gen M datapoints on an N-sized spectrum
def gen_data(train_dist="mesh", Δ=0.7*vars.Δ):

    if train_dist == "mesh":
        y_train = np.linspace(vars.λ0-Δ, vars.λ0+Δ,
                              int(sqrt(vars.M)))  # 1d array
        y_train = np.meshgrid(y_train, y_train)  # 2d mesh from that array
        y_train = np.reshape(y_train, (vars.Q, vars.M)).T

    elif train_dist == "uniform":
        y_train = np.random.uniform(vars.λ0-Δ, vars.λ0+Δ, [vars.M, vars.Q])

    y_test = np.random.uniform(vars.λ0-Δ, vars.λ0+Δ, [np.int(vars.test_M),
                                                      vars.Q])

    # broadcast shape: N, M, FBGN
    X_train = X(y_train, vars.λ, vars.A, vars.Δλ)
    X_test = X(y_test, vars.λ, vars.A, vars.Δλ)

    return X_train, y_train, X_test, y_test


def plot_datapoint(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(20, 10))
    plt.title("Datapoint visualization")
    plt.plot(vars.λ/vars.n, X_test[1, :], label="FBG1+FBG2")
    N_datapoint = 1
    plt.plot(vars.λ/vars.n, R(vars.λ[:, None], y_test[N_datapoint, None, 0], vars.A[None, 0],
                    vars.Δλ[None, 0]), linestyle='dashed', label="FBG1")
    plt.plot(vars.λ/vars.n, R(vars.λ[:, None], y_test[N_datapoint, None, 1], vars.A[None, 1],
                    vars.Δλ[None, 1]), linestyle='dashed', label="FBG2")
    plt.stem(y_test[1, :]/vars.n, np.full(2, 1), linefmt='r-.', markerfmt="None",
             basefmt="None", use_line_collection=True)
    plt.xlabel("Reflection spectrum")
    plt.xlabel("[nm]")
    plt.legend()


def plot_dist(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 10))
    plt.title("Distribution of samples")
    plt.scatter(y_train[:, 0]/vars.n, y_train[:, 1]/vars.n, s=2, label="train")
    plt.scatter(y_test[:, 0]/vars.n, y_test[:, 1]/vars.n, s=2, label="test")
    plt.ylabel("FBG1[nm]")
    plt.xlabel("FBG2[nm]")
    plt.legend()


def plot_freq_dist(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 5))
    plt.title("Train Histogram")
    plt.xlabel("[nm]")
    plt.hist(y_train/vars.n, bins=100, stacked=True, density=True,
             label=["FBG1", "FBG2"])
    plt.legend()
    plt.figure(figsize=(10, 5))
    plt.title("Test Histogram")
    plt.xlabel("[nm]")
    plt.hist(y_test/vars.n, bins=100, stacked=True, density=True,
             label=["FBG1", "FBG2"])
    plt.legend()


def normalize(X_train, y_train, X_test, y_test):
    # A = (A-A0)/d
    y_train = (y_train - vars.λ0)/vars.Δ
    y_test = (y_test - vars.λ0)/vars.Δ

    X_train = X_train/np.sum(vars.A)
    X_test = X_test/np.sum(vars.A)

    return X_train, y_train, X_test, y_test


def denormalize(X_train, y_train, X_test, y_test):
    y_train = y_train*vars.Δ + vars.λ0
    y_test = y_test*vars.Δ + vars.λ0

    X_train = X_train*np.sum(vars.A)
    X_test = X_test*np.sum(vars.A)

    return X_train, y_train, X_test, y_test
