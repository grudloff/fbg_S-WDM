
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
from numpy import exp as exp
from fbg_swdm.variables import λ, A, Δλ, λ0, Δ, Q, M, n, test_ratio, n_eff,\
                               S, M_p, π, sqrt


# reflection spectrum
def old_R(A, A_b, I=1, dA=0.4*n):
    return I*exp(-4*ln(2)*((A - A_b)/dA)**2)


def R(λ, λb, A=A[0], Δλ=Δλ[0]):

    # Δn in relation to Δλ assuming L=N/κ
    Δn = Δλ*2*n_eff/λb/np.sqrt(1 + (S*λb*π*M_p/λ0))

    κ0 = π*Δn/λ0*M_p
    L = S/κ0  # length
    κ = π*Δn/λ*M_p
    
    Δβ = 2*π*n_eff*(1/λ - 1/λb)  # Δβ = β - π/Λ
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


def X(A_b, I=A):
    x = np.sum(R(λ[:, None], A_b[None, :], I[None, :], Δλ[None, :]), axis=-1)
    return x


# gen M datapoints on an N-sized spectrum
def gen_data(train_dist="mesh"):

    if train_dist == "mesh":
        y_train = np.linspace(λ0-0.7*Δ, λ0+0.7*Δ,
                              int(sqrt(M)))  # 1d array
        y_train = np.meshgrid(y_train, y_train)  # 2d mesh from that array
        y_train = np.reshape(y_train, (Q, M)).T

    elif train_dist == "uniform":
        y_train = np.random.uniform(λ0-0.7*Δ, λ0+0.7*Δ, [M, Q])

    y_test = np.random.uniform(λ0-0.7*Δ, λ0+0.7*Δ, [np.int(M*test_ratio),
                                                    Q])

    # broadcast shape: N, M, FBGN
    X_train = np.sum(R(λ[None, :, None], y_train[:, None, :], A[None, None, :],
                       Δλ[None, None, :]), axis=-1)
    X_test = np.sum(R(λ[None, :, None], y_test[:, None, :], A[None, None, :],
                      Δλ[None, None, :]), axis=-1)

    return X_train, y_train, X_test, y_test


def plot_datapoint(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(20, 10))
    plt.title("Datapoint visualization")
    plt.plot(λ/n, X_test[1, :], label="FBG1+FBG2")
    N_datapoint = 1
    plt.plot(λ/n, R(λ[:, None], y_test[N_datapoint, None, 0], A[None, 0],
                    Δλ[None, 0]), linestyle='dashed', label="FBG1")
    plt.plot(λ/n, R(λ[:, None], y_test[N_datapoint, None, 1], A[None, 1],
                    Δλ[None, 1]), linestyle='dashed', label="FBG2")
    plt.stem(y_test[1, :]/n, np.full(2, 1), linefmt='r-.', markerfmt="None",
             basefmt="None", use_line_collection=True)
    plt.xlabel("Reflection spectrum")
    plt.xlabel("[nm]")
    plt.legend()


def plot_dist(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 10))
    plt.title("Distribution of samples")
    plt.scatter(y_train[:, 0]/n, y_train[:, 1]/n, s=2, label="train")
    plt.scatter(y_test[:, 0]/n, y_test[:, 1]/n, s=2, label="test")
    plt.ylabel("FBG1[nm]")
    plt.xlabel("FBG2[nm]")
    plt.legend()


def plot_freq_dist(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 5))
    plt.title("Train Histogram")
    plt.xlabel("[nm]")
    plt.hist(y_train/n, bins=100, stacked=True, density=True,
             label=["FBG1", "FBG2"])
    plt.legend()
    plt.figure(figsize=(10, 5))
    plt.title("Test Histogram")
    plt.xlabel("[nm]")
    plt.hist(y_test/n, bins=100, stacked=True, density=True,
             label=["FBG1", "FBG2"])
    plt.legend()


def normalize(X_train, y_train, X_test, y_test):
    # A = (A-A0)/d
    y_train = (y_train - λ0)/Δ
    y_test = (y_test - λ0)/Δ

    X_train = X_train/np.sum(A)
    X_test = X_test/np.sum(A)

    return X_train, y_train, X_test, y_test


def denormalize(X_train, y_train, X_test, y_test):
    y_train = y_train*Δ + λ0
    y_test = y_test*Δ + λ0

    X_train = X_train*np.sum(A)
    X_test = X_test*np.sum(A)

    return X_train, y_train, X_test, y_test
