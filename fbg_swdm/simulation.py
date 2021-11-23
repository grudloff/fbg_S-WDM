
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
from numpy import exp as exp
import fbg_swdm.variables as vars
from math import sqrt


# reflection spectrum
def gaussian_R(λb, λ, A=1, Δλ=0.4*vars.n):
    return A*exp(-4*ln(2)*((λ - λb)/Δλ)**2)

def partial_R(λb, λ, A=vars.A[0], Δλ=vars.Δλ[0]):
    # Δn in relation to Δλ assuming L=S/κ
    Δn = Δλ*2*vars.n_eff/λb/np.sqrt(1 + (λb*vars.π*vars.M_p/vars.λ0/vars.S))

    κ0 = vars.π*Δn/vars.λ0*vars.M_p
    L = vars.S/κ0  # length
    κ = vars.π*Δn/λ*vars.M_p
    
    Δβ = 2*vars.π*vars.n_eff*(1/λ - 1/λb)  # Δβ = β - π/Λ
    s_2 = κ**2 - Δβ**2  # s**2
    s = np.lib.scimath.sqrt(s_2)

    return s, s_2, L, κ, κ0, Δβ

def R(λb, λ, A=vars.A[0], Δλ=vars.Δλ[0]):

    s, s_2, L, κ, κ0, Δβ = partial_R(λb, λ, A, Δλ)

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
        T_prev = None
        for b, a, l in zip(A_b.T, A.T, Δλ.T):
            s, s_2, L, κ, κ0, Δβ = partial_R(b.T, np.squeeze(λ), a.T, l.T)
            sL = s*L
            cosh_sL = np.cosh(sL)
            sinh_sL = np.sinh(sL)
            T = np.array([[cosh_sL - 1j*Δβ/s*sinh_sL, -1j*κ/s*sinh_sL],
                          [1j*κ/s*sinh_sL, cosh_sL + 1j*Δβ/s*sinh_sL]])
            # atenuation
            crot_a = a**(1.0/4)
            T[0, :] *= 1/crot_a
            T[1, :] *= crot_a
            if T_prev is None:
                T_prev = T
            else:
                # matmul of first two dimensions
                T_prev = np.einsum('ij...,jk...->ik...', T, T_prev)

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2
        # normalization 
        #x = x/np.max(x)*np.max(A)

    elif vars.topology == 'serial_old':
        x = 0
        for i, j, k in zip(A_b.T, A.T, Δλ.T):
            x_next = R(i.T, np.squeeze(λ), j.T, k.T)
            x = x + (1-x)**2*x_next/(1-x_next*x)
    return x


# gen M datapoints on an N-sized spectrum
def gen_data(train_dist="mesh", portion=0.6):

    Δ = portion*vars.Δ

    if train_dist == "mesh":
        y_train = np.linspace(vars.λ0-Δ, vars.λ0+Δ,
                              int(vars.M**(1/vars.Q)))  # 1d array
        aux = [y_train for _ in range(vars.Q)]
        y_train = np.meshgrid(*aux)  # 2d mesh from that array
        y_train = np.reshape(y_train, (vars.Q, vars.M)).T

    elif train_dist == "uniform":
        y_train = np.random.uniform(vars.λ0-Δ, vars.λ0+Δ, [vars.M, vars.Q])

    y_test = np.random.uniform(vars.λ0-Δ, vars.λ0+Δ, [np.int(vars.test_M),
                                                      vars.Q])

    # broadcast shape: N, M, FBGN
    X_train = X(y_train, vars.λ, vars.A, vars.Δλ)
    X_test = X(y_test, vars.λ, vars.A, vars.Δλ)

    return X_train, y_train, X_test, y_test


def plot_datapoint(X, Y, N_datapoint = 1):
    plt.figure(figsize=(20, 10))
    plt.title("Datapoint visualization")
    plt.plot(vars.λ/vars.n, X[N_datapoint, :], label="$\sum FBGi$")
    for i in range(vars.Q):
        plt.plot(vars.λ/vars.n, R(vars.λ[:, None], Y[N_datapoint, None, i],
                 vars.A[None, i], vars.Δλ[None, i]), linestyle='dashed',
                 label="FBG"+str(i))
    plt.stem(Y[N_datapoint, :]/vars.n, np.full(vars.Q, 1), linefmt='r-.', markerfmt="None",
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

    if vars.topology == 'parallel':
        X_train = X_train/np.sum(vars.A)
        X_test = X_test/np.sum(vars.A)
    if vars.topology == 'serial':
        X_train = X_train/np.max(vars.A)
        X_test = X_test/np.max(vars.A)        

    return X_train, y_train, X_test, y_test


def denormalize(X_train, y_train, X_test, y_test):
    y_train = y_train*vars.Δ + vars.λ0
    y_test = y_test*vars.Δ + vars.λ0
    if vars.topology == 'parallel':
        X_train = X_train*np.sum(vars.A)
        X_test = X_test*np.sum(vars.A)
    if vars.topology == 'serial':
        X_train = X_train*np.max(vars.A)
        X_test = X_test*np.max(vars.A) 


    return X_train, y_train, X_test, y_test
