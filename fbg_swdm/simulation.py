
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
from numpy import exp as exp
import fbg_swdm.variables as vars
from math import sqrt
from functools import wraps


# ---------------------------------------------------------------------------- #
#                                  Decorators                                  #
# ---------------------------------------------------------------------------- #


def listify(func):
    """decorator for making generator functions return a list instead"""
    @wraps(func)
    def new_func(*args, **kwargs):
        output = list(func(*args, **kwargs))
        if len(output)==1:
            output = output[0]
        return output

    return new_func


# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
# reflection spectrum
def gaussian_R(λb, λ, A=1, Δλ=0.4*vars.n):
    return A*exp(-4*ln(2)*((λ - λb)/Δλ)**2)

def partial_R(λb, λ, A=vars.A[0], Δλ=vars.Δλ[0], S=vars.S[0]):
    # Δn in relation to Δλ assuming L=S/κ
    Δn = Δλ*2*vars.n_eff/λb/np.sqrt(1 + (λb*vars.π*vars.M_p/vars.λ0/S))

    κ0 = vars.π*Δn/vars.λ0*vars.M_p
    L = S/κ0  # length
    κ = vars.π*Δn/λ*vars.M_p
    
    Δβ = 2*vars.π*vars.n_eff*(1/λ - 1/λb)  # Δβ = β - π/Λ
    s_2 = κ**2 - Δβ**2  # s**2
    s = np.lib.scimath.sqrt(s_2)

    return s, s_2, L, κ, Δβ

def R(λb, λ, A=vars.A[0], Δλ=vars.Δλ[0], S=vars.S[0]):

    s, s_2, L, κ, Δβ = partial_R(λb, λ, A, Δλ, S)

    # auxiliary variables
    sL = s*L
    sinh_sL_2 = np.sinh(sL)**2
    cosh_sL_2 = np.cosh(sL)**2

    R = κ**2*sinh_sL_2/(Δβ**2*sinh_sL_2 + s_2*cosh_sL_2)

    R = np.abs(R)  # amplitude
    R = R*A

    return R

def get_max_R(S=vars.S):
    return np.tanh(S)**2

def get_S(P):
    if np.max(P) > 0.99:
        raise ValueError('Values should be lower than 0.99')
    S = np.arctanh(np.sqrt(P))
    return S

def transferMatrix(λb, λ, A=vars.A[0], Δλ=vars.Δλ[0], S=vars.S[0]):
    s, s_2, L, κ, Δβ = partial_R(λb, λ, A, Δλ, S)
    sL = s*L
    cosh_sL = np.cosh(sL)
    sinh_sL = np.sinh(sL)

    T = np.array([[cosh_sL - 1j*Δβ/s*sinh_sL, -1j*κ/s*sinh_sL],
                    [1j*κ/s*sinh_sL, cosh_sL + 1j*Δβ/s*sinh_sL]])
    return T

# ---------------------------------------------------------------------------- #
#                                Data Generation                               #
# ---------------------------------------------------------------------------- #

def X(A_b, λ=vars.λ, A=vars.A, Δλ=vars.Δλ, S=vars.S, batch_size=None):

    if batch_size != None:
        n = vars.M//batch_size
        A_b = np.array_split(A_b, n)
        x = np.concatenate([X(a_b, λ, A, Δλ, S) for a_b in A_b])
        return x

    if len(A_b.shape) > 1:
        A_b = A_b[:, None, :]
        λ = λ[None, :, None]
        A = A[None, None, :]
        Δλ = Δλ[None, None, :]
        S = S[None, None, :]
    else:
        A_b = A_b[None, :]
        λ = λ[:, None]
        A = A[None, :]
        Δλ = Δλ[None, :]
        S = S[None, :]

    if vars.topology == 'parallel':
        x = np.sum(R(A_b, λ, A, Δλ, S), axis=-1)

    elif vars.topology == 'serial_new':
        """Serial topology matrix simulation, no incoherent interface between FBGs"""

        T_prev = np.identity(2)
 
        for b, a, l, s in zip(A_b.T, A.T, Δλ.T, S.T):
            T = transferMatrix(b.T, np.squeeze(λ), a.T, l.T ,s.T)

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, At)

            # 'ij...,jk...->ik...' einsum is matmul of first two dimensions
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, T)

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2

    elif vars.topology == 'serial_rand':
        """serial_new but with incoherent interface by simulation of random multiple paths"""
        
        M = 10000 # batch_size
        T_prev = np.identity(2)
        i = 1 # fbg number
        for b, a, l, s in zip(A_b.T, A.T, Δλ.T, S.T):
            T = transferMatrix(b.T, np.squeeze(λ), a.T, l.T, s.T)

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, At)

            # 'ij...,jk...->ik...' einsum is matmul of first two dimensions
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, T[..., None])

            # random gap phase change
            if i < vars.Q:
                F = 1j*vars.π*2*np.random.rand(M)[None, :]*np.array([-1, 1])[:, None]
                F = np.exp(F)
                F = np.identity(2)[..., None]*F[None,...]
                T_prev = np.einsum('ij...,jk...->ik...', T_prev, F[:,:,None])

            i += 1

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2
        x = np.mean(x, axis=-1) # average different paths


    elif vars.topology == 'serial_phase':
        """serial_rand equivalent but with a phase sweep instead, making it more efficient"""
        
        N=vars.φN
        phase = np.arange(0,1,1.0/N)
        #phase = np.linspace(1/N/2,1- 1/N/2, N)
        phase = 1j*vars.π*phase
        F = phase[None, :]*np.array([-1, 1])[:, None]
        F = np.exp(F)
        F = np.identity(2)[..., None]*F[None,...]

        T_prev = np.identity(2)
        i = 1 # fbg number
        for b, a, l, s in zip(A_b.T, A.T, Δλ.T, S.T):
            T = transferMatrix(b.T, np.squeeze(λ), a.T, l.T, s.T)

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, At)

            # 'ij...,jk...->ik...' einsum is matmul of first two dimensions
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, T[..., None])

            # scan gap phase change
            if i < vars.Q:
                # simultaneous matmul of first two dimensions
                # and outer product of gap phase dimensions
                T_prev = np.einsum('ij...l,jk...m->ik...lm', T_prev, F[:,:,None,None])
                # flatten outer product
                T_prev = T_prev.reshape(*T_prev.shape[:-2], -1)

            i += 1

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2
        x = np.mean(x, axis=-1) # average different paths

    elif vars.topology == 'serial_abs':
        """Approximation of serial_phase"""
        
        T_prev = np.identity(2)

        for b, a, l, s in zip(A_b.T, A.T, Δλ.T, S.T):
            r = R(b.T, np.squeeze(λ), 1, l.T, s.T)
            t = 1-r

            T = 1/t[None,None,:] \
                *np.array([[np.ones_like(r), -r],
                           [r, 1-2*r]])
            # t**2-r**2 = 1-2r

            # atenuation
            At = np.diag([a**(-1.0/2), a**(1.0/2)])
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, At)

            # 'ij...,jk...->ik...' einsum is matmul of first two dimensions
            T_prev = np.einsum('ij...,jk...->ik...', T_prev, T)

        T = T_prev
        x = T[1,0]/T[0,0]

    elif vars.topology == 'serial':
        """Old, only valid for 2-FBGs"""
        x = 0
        for b, a, l, s in zip(A_b.T, A.T, Δλ.T, S.T):
            x_next = R(b.T, np.squeeze(λ), a, l.T, s.T)
            x = x + (1-x)**2*x_next/(1-x_next*x)

    elif vars.topology == 'serial_rec':
        """Recurrent equivalent of serial_abs """
        r = 0
        t = 1
        at = 1
        for b, a, l, s in zip(A_b.T, A.T, Δλ.T, S.T):
            at = float(a)/at
            r_next = R(b.T, np.squeeze(λ), 1, l.T, s.T)
            t_next = 1 - r_next
            F = 1/(1 - at*r*r_next) # resonance
            r = r + at*r_next*t**2*F
            t = sqrt(at)*t*t_next*F
        x = r

    else:
        raise ValueError("Topology type not valid")

    x = np.squeeze(x)

    return x


# gen M datapoints on an N-sized spectrum
def gen_data(train_dist="uniform", portion=0.6, batch_size=None):

    if vars.topology=='parallel' and np.sum(vars.A)>1:
        raise ValueError('Attenuation vector should sum less than one in the parallel topology.')

    Δ = portion*vars.Δ

    if train_dist == "mesh":
        y_train = np.linspace(vars.λ0-Δ, vars.λ0+Δ,
                              int(np.ceil(vars.M**(1/vars.Q))))  # 1d array
        aux = [y_train for _ in range(vars.Q)]
        y_train = np.meshgrid(*aux)  # 2d mesh from that array
        y_train = np.reshape(y_train, (vars.Q, vars.M)).T

    elif train_dist == "uniform":
        y_train = np.random.uniform(vars.λ0-Δ, vars.λ0+Δ, [vars.M, vars.Q])

    y_test = np.random.uniform(vars.λ0-Δ, vars.λ0+Δ, [np.int(vars.test_M),
                                                      vars.Q])

    # broadcast shape: N, M, FBGN
    X_train = X(y_train, vars.λ, vars.A, vars.Δλ, vars.S, batch_size)
    X_test = X(y_test, vars.λ, vars.A, vars.Δλ, vars.S, batch_size)

    return X_train, y_train, X_test, y_test

# ---------------------------------------------------------------------------- #
#                      Normalizations and Denormalization                      #
# ---------------------------------------------------------------------------- #


@listify
def normalize(X=None, y=None, *args):
    if X is not None:
        # x is not normalized
        yield X

    if y is not None:
        y = (y - vars.λ0)/vars.Δ
        yield y    

    if args:
        yield from normalize(*args)

@listify
def denormalize(X=None, y=None, *args):
    if X is not None:
        # x is not normalized
        yield X

    if y is not None:
        y = y*vars.Δ + vars.λ0
        yield y    
        
    if args:
        yield from normalize(*args)