
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
def gaussian_R(λb, λ, I=1, Δλ=0.4*vars.n):
    return I*exp(-4*ln(2)*((λ - λb)/Δλ)**2)

def partial_R(λb, λ, Δλ, S, Δn_dc):
    # Δn in relation to Δλ assuming L=S/κ
    Δn = Δλ*2*vars.n_eff/λb/np.sqrt(1 + (λb*vars.π*vars.η/vars.λ0/S))

    κ0 = vars.π*Δn/vars.λ0*vars.η
    L = S/κ0  # length
    κ = vars.π*Δn/λ*vars.η
    
    Δβ = 2*vars.π*vars.n_eff*(1/λ - 1/λb)  # Δβ = β - π/Λ
    σ = 2*vars.π/λ*Δn_dc*vars.η
    σ_hat = Δβ + σ
    s_2 = κ**2 - σ_hat**2  # s**2
    s = np.lib.scimath.sqrt(s_2)

    return s, s_2, L, κ, σ_hat

def true_R(λb, λ, A, Δλ, I, Δn_dc):
    S = get_S(I)
    s, s_2, L, κ, σ_hat = partial_R(λb, λ, Δλ, S, Δn_dc)

    # auxiliary variables
    sL = s*L
    sinh_sL_2 = np.sinh(sL)**2
    cosh_sL_2 = np.cosh(sL)**2

    R = κ**2*sinh_sL_2/(σ_hat**2*sinh_sL_2 + s_2*cosh_sL_2)

    R = np.abs(R)  # amplitude
    R = R*A

    return R

def get_I(S):
    """ Get Peak Reflectivity from saturation """
    return np.tanh(S)**2

def get_S(I):
    """ Get Saturation from Peak Reflectivity """
    if np.max(I) > vars.I_max:
        raise ValueError('Values should be lower than 0.99')
    S = np.arctanh(np.sqrt(I))
    return S

#TODO: transferMatrix for gaussian simulation ?
def transferMatrix(λb, λ, Δλ, I, Δn_dc):
    S = get_S(I)
    s, s_2, L, κ, σ_hat = partial_R(λb, λ, Δλ, S, Δn_dc)
    sL = s*L
    cosh_sL = np.cosh(sL)
    sinh_sL = np.sinh(sL)

    T = np.array([[cosh_sL - 1j*σ_hat/s*sinh_sL, -1j*κ/s*sinh_sL],
                    [1j*κ/s*sinh_sL, cosh_sL + 1j*σ_hat/s*sinh_sL]])
    return T

def batch_matmul(A, B):
    """ Matrix multiplication of first two dimensions of A and B"""
    return np.einsum('ij...,jk...->ik...', A, B)

def batch_matmul_outprod(A, B):
    """ Simultaneous matmul of first two dimensions and outer product of last
        dimension"""
    return np.einsum('ij...l,jk...m->ik...lm', A, B)

def prep_dims(A_b, λ, A, Δλ, I, Δn_dc):
    if len(A_b.shape) > 1:
        A_b = A_b[:, None, :]
        λ = λ[None, :, None]
        A = A[None, None, :]
        Δλ = Δλ[None, None, :]
        I = I[None, None, :]
        Δn_dc = Δn_dc[None, None, :]
    else:
        A_b = A_b[None, :]
        λ = λ[:, None]
        A = A[None, :]
        Δλ = Δλ[None, :]
        I = I[None, :]
        Δn_dc = Δn_dc[None, :]
    return A_b, λ, A, Δλ, I, Δn_dc

def R(*args, **kwargs):
    simulation = kwargs.pop('simulation', vars.simulation)
    if simulation == 'gaussian':
        return gaussian_R(*args, **kwargs)
    elif simulation == 'true':
        return true_R(*args, **kwargs)
    else:
        raise ValueError("simulation must be in {'gaussian','true'}")

# ---------------------------------------------------------------------------- #
#                                Data Generation                               #
# ---------------------------------------------------------------------------- #

def X(A_b, λ=None, A=None, Δλ=None, I=None, Δn_dc=None, batch_size=None, **kwargs):

    # defaults if None
    λ = vars.λ if λ is None else λ 
    A = vars.A if A is None else A
    Δλ = vars.Δλ if Δλ is None else Δλ 
    I = vars.I if I is None else I
    Δn_dc = vars.Δn_dc if Δn_dc is None else Δn_dc 

    if batch_size != None:
        A_b = np.array_split(A_b, vars.M//batch_size)
        x = np.concatenate([X(a_b, λ, A, Δλ, I, Δn_dc) for a_b in A_b])
        return x

    A_b, λ, A, Δλ, I, Δn_dc = prep_dims(A_b, λ, A, Δλ, I, Δn_dc)

    if vars.topology == 'parallel':
        x = np.sum(R(A_b, λ, A, Δλ, I, Δn_dc, **kwargs), axis=-1)

    elif vars.topology == 'serial_new':
        """Serial topology matrix simulation, no incoherent interface between FBGs"""

        T_prev = np.identity(2)

        for T in np.rollaxis(transferMatrix(A_b, λ, A, Δλ, I, Δn_dc)):

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = batch_matmul(T_prev, At)
            T_prev = batch_matmul(T_prev, T)

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2

    elif vars.topology == 'serial_rand':
        """serial_new but with incoherent interface by simulation of random multiple paths"""
        
        L = 10000 # batch_size
        T_prev = np.identity(2)
        i = 1 # fbg number
        for T in np.rollaxis(transferMatrix(A_b, λ, A, Δλ, I, Δn_dc)):

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = batch_matmul(T_prev, At)
            T_prev = batch_matmul(T_prev, T[..., None])

            # random gap phase change
            if i < vars.Q:
                F = 1j*vars.π*2*np.random.rand(L)[None, :]*np.array([-1, 1])[:, None]
                F = np.exp(F)
                F = np.identity(2)[..., None]*F[None,...]
                T_prev = batch_matmul(T_prev, F[:,:,None])

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
        for T in np.rollaxis(transferMatrix(A_b, λ, A, Δλ, I, Δn_dc)):

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = batch_matmul(T_prev, At)
            T_prev = batch_matmul(T_prev, T[..., None])

            # scan gap phase change
            if i < vars.Q:
                T_prev = batch_matmul_outprod(T_prev, F[:,:,None,None])
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

        for r in np.rollaxis(R(A_b, λ, A, Δλ, I, Δn_dc, **kwargs)):
            t = 1-r

            T = 1/t[None,None,:] \
                *np.array([[np.ones_like(r), -r],
                           [r, 1-2*r]])
            # t**2-r**2 = 1-2r

            # atenuation
            a = a.squeeze()
            At = np.diag([a**(-1.0/2), a**(1.0/2)])
            T_prev = batch_matmul(T_prev, At)
            T_prev = batch_matmul(T_prev, T)

        T = T_prev
        x = T[1,0]/T[0,0]

    elif vars.topology == 'serial':
        """Old, only valid for 2-FBGs"""
        x = 0
        for x_next in np.rollaxis(R(A_b, λ, A, Δλ, I, Δn_dc, **kwargs), -1):
            x = x + (1-x)**2*x_next/(1-x_next*x)

    elif vars.topology == 'serial_rec':
        """Recurrent equivalent of serial_abs """
        r = 0
        t = 1
        for r_next in np.rollaxis(R(A_b, λ, A, Δλ, I, Δn_dc, **kwargs), -1):
            t_next = 1 - r_next
            F = 1/(1 - a*r*r_next) # resonance
            r = r + a*r_next*t**2*F
            t = sqrt(a)*t*t_next*F
        x = r

    else:
        raise ValueError("Topology not valid")

    x = np.squeeze(x)

    return x


# gen M datapoints on an N-sized spectrum
def gen_data(train_dist="uniform", batch_size=None):

    if vars.topology=='parallel' and np.sum(vars.A)>1:
        raise ValueError('Attenuation vector should sum less than one in the parallel topology.')

    Δ = vars.portion*vars.Δ

    if train_dist == "mesh":
        y_train = np.linspace(vars.λ0-Δ, vars.λ0+Δ,
                              int(np.ceil(vars.M**(1/vars.Q))))  # 1d array
        aux = [y_train for _ in range(vars.Q)]
        y_train = np.meshgrid(*aux)  # 2d mesh from that array
        y_train = np.reshape(y_train, (vars.Q, vars.M)).T

    elif train_dist == "uniform":
        y_train = vars.λ0 + np.random.uniform(-Δ, Δ, [vars.M, vars.Q])

    m = int(0.1*vars.M)
    y_test = vars.λ0 + np.random.uniform(-Δ, Δ, [m, vars.Q])

    # broadcast shape: N, M, FBGN
    X_train = X(y_train, batch_size=batch_size)
    X_test = X(y_test, batch_size=batch_size)

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