
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
from numpy import exp as exp
import fbg_swdm.variables as config
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
def gaussian_R(λb, λ, A, Δλ, I, Δn_dc):
    if I is None:
        I = config.I
    if Δλ is None:
        Δλ = config.Δλ
    return A*I*exp(-4*ln(2)*((λ - λb)/Δλ)**2)

def partial_R(λb, λ, Δλ, ζ, Δn_dc):
    if Δλ is None and ζ is None:
        Δn = config.Δn
        κ = config.κ
        L = config.L
    elif Δλ is None or ζ is None:
        raise ValueError("Both Δλ and ζ should be None to use precomputed parameters.")
    else:
        S = config.get_saturation(ζ)
        Δn = Δλ*config.n_eff/config.λ0/config.η/np.sqrt(1 + (config.π/ζ)**2)/S
        κ = config.π*Δn/config.λ0*config.η
        L = ζ/κ  # length
    
    Δβ = 2*config.π*config.n_eff*(1/λ - 1/λb)  # Δβ = β - π/Λ
    σ = 2*config.π/λ*Δn_dc*config.η
    σ_hat = Δβ + σ
    s_2 = κ**2 - σ_hat**2  # s**2
    s = np.lib.scimath.sqrt(s_2)

    return s, s_2, L, κ, σ_hat

def true_R(λb, λ, A, Δλ, I, Δn_dc):
    ζ = config.get_zeta(I)
    s, s_2, L, κ, σ_hat = partial_R(λb, λ, Δλ, ζ, Δn_dc)

    # auxiliary variables
    sinh_sL_2 = np.sinh(s*L)**2
    cosh_sL_2 = 1 + sinh_sL_2 # hyperbolic identity

    R = κ**2*sinh_sL_2/(σ_hat**2*sinh_sL_2 + s_2*cosh_sL_2)

    R = np.abs(R) # amplitude
    R = R*A # attenuation

    return R

def get_I(ζ):
    """ Get Peak Reflectivity from saturation """
    return np.tanh(ζ)**2

#TODO: transferMatrix for gaussian simulation ?
# TODO: accepts multiple sensors computation in parallel?
def transferMatrix(λb, λ, Δλ, I, Δn_dc):
    ζ = config.get_zeta(I)
    s, s_2, L, κ, σ_hat = partial_R(λb, λ, Δλ, ζ, Δn_dc)
    sL = s*L
    cosh_sL = np.cosh(sL)
    sinh_sL = np.sinh(sL)

    # TODO: probably have to update this np.array to an adequate stack function
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
        if Δλ is not None:
            Δλ = Δλ[None, None, :]
        if I is not None:
            I = I[None, None, :] 
        Δn_dc = Δn_dc[None, None, :]
    else:
        A_b = A_b[None, :]
        λ = λ[:, None]
        A = A[None, :]
        if Δλ is not None:  
            Δλ = Δλ[None, :]
        if I is not None:
            I = I[None, :]
        Δn_dc = Δn_dc[None, :]
    return A_b, λ, A, Δλ, I, Δn_dc

def R(*args, **kwargs):
    simulation = kwargs.pop('simulation', config.simulation)
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

    # TODO: move default to R functions
    # defaults if None
    λ = config.λ if λ is None else λ 
    A = config.A if A is None else A
    # Don't set I and Δλ to default to use precomputed parameters
    # Δλ = vars.Δλ if Δλ is None else Δλ 
    # I = vars.I if I is None else I
    Δn_dc = config.Δn_dc if Δn_dc is None else Δn_dc 

    if batch_size != None:
        A_b = np.array_split(A_b, config.M//batch_size)
        x = np.concatenate([X(a_b, λ, A, Δλ, I, Δn_dc) for a_b in A_b])
        return x

    A_b, λ, A, Δλ, I, Δn_dc = prep_dims(A_b, λ, A, Δλ, I, Δn_dc)

    if config.topology == 'parallel':
        x = np.sum(R(A_b, λ, A, Δλ, I, Δn_dc, **kwargs), axis=-1)

    elif config.topology == 'serial_new':
        """Serial topology matrix simulation, no incoherent interface between FBGs"""

        T_prev = np.identity(2)

        for a, T in zip(A.T, np.rollaxis(transferMatrix(A_b, λ, Δλ, I, 
                                                      Δn_dc, **kwargs), -1)):

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = batch_matmul(T_prev, At)
            T_prev = batch_matmul(T_prev, T)

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2

    elif config.topology == 'serial_rand':
        """serial_new but with incoherent interface by simulation of random multiple paths"""
        
        L = 10000 # batch_size
        T_prev = np.identity(2)
        i = 1 # fbg number
        for a, T in zip(A.T, np.rollaxis(transferMatrix(A_b, λ, Δλ, I, 
                                                      Δn_dc, **kwargs), -1)):

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = batch_matmul(T_prev, At)
            T_prev = batch_matmul(T_prev, T[..., None])

            # random gap phase change
            if i < config.Q:
                F = 1j*config.π*2*np.random.rand(L)[None, :]*np.array([-1, 1])[:, None]
                F = np.exp(F)
                F = np.identity(2)[..., None]*F[None,...]
                T_prev = batch_matmul(T_prev, F[:,:,None])

            i += 1

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2
        x = np.mean(x, axis=-1) # average different paths


    elif config.topology == 'serial_phase':
        """serial_rand equivalent but with a phase sweep instead, making it more efficient"""
        
        N=config.φN
        phase = np.arange(0,1,1.0/N)
        #phase = np.linspace(1/N/2,1- 1/N/2, N)
        phase = 1j*config.π*phase
        F = phase[None, :]*np.array([-1, 1])[:, None]
        F = np.exp(F)
        F = np.identity(2)[..., None]*F[None,...]

        T_prev = np.identity(2)
        i = 1 # fbg number
        for a, T in zip(A.T, np.rollaxis(transferMatrix(A_b, λ, Δλ, I, 
                                                      Δn_dc, **kwargs), -1)):

            # atenuation
            At = np.diag([a**(-1.0/4), a**(1.0/4)])
            T_prev = batch_matmul(T_prev, At)
            T_prev = batch_matmul(T_prev, T[..., None])

            # scan gap phase change
            if i < config.Q:
                T_prev = batch_matmul_outprod(T_prev, F[:,:,None,None])
                # flatten outer product
                T_prev = T_prev.reshape(*T_prev.shape[:-2], -1)

            i += 1

        T = T_prev
        x = T[1,0]/T[0,0]
        x = np.abs(x)**2
        x = np.mean(x, axis=-1) # average different paths

    elif config.topology == 'serial_abs':
        """Approximation of serial_phase"""
        
        T_prev = np.identity(2)

        for a, r_next in zip(A.T, np.rollaxis(R(A_b, λ, np.ones(config.Q), 
                                              Δλ, I, Δn_dc, **kwargs), -1)):
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

    elif config.topology == 'serial':
        """Old, only valid for 2-FBGs"""
        x = 0
        for x_next in np.rollaxis(R(A_b, λ, A, Δλ, I, Δn_dc, **kwargs), -1):
            x = x + (1-x)**2*x_next/(1-x_next*x)

    elif config.topology == 'serial_rec':
        """Recurrent equivalent of serial_abs """
        r = 0
        t = 1
        for a, r_next in zip(A.T, np.rollaxis(R(A_b, λ, np.ones(config.Q), 
                                              Δλ, I, Δn_dc, **kwargs), -1)):
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

    if config.topology=='parallel' and np.sum(config.A)>1:
        raise ValueError('Attenuation vector should sum less than one in the parallel topology.')

    Δ = config.portion*config.Δ

    if train_dist == "mesh":
        y_train = np.linspace(config.λ0-Δ, config.λ0+Δ,
                              int(np.ceil(config.M**(1/config.Q))))  # 1d array
        aux = [y_train for _ in range(config.Q)]
        y_train = np.meshgrid(*aux)  # 2d mesh from that array
        y_train = np.reshape(y_train, (config.Q, config.M)).T

    elif train_dist == "uniform":
        y_train = config.λ0 + np.random.uniform(-Δ, Δ, [config.M, config.Q])

    m = int(0.1*config.M)
    y_test = config.λ0 + np.random.uniform(-Δ, Δ, [m, config.Q])

    # broadcast shape: N, M, FBGN
    X_train = X(y_train, batch_size=batch_size)
    X_test = X(y_test, batch_size=batch_size)

    return X_train, y_train, X_test, y_test

# ---------------------------------------------------------------------------- #
#                      Normalizations and Denormalization                      #
# ---------------------------------------------------------------------------- #

@listify
def normalize(X=None, y=None, *args, **kwargs):
    if X is not None:
        # x is not normalized
        yield X

    if y is not None:
        y = (y - config.λ0)/config.Δ
        yield y    

    if args:
        yield from normalize(*args)

@listify
def denormalize(X=None, y=None, *args):
    if X is not None:
        # x is not normalized
        yield X

    if y is not None:
        y = y*config.Δ + config.λ0
        yield y    
        
    if args:
        yield from normalize(*args)