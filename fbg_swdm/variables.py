import numpy as np
from math import pi as π
from math import sqrt
from sys import modules
from os import makedirs
from os.path import join
from json import dumps, JSONEncoder
from datetime import datetime

_module = modules[__name__]

base_dir = ''
exp_name = 'base_exp'
exp_dir = join(base_dir, exp_name)

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def setattrs(**kwargs):
    """Set multiple attributes of module from dictionary"""
    for k, v in kwargs.items():
        setattr(_module, k, v)
        if k == 'exp_name':
            setattr(_module, 'exp_dir', join(base_dir, exp_name))
            makedirs(exp_dir, exist_ok=True)
    kwargs.pop("λ", None)
    with open(exp_dir+'\\log.txt','a') as file:
        file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S\n"))
        file.write(dumps(kwargs, indent=4, ensure_ascii=False, cls=NumpyEncoder))
        file.write('\n')

figsize = (16, 12)
dpi = 216

# prefixes
μ = 10**-6  # micro
n = 10**-9  # nano
p = 10**-12  # pico

topology='parallel'

N = 300  # number of spectral sampling points
M = 10000  # numbers of sampling points of test sweep

# FBG Characterization
Q = 2  # Number of FBGs
A = np.array([1, 0.5])  # Atenuation
Δλ = np.array([0.2*n, 0.2*n])  # Linewidth
λ0 = 1550*n  # Central Wavelength
S = np.array([1,1])  # fbg strength
#S = κ*L in [1,3] ranging from saturated to strong grating

Δ = 2*n  # Range of Wavelength change
λ = np.linspace(λ0 - Δ, λ0 + Δ, N)  # Wavelength
bounds = (λ0 - Δ, λ0 + Δ)

# fiber characteristics
a = 6*μ  # core radius
n1 = 1.48  # core n
n2 = 1.478  # cladding n

# mode properties
V = (2*π/λ0)*a*sqrt(n1**2-n2**2)
try: 
    # normalized frequency LP01
    from ofiber import LP_mode_value, LP_clad_irradiance, LP_total_irradiance
    ell = 0
    em = 1
    b = LP_mode_value(V, ell, em)
    clad = LP_clad_irradiance(V, b, ell)
    total = LP_total_irradiance(V, b, ell)
    eta = clad/total
except ImportError:
    # normalized frequency LP01 approximation
    b = (1.1428-0.9960/V)**2
    eta = 1-1/V**2  # Portion of power in core
n_eff = n2 + b*(n1-n2)  # effective refractive index


φN = 10 # Number of points for phase sweep