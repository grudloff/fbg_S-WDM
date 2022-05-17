from warnings import WarningMessage
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
tag = None

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def set_base_dir(dir):
    # Set dir as base directory
    global base_dir
    base_dir = dir

def setattrs(**kwargs):
    """Set multiple attributes of module from dictionary"""
    for k, v in kwargs.items():
        setattr(_module, k, v)
        if k == 'exp_name':
            global exp_dir
            exp_dir = join(base_dir, exp_name)
            makedirs(exp_dir, exist_ok=True)
        elif k=="N":
            global λ
            λ = np.linspace(λ0 - Δ, λ0 + Δ, N)
        if k =='Q':
            variable = ["A", "Δλ", "I", "Δn_dc"]
            for i in range(len(variable)):
                var = getattr(_module, variable[i])
                if len(var) < Q:
                    setattr(_module, variable[i], np.resize(var, Q))
        if k == 'portion':
            global bounds
            bounds = (λ0 - portion*Δ, λ0 + portion*Δ)
        if k in ['λ0', 'a', 'n1', 'n2']:
            set_mode_properties()
    with open(exp_dir+'\\log.txt','a', encoding="utf-8") as file:
        file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S\n"))
        file.write(dumps(kwargs, indent=4, ensure_ascii=False, cls=NumpyEncoder))
        file.write('\n')

figsize = (8, 6)
dpi = 216
pre_test=False

# prefixes
μ = 10**-6  # micro
n = 10**-9  # nano
p = 10**-12  # pico

# topology of fbg array
topology = 'parallel'
# type of simulation {'true', 'gaussian'}
simulation = 'true'

N = 300  # number of spectral sampling points
M = 10000  # numbers of sampling points of test sweep

# FBG Characterization
Q = 2  # Number of FBGs
A = np.array([1, 0.5])  # Attenuation
Δλ = np.array([0.2*n, 0.2*n])  # Linewidth
λ0 = 1550*n  # Central Wavelength
I = np.array([0.9, 0.9])  # peak reflectivities
I_max = 0.99
#S = κ*L in [1,3] ranging from saturated to strong grating
Δn_dc = np.array([0, 0])

Δ = 2*n  # Range of Wavelength change
λ = np.linspace(λ0 - Δ, λ0 + Δ, N)  # Wavelength
portion = 0.6 # portion of Δ 
bounds = (λ0 - portion*Δ, λ0 + portion*Δ)

# fiber characteristics
a = 6*μ  # core radius
n1 = 1.48  # core n
n2 = 1.478  # cladding n

try: 
    def set_mode_properties():
        global b, η, V, n_eff
        V = (2*π/λ0)*a*sqrt(n1**2-n2**2)
        # normalized frequency LP01
        from ofiber import LP_mode_value, LP_core_irradiance, LP_total_irradiance
        ell = 0
        em = 1
        b = LP_mode_value(V, ell, em)
        # Power fraction in core
        core = LP_core_irradiance(V, b, ell)
        total = LP_total_irradiance(V, b, ell)
        η = core/total
        # effective refractive index
        n_eff = n2 + b*(n1-n2) 
except ImportError:
    # if ofiber is not installed use approximation
    def set_fiber_properties():
        global b, η, V, n_eff
        V = (2*π/λ0)*a*sqrt(n1**2-n2**2)
        # normalized frequency LP01
        b = (1.1428-0.9960/V)**2
        if V<1:
            raise WarningMessage("Core power fraction approximation isn't valid for V<1")
        # power fraction in core
        η = 1-1/V**2
        # effective refractive index
        n_eff = n2 + b*(n1-n2) 

set_mode_properties()

φN = 10 # Number of points for phase sweep