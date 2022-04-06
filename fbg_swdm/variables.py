import numpy as np
from math import pi as π
from math import sqrt

figsize = (16, 12)
dpi = 216

# prefixes
μ = 10**-6  # micro
n = 10**-9  # nano
p = 10**-12  # pico

topology='parallel'

N = 300  # number of spectral sampling points
M = 10000  # numbers of sampling points of test sweep
test_M = 1000

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
    M_p = clad/total
except ImportError:
    # normalized frequency LP01 approximation
    b = (1.1428-0.9960/V)**2
    M_p = 1-1/V**2  # Portion of power in core
n_eff = n2 + b*(n1-n2)  # effective refractive index


φN = 10 # Number of points for phase sweep