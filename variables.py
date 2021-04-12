import numpy as np

figsize = (10, 5)
n = 10**-9 #nano
p = 10**-12

N = 300
M = 10000
test_ratio = 0.1

#FBG Characterization
FBGN = 2 #Number of FBGs
I = np.array([1, 0.5]) #Peak Intensities
dA = np.array([0.2*n, 0.2*n]) #Linewidth
A0 = 1550*n #Central Wavelength
D = 2*n #Range of Wavelength change
A = np.linspace(A0-D, A0+D, N); #Wavelength
bounds = (A0 - D, A0 + D)