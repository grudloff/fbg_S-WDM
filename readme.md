# Summary

A modulation in the refraction index of the core of an optical fiber generates a structure known as Fiber Bragg Grating (FBG), which behaves as a mirror for a narrow optical bandwidth. The reflection spectral profile depends on the index modulation profile. In consequence, if the modulation is modified by external factors, the reflection spectral profile is also affected. In particular, a change in the fiber temperature or axial strain induces a change in the length and refraction index of the FBG, which in turn translates into a lineal shift on the spectral profile's position. Leveraging this behavior, an FBG can be easily used as a sensor by measuring this spectral shift. In addition, an array of FBGs, inscribed in cascade along the optical fiber, can be used to construct a quasi-distributed sensor.

The main multiplexing methods of such an array are *(i)* in wavelength, by positioning each FBG at a different spectral location to avoid spectral overlapping, and *(ii)* in time by positioning each FBG with sufficient distance to avoid temporal overlapping of the multiple interrogating pulse reflections. In wavelength domain there is a limitation in the number of FBGs that can be multiplexed given by the source bandwidth and the spectral width of each FBG. This limitation can be partially overcome by allowing spectral overlap and using more expensive signal processing schemes, which allow for adequate demultiplexation as long as there is sufficient difference between the spectra to discern each FBG sensor. This is known as Spectrally overlaped WDM (S-WDM).

This repository intends to give the foundations for a fair comparison of most methods proposed currently in the literature, by providing the tools for simulation and performance visualization as well as comprehensive implementations of methods with no current public python implementations.

# Problem Setting

Conventionally a WDM FBG network requires no overlap on the FBGs spectra for conventional peak detection (CPD) techniques to have adequate results (for a review on these refer to \cite{tosi}). If we consider a parallel topology (where overlapping FBG are on parallel optical fibers) we only have to take into account the crosstalk originated from the observed spectra being the sum of each original spectra. Alternatively, if we consider the serial topology we have to take into account two kinds of cross-talk, typically linked to TDM, namely spectral-shadowing and multiple-reflection crosstalk \cite{demux_lim}. For this reason, spectrally-overlapped WDM is usually implemented in parallel topologies.

The FBG spectral profile characterized by equation \ref{eq:fbg}, can be reasonably approximated to a Gaussian shape \cite{gauss}, which is characterized as

<!-- \begin{equation}
\label{eqn:apo_fbg}
R_i(\lambda) = I_{i}e^{-4 ln(2)\left[\frac{\lambda - \lambda_{Bi}}{\Delta\lambda_{Bi}}\right]^2}
\end{equation} -->

![R_i(\lambda) = I_{i}e^{-4 ln(2)\left[\frac{\lambda - \lambda_{Bi}}{\Delta\lambda_{Bi}}\right]^2}](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+R_i%28%5Clambda%29+%3D+I_%7Bi%7De%5E%7B-4+ln%282%29%5Cleft%5B%5Cfrac%7B%5Clambda+-+%5Clambda_%7BBi%7D%7D%7B%5CDelta%5Clambda_%7BBi%7D%7D%5Cright%5D%5E2%7D)

Where *I<sub>i</sub>* is the peak reflectance, *λ<sub>Bi</sub>* is the Bragg's wavelength and *Δλ<sub>Bi</sub>* is the FWHM bandwidth. This approximation is depicted in figure \ref{fig:FBG_spectra} for the matching values of *I<sub>i</sub>*, *λ<sub>Bi</sub>* and *Δλ<sub>Bi</sub>*.

For the parallel WDM FBG setup, the problem consists in finding the spectral shifts of each sensor even when there is overlapping of their spectra, given that there is sufficient difference between the spectral profiles either in I<sub>i</sub>, in Δλ<sub>Bi</sub>, or in both. We can formulate the problem in a more formal mathematical matter as follows. Given an array of size *N* with spectral profiles characterized by equation \ref{eqn:apo_fbg} with parameters *λ<sub>Bi</sub>*, *I<sub>i</sub>* and *Δλ<sub>Bi</sub>*, we wish to obtain an estimation ![\hat{y}](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Chat%7By%7D)
 of ![y = \{\lambda_{B_i}\}_{i\in[1,...,N]}](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+y+%3D+%5C%7B%5Clambda_%7BB_i%7D%5C%7D_%7Bi%5Cin%5B1%2C...%2CN%5D%7D) (with known *I<sub>i</sub>* and *Δλ<sub>Bi</sub>* for each FBG) from *x*, a discrete observation of *M* points over *λ \in [λ<sub>0</sub> - Δλ/2, λ<sub>0</sub>+Δλ/2* of the observed spectrum *X(λ)* characterized as

<!-- \begin{equation}
    X(λ) = \sum_{i=1}^N R_i(λ)
\end{equation} -->

![formula](https://render.githubusercontent.com/render/math?math=X(\lambda)%20=%20\sum_{i=1}^N%20R_i(\lambda))

# Methods

One approach to retrieve the peak location of overlapping FBGs is through evolutionary algorithms (EA). Given that we have a good approximation of the discrete spectrum ![\hat{x} = f(\hat{y})](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Chat%7Bx%7D+%3D+f%28%5Chat%7By%7D%29), we wish to find ![\hat{y} ](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Chat%7By%7D+) that yields an approximated discrete spectrum ![\hat{x} ](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Chat%7Bx%7D+) similar to the observed discrete spectrum *x*, generally quantified by the ![\ell_1](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cell_1) or ![\ell_2](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cell_2) norm. The optimization problem can be formulated as
![\hat{y} = \min_{\hat{y}} \mathcal{L}(x, \hat{x})](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Chat%7By%7D+%3D+%5Cmin_%7B%5Chat%7By%7D%7D+%5Cmathcal%7BL%7D%28x%2C+%5Chat%7Bx%7D%29)
where ![\mathcal{L}(\cdot, \cdot) ](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cmathcal%7BL%7D%28%5Ccdot%2C+%5Ccdot%29+) is a loss function that quantifies the dissimilarity between its inputs.

The other approach to solve this problem is to build a regression model \cite{stat_learn}. In this approach we wish to invert the function *x=f(y)* that generates the observed spectrum. Since it is not possible to find an analytical solution to the problem, we can find an approximation given by a parameterized function ![g_\theta(x)=\hat{y}\approx y = f^{-1}(x)](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+g_%5Ctheta%28x%29%3D%5Chat%7By%7D%5Capprox+y+%3D+f%5E%7B-1%7D%28x%29), characterized by the parameter vector ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Ctheta). The optimal vector ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Ctheta) is found by minimizing a loss function over an *n*-sized paired dataset ![Y=\{y_i\}_{i\in[1,n]}, X=\{x_i\}_{i\in[1,n]}](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+Y%3D%5C%7By_i%5C%7D_%7Bi%5Cin%5B1%2Cn%5D%7D%2C+X%3D%5C%7Bx_i%5C%7D_%7Bi%5Cin%5B1%2Cn%5D%7D),  where *x<sub>i</sub>=f(y<sub>i</sub>)*. The loss function quantifies the difference between the predictions and the actual values, usually through the ![\ell_1](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cell_1) or ![\ell_2](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cell_2) norm. This is formalized as
<!-- $$\theta = \min_{\theta} \mathcal{L}(Y, g_\theta(X))$$ -->


![formula](https://render.githubusercontent.com/render/math?math=\theta%20=%20\min_{\theta}%20\mathcal{L}(Y,%20g_\theta(X)))

This optimization problem can be solved either through gradient descend or by finding an analytical solution under certain formulations. This approach has lower accuracy compared to EA but has an inference time that is significantly shorter since there is a previous learning step.

# Repository Structure
## General Algorithm Syntax
* Each algorithm is a class that holds hyperparameters
* Trainable algorithms (Regressions) have a `fit` method for parameter learning

```python
model = alg.fit(X_train, y_train)
```

* All algorithms have a `predict` method for prediction
```python
y_test_hat = alg.predict(X_test)
```
## Repo Files
* simulation.py: Functions for data generation
* variables.py: FBG array parameters for data generation and model construction
* visualization.py: Functions for various visualization tasks
* regresors.py: Classes for regresion algorithms
* evolutionary.py: Classes for evolutionary algorithms

# References

## Used Repositories

[1] Least Squares Support Vector Regression
https://github.com/zealberth/lssvr

[2]LEAP: Evolutionary Algorithms in Python
    Written by Dr. Jeffrey K. Bassett, Dr. Mark Coletti, and Eric Scott
    https://github.com/AureumChaos/LEAP

[3] Scikit-Learn: Machine Learning in {P}ython
https://github.com/scikit-learn/scikit-learn

[4] Array programming with {NumPy}
https://github.com/numpy/numpy

## Papers 

\bibitem{othonos}
A. Othonos, K. Kalli, D. Pureur, and A. Mugnier, “Fibre Bragg Gratings,” in Springer Series in Optical Sciences, Springer Berlin Heidelberg, pp. 189–269.

\bibitem{raman}
R. Kashyap, Fiber Bragg gratings, 2nd ed. San Diego, CA: Academic Press, 2009.

\bibitem{morey} W. W. Morey, J. R. Dunphy, and G. Meltz, “Multiplexing fiber bragg grating sensors,” Fiber and Integrated Optics, vol. 10, no. 4, pp. 351–360, Oct. 1991, doi: 10.1080/01468039108201715.

\bibitem{kersey} A. D. Kersey et al., “Fiber grating sensors,” J. Lightwave Technol., vol. 15, no. 8, pp. 1442–1463, 1997, doi: 10.1109/50.618377.

\bibitem{cdma}
K. P. Koo, A. B. Tveten, and S. T. Vohra, “Dense wavelength division multiplexing of fibre Bragg grating sensors using CDMA,” Electron. Lett., vol. 35, no. 2, p. 165, 1999, doi: 10.1049/el:19990135.

\bibitem{demux_lim}
M. Fajkus et al., “Capacity of Wavelength and Time Division Multiplexing for Quasi-Distributed Measurement Using Fiber Bragg Gratings,” AEEE, vol. 13, no. 5, Dec. 2015, doi: 10.15598/aeee.v13i5.1508.

\bibitem{tosi}
D. Tosi, “Review and Analysis of Peak Tracking Techniques for Fiber Bragg Grating Sensors,” Sensors, vol. 17, no. 10, p. 2368, Oct. 2017, doi: 10.3390/s17102368.

\bibitem{gauss}
V. Mizrahi and J. E. Sipe, “Optical properties of photosensitive fiber phase gratings,” J. Lightwave Technol., vol. 11, no. 10, pp. 1513–1517, 1993, doi: 10.1109/50.249888.

\bibitem{genetic_algo}
C. Z. Shi, C. C. Chan, W. Jin, Y. B. Liao, Y. Zhou, and M. S. Demokan, “Improving the performance of a FBG sensor network using a genetic algorithm,” Sensors and Actuators A: Physical, vol. 107, no. 1, pp. 57–61, Oct. 2003, doi: 10.1016/s0924-4247(03)00323-6.
\bibitem{swap_differential_evolution}
H. Jiang, J. Chen, T. Liu, and W. Huang, “A novel wavelength detection technique of overlapping spectra in the serial WDM FBG sensor network,” Sensors and Actuators A: Physical, vol. 198, pp. 31–34, Aug. 2013, doi: 10.1016/j.sna.2013.04.023

\bibitem{DMS_PSO}
G. Guo, D. Hackney, M. Pankow, and K. Peters, “Interrogation of a spectral profile division multiplexed FBG sensor network using a modified particle swarm optimization method,” Meas. Sci. Technol., vol. 28, no. 5, p. 055204, Mar. 2017, doi: 10.1088/1361-6501/aa637f.

\bibitem{distributed_estimation_algorithm}
Q.-X. Zhou, H. Jiang, Y.-T. Lin, and J. Chen, “Application of Distributed Estimation Algorithm in Wavelength Demodulation of Overlapping Spectra of Fiber Bragg Grating Sensor Networks,” presented at the 2019 Chinese Control Conference (CCC), Jul. 2019, doi: 10.23919/chicc.2019.8866493

\bibitem{elm}
Hao Jiang, Jing Chen, and Tundong Liu, “Wavelength Detection in Spectrally Overlapped FBG Sensor Network Using Extreme Learning Machine,” IEEE Photon. Technol. Lett., vol. 26, no. 20, pp. 2031–2034, Oct. 2014, doi: 10.1109/lpt.2014.2345062.

\bibitem{LS_SVM}
J. Chen, H. Jiang, T. Liu, and X. Fu, “Wavelength detection in FBG sensor networks using least squares support vector regression,” J. Opt., vol. 16, no. 4, p. 045402, Mar. 2014, doi: 10.1088/2040-8978/16/4/045402.

\bibitem{GPR}
M. S. E. Djurhuus, S. Werzinger, B. Schmauss, A. T. Clausen, and D. Zibar, “Machine Learning Assisted Fiber Bragg Grating-Based Temperature Sensing,” IEEE Photon. Technol. Lett., vol. 31, no. 12, pp. 939–942, Jun. 2019, doi: 10.1109/lpt.2019.2913992.

\bibitem{CNN}
B. Li et al., “Robust Convolutional Neural Network Model for Wavelength Detection in Overlapping Fiber Bragg Grating Sensor Network,” presented at the Optical Fiber Communication Conference, 2020, doi: 10.1364/ofc.2020.t4b.5.

\bibitem{stat_learn}
T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning. New York, NY, USA: Springer New York Inc., 2001.
