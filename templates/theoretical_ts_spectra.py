# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:17:45 2018

@author: jdhare
"""

from spectral_density_functions import S_k_omega, S_k_omega_conv
import numpy as np
import matplotlib.pyplot as plt

lambda_in=532e-9#nm
bandwidth=5e-9#nm change this to zoom in on the spectrum
lambda_range=np.linspace(lambda_in-bandwidth, lambda_in+bandwidth, 1000)
shift=lambda_range-lambda_in

theta=90 #degrees
A=12#atomic mass for carbon
T_e=1#eV
T_i=1#eV
Z=1#guess, or find an nLTE model to give consistent Z for T_e and n_e
n_e=1e17#cm^-3

response_width=4e-11#0.4 Angstroms, appropriate for our finest grating
response=np.exp(-0.5*(shift/response_width)**2)

Skw=S_k_omega(lambda_range, lambda_in, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0)
Skw_conv=S_k_omega_conv(lambda_range, lambda_in, response, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0)

fig, ax=plt.subplots()
ax.plot(shift, Skw)
ax.plot(shift, Skw_conv)