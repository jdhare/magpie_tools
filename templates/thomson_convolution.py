%matplotlib inline
from spectral_density_functions import S_k_omega, S_k_omega_conv
import numpy as np
import matplotlib.pyplot as plt

def add_points_evenly(initial_array, scale):
    return np.linspace(initial_array[0], initial_array[-1], initial_array.size*scale-scale+1)

lambda_in=532e-9#nm
bandwidth=1e-10#nm change this to zoom in on the spectrum
delta_lambda=9.78e-12
lambda_range=np.arange(lambda_in-bandwidth, lambda_in+bandwidth, delta_lambda)
shift=lambda_range-lambda_in

theta=45 #degrees
A=183#atomic mass for carbon
T_e=300#eV
T_i=100#eV
Z=5#guess, or find an nLTE model to give consistent Z for T_e and n_e
n_e=1e18#cm^-3

response_width=1e-11#0.4 Angstroms, appropriate for our finest grating
response=np.exp(-0.5*(shift/response_width)**2)

iscale=100
l_i=add_points_evenly(lambda_range,iscale)
shift_i=l_i-lambda_in
response_i=np.exp(-0.5*(shift_i/response_width)**2)


Skw=S_k_omega(lambda_range, lambda_in, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0)
Skw_conv=S_k_omega_conv(lambda_range, lambda_in, response, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0)


Skw_i=S_k_omega(l_i, lambda_in, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0)
Skw_conv_i=S_k_omega_conv(l_i, lambda_in, response_i, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0)

fig, ax=plt.subplots()
ax.plot(shift_i, Skw_i, c='orange')
ax.plot(shift_i, Skw_conv_i, c='g')

ax.plot(shift, Skw, ls=':', c='orange')
ax.plot(shift, Skw_conv, ls=':', c='g')

ax.plot(shift, Skw_conv_i[::iscale], c='b')