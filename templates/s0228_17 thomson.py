# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:48:33 2017

@author: jdhare
"""

import ts_analysis as TsA
import os

#%%

folder=r"\\LINNA\Users\Magpie\Documents\MAGPIE data\2017\Feb_2017\s0228_17 Reconnection Al - Kz TS\s0228_17 TS"
shot_fn='shot/s0228_17 shot.asc'
ts=TsA.TS_Analysis(folder, shot_fn,['bk/s0215_17 bundle A in beam.asc','bk/s0224_17 bundle B in beam.asc'])
#spacing 17.5, offset 9

ts.find_fibre_edges()

#%%
'''Split image into fibres'''
ts.split_into_fibres(discard_rows=0)
ts.zero_fibres(upper=1150, lower=850)
a_angle=[90]*14 #here we have fibres 1A through 14A at an angle of 45'
b_angle=[120]*14
ts.pair_fibres(a_angle,b_angle)

#%%
'''
Specify the plasma parameters in the form (V
alues, Minimum, Independent)
If Independent is True, then the value is fixed and minimum has no meaning
If Independent is False, then the value is an initial guess for the fit
and minimum is either a float, or None to specify no given minimum
For multi-species fits, Aj, Zj and Fj only can have their value as a tuple (A1,A2...Aj)
Even for a single-species fit, enclose A, Z and F in (), eg. 'Aj':((12), True, None)
'''
'''Choose the fibre to fit'''
Fnum=9
Fset='A'

plasma_parameters={ 'model': 'nLTE',
                    'n_e': (1e18,True),#in cm^-3, must be specified
                    'T_e': (10, True,1),#In electron volts. 
                    'V_fe':(0, True),#m/s.
                    'A':  (183, True),#in nucleon masses, must be specified
                    'T_i': (700, False,1),#In electron volts.
                    'V_fi':(0, False),#m/s.
                    'stray':(0, True,1), #Fraction of signal made up of stray light. >0, <1.0!!
                    'amplitude': (100000, False), #Fraction of signal made up of spectrum.
                    'offset': (120, False), #
                    'shift': (0, True)
                    }

'''This code fits and plots the result'''
f=ts.select_fibre(Fnum,Fset)
f.voigt_response()
f.symmetric_crop_around_l0()
#%%
f.fit_fibre(plasma_parameters)
#ts.pretty_plot(Fnum,Fset, tm=1.6, sr=6)
print(f.skw_res.fit_report())
for i in np.arange(0,14):
    f=ts.select_fibre(i,'A')
    f.voigt_response()
    f.symmetric_crop_around_l0()
    f.calculate_integrated_intensity()
    print(i+1,'\t',np.sum(f.shot),'\t',f.params['integrated_intensity'])
    