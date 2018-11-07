#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:26:26 2018

@author: irenebonati
"""

# Labrosse 2003

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.integrate as scODE


year = 365.25*3600*24

# ----------------------------- Model constants ----------------------------- #

alpha_c_E       =  1.3e-5          # Thermal expansion coefficient at center (K-1)
b_E             =  3480e3          # Core radius (m)
c_E             =  1221e3          # Present inner core radius (m) --> same as c
CP_E            =  750             # Specific heat (Jkg-1K-1)            
Deltarho_ICB_E  =  500             # Density jump at ICB (kgm-3)
DeltaS_E        =  118             # Entropy of crystallization (Jkg-1K-1)
DTS_DTAD        =  1.65            # Adiabatic temperature gradient (K/m)
gamma_E         =  1.5             # Gruneisen parameter
G               =  6.67408e-11     # Gravitational constant (m3kg-1s-2)
H               =  0               # Radioactivity
K_c_E           =  1403e9          # Bulk modulus at center (Pa) (Labrosse+2015)
Kprime_c_E      =  3.567e9         # Pa
k_c_E           =  163             # Thermal conductivity at center (Wm-1K-1)
L_rho_E         =  7400e3          # Density length scale (m)
L_T_E           =  6042e3          # Temperature length scale (m) 
Q_CMB           =  7.4e12          # Today's CMB flux (W) 
rho_c_E         =  12451           # Density at center (kgm-3)
rho_oc_E        =  10e3            # Density outer core (kgm-3)
rho_0_E         =  7.5e3           # Density at 0 pressure (kgm-3)
T_s0_E          =  5270            # Solidification temperature at center (K,Labrosse 2001)
Ts_ICB_E        =  5600            # Present temperature at ICB (K)

# -----------------------------------------------------------------------------

def calc_g(r,rho_c,L_rho,A_rho): 
    """Gravity"""
    return 4. * np.pi / 3. * G * rho_c * r * (1 - 3. / 5. * r**2. / L_rho**2. - 3. * A_rho / 7. * r**4. / L_rho**4.)

def calc_kappa(k,rho,CP):
    '''Thermal diffusivity'''
    return k/rho/CP

# Density length scale
def calc_L_rho(K_c,rho_c):
    '''Density length scale'''
    return np.sqrt(3. * K_c / (2. * np.pi * G * rho_c**2.))#*(np.log(rho_c/rho_0)+1.))

'''Density length scale for Earth'''
L_rho_Earth = calc_L_rho(K_c_E,rho_c_E)

def calc_A_rho(Kprime_0):
    return (5. * Kprime_0 - 13.) / 10.

'''A_rho for Earth'''
A_rho_Earth = calc_A_rho(Kprime_c_E)


def calc_rho(r,L_rho,A_rho,rho_c):
    """ Density"""
    return rho_c * (1. - r**2. / L_rho**2. - A_rho * r**4. / L_rho**4.)



















