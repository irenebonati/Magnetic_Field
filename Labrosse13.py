#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:48:43 2018

@author: irenebonati
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

year = 365.25*3600*24

# ----------------------------- Model constants ----------------------------- #

rho_c_E         =  12.5e3          # Density at center (kgm-3)
C_p_E           =  850             # Specific heat (Jkg-1K-1)
alpha_c_E       =  1.3e-5          # Thermal expansion coefficient at center (K-1)
gamma_E         =  1.5             # Gruneisen parameter
L_rho_E         =  7400e3          # Density length scale (m)
L_T_E           =  6042e3          # Temperature length scale (m) 
G               =  6.67408e-11     # Gravitational constant (m3kg-1s-2)
b_E             =  3480e3          # Core radius (m)
DeltaS_E        =  118             # Entropy of crystallization (Jkg-1K-1)
T_s0_E          =  5270            # Solidification temperature at center (K,Labrosse 2001)
c_E             =  1221e3          # Present inner core radius (m) --> equal to c
rho_0_E         =  7.5e3           # Density at 0 pressure (kgm-3)
K_0_E           =  1403e9           # Incompressibility at surface (Tachinami+2011, Table 1)
rho_oc_E        =  10e3            # Density outer core (kgm-3)
DeltaS_E        =  118             # Entropy of crystallization (Jkg-1K-1)
Ts_ICB_E        =  5600            # Present temperature at ICB (K)
Deltarho_ICB_E  =  500             # Density jump at ICB (kgm-3)

# -----------------------------------------------------------------------------

# Secular cooling of the core
def calc_Pc(H,rho_c,c,L_T,I,C_p,T_sol0,gamma):
    return 4.*np.pi*H**3.*rho_c*C_p*T_sol0*(1.-2./3./gamma)*(c/L_T**2)*np.exp((2./3./gamma-1.)*c**2./(L_T**2.))*I
# c is ICB radius!!
    
# Length scale
def calc_H(L_rho,L_T):
    return np.sqrt(1./((1./L_rho**2.)+(1./L_T**2.)))

# Temperature length scale
def calc_L_T(C_p,alpha_c,rho_c):
    return np.sqrt(3.*C_p/(2.*np.pi*alpha_c*rho_c*G))

# Density length scale
def calc_L_rho(K_0,rho_0,rho_c):
    return np.sqrt(3.*K_0/(2.*np.pi*G*rho_0*rho_c)*(np.log(rho_c/rho_0)+1.))

# Function coming from integration of r2exp(-r2/H2)
def calc_I(H,b):
    return np.sqrt(np.pi)/2.*erf(b/H)-b/H*np.exp(-b**2./H**2.)
# b is core radius!!

# -----------------------------------------------------------------------------

# Latent heat of the core
    
# T_sol: solidus temperature at ICB (Earth: 5600 K)
# r_ICB: radius of ICB (obtained from Lena)
def calc_P_L(r_ICB,rho_ICB,T_sol_ICB,DeltaS):    
    return 4*np.pi*r_ICB**2*rho_ICB*T_sol_ICB*DeltaS

# As we consider planets of different mass, we calculate T_sol_ICB using Lindemann's law
# T_sol0: solidification of core material at pressure at the center of the planet
# Stixrude, 2014
def calc_T_sol(gamma,r,L_T,T_sol0):
    return T_sol0*np.exp(-2*(1-1./3./gamma)*r**2/L_T**2)

# Density profile
def calc_rho(rho_c,r,L_rho):
    return rho_c*np.exp(-r**2/L_rho**2)

# -----------------------------------------------------------------------------

# Gravitational energy

# rho_c is the density at center (not core density)
def calc_P_G(Delta_rho,rho_c,c,b):
    return 8*np.pi**2/3.*G*Delta_rho*rho_c*c**2*b**2*(3./5.-c**2/b**2)

def calc_Delta_rho(rho_c,rho_oc):
    return rho_c-rho_oc

# -----------------------------------------------------------------------------    
#                             Example for Earth   
# -----------------------------------------------------------------------------    

# ------------------------------ P_c=P_c(c) -----------------------------------

L_T_Earth = calc_L_T(C_p_E,alpha_c_E,rho_c_E)

L_rho_Earth = calc_L_rho(K_0_E,rho_0_E,rho_c_E)

H_Earth = calc_H(L_rho_Earth,L_T_Earth)

I_Earth = calc_I(H_Earth,b_E)

P_c_Earth = calc_Pc(H_Earth,rho_c_E,c_E,L_T_Earth,I_Earth,C_p_E,T_s0_E,gamma_E)

# ------------------------------ P_g=P_g(c) -----------------------------------

#Delta_rho_Earth = calc_Delta_rho(rho_c_E,rho_oc_E)

P_G_Earth = calc_P_G(Deltarho_ICB_E,rho_c_E,c_E,b_E)

# ------------------------------ P_L=P_L(c) -----------------------------------

Ts_ICB_Earth = calc_T_sol(gamma_E,c_E,L_T_Earth,T_s0_E)

rho_ICB_Earth = calc_rho(rho_c_E,c_E,L_rho_E)

P_L_Earth = calc_P_L(c_E,rho_ICB_Earth,Ts_ICB_E,DeltaS_E) 

E_G_Earth = P_G_Earth*c_E
E_L_Earth = P_L_Earth*c_E
E_c_Earth = P_c_Earth*c_E


total_energy = (P_G_Earth + P_L_Earth + P_c_Earth)*c_E



























 
