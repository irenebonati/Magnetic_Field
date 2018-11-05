#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 18:01:29 2018

@author: irenebonati
"""

import numpy as np
import matplotlib.pyplot as plt

YEAR = 365*24*3600
sig = 5.67e-8

def equilibrium_temperature(S):
    a = 0
    return (S*(1-a)/(4*sig))**0.25
def radiated_flux(T, R):
    return sig*(4*np.pi*R**2)*T**4
def cmb_flux(Qs, urey):
    return Qs*urey

RE = 6371e3   # Planet radius
RC = 0.35*RE  # Core radius
RHOC = 1e4    # Core density
CP = 1e3      # Heat capacity
E0 = (4*np.pi*RC**3)/3*RHOC*CP*100   # MF
S0 = 1361     # Solar constant
urey = 0.5    # Urey ratio

distance = np.linspace(1, 30, 500)
atmosphere = np.linspace(1, 10, 500).reshape(500, 1)

X, Y = np.meshgrid(distance, atmosphere)

S = S0*(1/distance)**2
Ts = equilibrium_temperature(S)

Qo = np.multiply(radiated_flux(Ts, RE), 1/atmosphere) # Q_out

Qc = cmb_flux(Qo, urey) # Q_core

Qc[np.where(Y > 5+5*(X/30)**3)]= 1e99
Qc[np.where(Y > 9)] = 2e13

plt.contourf(X, Y, E0/Qc/(1e6*YEAR))
plt.colorbar()