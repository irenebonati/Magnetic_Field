#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 01:35:00 2018

@author: irenebonati
"""

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
import scipy.integrate as integrate


year = 365.25*3600*24

# ----------------------------- Model constants ----------------------------- #

# These parameters are for the Earth! Used to test the code

alpha_c       =  1.3e-5          # Thermal expansion coefficient at center (K-1)
r_OC          =  3480e3          # Core radius (m)
r_IC          =  1221e3          # Present inner core radius (m) --> same as c
CP            =  750             # Specific heat (Jkg-1K-1)            
Deltarho_ICB  =  500             # Density jump at ICB (kgm-3)
DeltaS        =  127             # Entropy of crystallization (Jkg-1K-1)
DTS_DTAD      =  1.65            # Adiabatic temperature gradient (K/m)
gamma         =  1.5             # Grueneisen parameter
GC             =  6.67384e-11     # Gravitational constant (m3kg-1s-2)
H             =  0               # Radioactivity
K_c           =  1403e9          # Bulk modulus at center (Pa) (Labrosse+2015)
Kprime_c      =  3.567           # unitless
k_c           =  163             # Thermal conductivity at center (Wm-1K-1)
#L_rho         =  7400e3         # Density length scale (m)
L_T           =  6042e3          # Temperature length scale (m) 
Q_CMB         =  7.4e12          # Today's CMB flux (W) 
rho_c         =  12502           # Density at center (kgm-3)
rho_oc        =  10e3            # Density outer core (kgm-3)
rho_0         =  7.5e3           # Density at 0 pressure (kgm-3)
T_s0          =  5270            # Solidification temperature at center (K,Labrosse 2001)
Ts_ICB        =  5600            # Present temperature at ICB (K)
beta          =  0.83
dTL_dchi      =  -21e3           # Compositional dependence of liquidus temperature (K)
dTL_dP        =  9E-9            # Pressure dependence of liquidus temperature (Pa-1) 
chi0          =  0.056           # Difference in mass fraction of light elements across ICB
TL0           =  5700            # Melting temperature at center (K)

# -----------------------------------------------------------------------------

'''Density length scale for Earth'''
L_rho = np.sqrt(3. * K_c / (2. * np.pi * GC * rho_c**2.))


def A_rhox(Kprime_0):
    return (5. * Kprime_0 - 13.) / 10.

'''A_rho for Earth'''
A_rho = A_rhox(Kprime_c)

def dTL_dr_IC(r):#,L_rho):#,K_c,gamma,TL,dTL_dchi,chi0,L_rho,fC,r_OC):
    ''' Melting temperature jump at ICB '''
    #dTL_dP=2.*(gamma-1./3.)*T_melt/K_c
    return -K_c * 2.*dTL_dP * r / L_rho**2. \
      + 3. * dTL_dchi * chi0 * r**2. / (L_rho**3. * fC(r_OC / L_rho, 0.))

def fC(r,gamma):#A_rho): 
    '''fC (Eq. A1 Labrosse 2016)'''
    return r**3. * (1 - 3. / 5. * (gamma + 1) * r**2.- 3. / 14. * (gamma + 1) \
           * (2 * A_rho - gamma) * r**4.)

def fX(r,r_IC):#,L_rho):#r_IC
    return (r)**3. * (-r_IC**2. / 3. / L_rho**2. + 1./5. * (1.+r_IC**2./L_rho**2.) \
            *(r)**2.-13./70. * (r)**4.) 

def rho(r):#,L_rho,A_rho): #rho_c
    ''' Density '''
    return rho_c * (1. - r**2. / L_rho**2. - A_rho * r**4. / L_rho**4.)

def T_melt(r):#,L_rho): #K_c,dTL_dP,dTL_dchi,chi0,fC,r_OC)
    ''' Melting temperature at ICB '''
    return TL0 - K_c * dTL_dP * r**2. / L_rho**2. + dTL_dchi * chi0 * r**3. \
            / (L_rho**3. * fC(r_OC / L_rho, 0.))

# ------------------------------- Powers  ------------------------------------

def PL(r):#,T_m,DeltaS):
    '''Latent heat power'''
    return 4. * np.pi * r**2. * T_melt(r) * rho(r) * DeltaS

def LH(r):
    LH, i = integrate.quad(PL, 0, r)
    return LH

def Pc(r):#,rho_c,CP,L_rho,A_rho,dTL_dr_IC,gamma,T_m):
    '''Secular cooling core'''
    return -4. * np.pi / 3. * rho_c * CP * L_rho**3. * (1 - r**2. / L_rho**2 \
            - A_rho* r**4. / L_rho**4.)**(-gamma) * (dTL_dr_IC(r) + 2. * gamma \
            * T_melt(r) * r / L_rho**2. *(1 + 2. * A_rho * r**2. / L_rho**2.) \
             /(1 - r**2. / L_rho**2. - A_rho * r**4. / L_rho**4.)) \
             * (fC(r_OC / L_rho, gamma) - fC(r_IC / L_rho, gamma))

def SC(r):
    SC, i = integrate.quad(Pc, 0, r)
    return SC

def Px(r):#,chi0,rho_c,beta,L_rho,fX):
    ''' Gravitational energy (Eq. A14)'''
    return 8 * np.pi**2 * chi0 * GC * rho_c**2 * beta * r**2. \
      * L_rho**2. / fC(r_OC / L_rho, 0) \
      * (fX(r_OC / L_rho, r) - fX(r / L_rho, r))

def GH(r):
    GH, i = integrate.quad(Px, 0, r)
    return GH

# -------------------------------- Energies --------------------------------- #
           
# CALCULATIONS FOR R+R_ICB (EARTG)


''' Latent heat '''
L = 4. * np.pi / 3. * rho_c * TL0 * DeltaS * r_IC**3. * (1 - 3. / 5. \
    * (1 + K_c / TL0 * dTL_dP) * r_IC**2. / L_rho**2. \
    + chi0 / (2 * fC(r_OC / L_rho, 0.) * TL0) * dTL_dchi * r_IC**3. / L_rho**3.)
print "Latent heat", L,"J"        


''' Secular cooling '''
C = 4. * np.pi / 3. * rho_c * CP * L_rho * r_IC**2 * fC(r_OC / L_rho, gamma)\
        * (dTL_dP * K_c - gamma * TL0 - dTL_dchi * chi0 / fC(r_OC / L_rho, 0.) * r_IC / L_rho)    
C,i = integrate.quad(Pc, 0, r_IC)
print "Secular cooling", C,"J"

''' Gravitational energy '''
G = 8 * np.pi**2. / 15. * chi0 * GC * rho_c**2. * beta * r_IC**3. * r_OC**5. / L_rho**3. \
    / fC(r_OC/L_rho,0)*(1. - r_IC**2 / r_OC**2 + 3. * r_IC**2. / 5. / L_rho**2. \
        - 13. * r_OC**2. / 14. / L_rho**2. + 5./18. * r_IC**3. * L_rho**2. /r_OC**5.)
print "Gravitational energy", G,"J"

# Total energy
E_tot = L + C + G
print "Total energy", E_tot,"J"


x =50

r_IC_vec = np.linspace(0,r_IC,x)
Q_cmb = np.linspace(3,10,4)*1e12   # From Labrosse+2001

L_H= np.zeros(x)
S_C = np.zeros(x)
G_H = np.zeros(x)
time = np.zeros(x)


for i in range(len(r_IC_vec)):
    L_H[i] = LH(r_IC_vec[i])
    S_C[i] = SC(r_IC_vec[i])
    G_H[i] = GH(r_IC_vec[i])

E_tot = L_H + S_C + G_H

time = (L_H + S_C + G_H) / Q_CMB/(np.pi*1e7*1e9)

plt.figure(1)
label = ['Latent heating','Secular cooling','Gravitational heating','Total energy']
plt.plot(time, L_H,label=label[0])
plt.plot(time, S_C,label=label[1]) 
plt.plot(time, G_H, label=label[2])
plt.plot(time, L_H + S_C + G_H,label=label[3])
plt.legend()
plt.xlabel('Time (Gyr)')
plt.ylabel('Energy (J)')
plt.show()

plt.figure(2)
plt.plot(time, r_IC_vec*1e-3,label='$Q_{\mathrm{CMB}}$=7.4e12 W')
plt.xlabel('Time (Gyr)')
plt.ylabel('Inner core size (km)')
plt.legend()
plt.show()

# For different Qcmb according to Labrosse 2001    

L_H= np.zeros((x,4),dtype=np.float32)
S_C = np.zeros((x,4),dtype=np.float32)
G_H = np.zeros((x,4),dtype=np.float32)
time = np.zeros((x,4),dtype=np.float32)
   
for i in range(len(r_IC_vec)):
    L_H[i] = LH(r_IC_vec[i])
    S_C[i] = SC(r_IC_vec[i])
    G_H[i] = GH(r_IC_vec[i])
    
E_tot = L_H + S_C + G_H

time = (L_H + S_C + G_H) / Q_cmb/1e9

plt.figure(2)
plt.plot(time, r_IC_vec*1e-3)
plt.xlabel('Time (Gyr)')
plt.ylabel('Inner core size (km)')
plt.show()

