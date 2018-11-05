#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:34:34 2018

@author: irenebonati
"""

# Adiabatic temperature profile

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# 1 R_Earth
# Thermodynamic parameters
Tsurf  = 300     # K
Rp     = 6378    # km
Tm_u   = 2e3     # K
Tm_l   = 2850    # K
Tcmb   = 3900    # K
pcmb   = 127     # GPa
Ra     = 7.56e5  # viscosity: 1023
Dm     = 2780    # Mantle thickness, km
g      = 9.81    # m/s2
rho    = 4.5e3   # kg/m3
dbl_l  = 800     # km
dbl_u  = 250     # km

Rp     = 10972   # km
Tm_l   = 4500    # K
Tcmb   = 5600    # K
Dm     = 5179    # Mantle thickness, km
dbl_l  = 800     # km
dbl_u  = 250     # km

d = np.linspace(0,Dm,(Dm+1))
T = np.zeros((len(d)),dtype=np.float32)
T[0:dbl_u+1] = np.linspace(Tsurf,Tm_u,dbl_u+1)
T[dbl_u+1:Dm-dbl_l+1] = np.linspace(Tm_u,Tm_l,Dm-dbl_l-dbl_u)
T[Dm-dbl_l+1:Dm+1] = np.linspace(Tm_l,Tcmb,dbl_l)  

plt.figure(1)
plt.plot(T,d)
plt.xlim(0,7e3)
plt.ylim(0,6e3)
plt.gca().invert_yaxis()

#------------------------ Driscoll & Bercovici 2014 ------------------------- #

T_g = 300 
year      = 365*24*3600  # 1 year

# Mantle 

c_m         = 1265         # Specific heat of mantle (Jkg-1K-1)
M_m         = 4.06e24      # Mantle mass (kg)
k_UM        = 4.2          # Upper mantle thermal conductivity (Wm-1K-1)
alpha       = 3e-5         # Thermal expansivity mantle (K-1)
g           = 9.8          # Gravitational acceleration (ms-2)
kappa       = 1e-6         # Mantle thermal diffusivity (m2s-1)
nu_0        = 7e7          # Reference viscosity (m2s-1)
A_v         = 3e5          # Viscosity activation energy (Jmol-1)  
R_g         = 8.314472     # Gas constant (JK-1mol-1)
Ra_c        = 660          # Critical Rayleigh number
beta        = 1./3         # Section 4 of the paper
D           = 2891e3       # Mantle depth (m)
eta_UM      = 0.7          # Adiabatic temperature decrease from average mantle 
                           # temperature to the bottom of the UM TBL        
eta_LM      = 1.3
eta_c       = 0.8
Q_rad0      = 13e12        # Present-day radiogenic heating (W=Js-1)   
tau_rad_m   = 2.94e9*year  # Mantle radioactive decay time scale (s)
k_LM        = 10           # Lower manle thermal conductivity (Wm-1K-1)
SL_ML       = 1./25        # with T_ref = 2000 K and A_v = 3e5 Jmol-1
L_melt      = 3.2e5        # Latent heat of mantle melting (Jkg-1)
rho_solid   = 3300         # Mantle upwelling solid density (kgm-3)
R           = 6371e3 
T_sol0      = 1244         # Mantle solidus at surface (K)
gamma_ad    = 1e-3         # Adiabatic gradient of the melt (Kkm-1)
gamma_z     = 3.9e-3       # Solidus gradient (Km-1)
f_vol0      = 1e-4         # Reference melt fraction
rho_melt    = 2700         # Mantle melt density (kgm-3)
St          = 3.5          # Stefan number --> c_m*dT_melt/L_melt
 
# Core

M_c         = 1.95e24      # Core mass (kg)
c_c         = 840          # Specific heat of core (Jkg-1K-1)
eta_c       = 0.79
tau_rad_c   = 1.2e9*year 
L_Fe        = 750e3        # Latent heat of inner core crystallization (kJkg-1) 
E_G         = 3e5          # Gravitational energy ICB (Jkg-1)
R_c         = 3480e3       # Core radius
D_N         = 6340e3       # Core adiabatic length scale
D_Fe        = 7000e3       # Iron solidus length scale
T_Fe0       = 5600             
gamma_c     = 1.3          # Core Grueneisen parameter
rho_ic      = 13000        # Inner core density (kgm-3)
gamma_d     = 0.2          # Saturation constant fast rotating dynamos
mu_0        = 4*np.pi*1e-7 # Magnetic permeability (Hm-1)
alpha_c     = 1e-5         # Thermal expansivity core (K-1)
g_c         = 10.5         # CMB gravity (ms-2)
rho_c       = 11900        # Core density (kgm-3)
sigma_c     = 1e6          # Elecrtical consuctivity core (ohm-1m-1)
L_c         = 2.5*1e-8     # Lorentz number (WohmK-1)
beta        = 1./3.

def calc_Q_surf(Q_rad,Q_cmb,Q_man):
    return Q_rad+Q_cmb+Q_man
 
def calc_Tdot_m (Q_rad,Q_cmb,Q_conv,Q_melt):  # Q_surf 
    return (Q_cmb+Q_rad-Q_conv-Q_melt)/M_m/c_m

# Should we consider tidal heating for close-in planets?

def calc_Q_man(Tdot_m):    
    return -c_m*M_m*Tdot_m

def calc_Q_conv (T_m, T_g, nu_UM):
    #Q_conv = A*k_UM*dT_UM/delta_UM    dT_UM = eta_UM*T_m   dT_m = T_m-T_g
    return A*k_UM*((alpha*g/Ra_c/kappa)**beta)*(eta_UM*(T_m-T_g))**(beta+1)*(nu_UM)**(-beta)
    
def Rayleigh_UM(dT_UM, delta_UM, nu_UM):
    return alpha*g*dT_UM*delta_UM**3/kappa/nu_UM

def calc_nu_mantle(T_m):
    return nu_0*np.exp(A_v/R_g/T_m)

def calc_nu_UM(T_UM):
    return calc_nu_mantle(T_UM)/10.

def calc_dT_UM(T_UM,T_g):
    return T_UM-T_g

def calc_delta_UM(nu_UM,dT_UM):
    return D*(Ra_c*nu_UM*kappa/alpha/g/dT_UM/D**3)**(beta)

def calc_Q_rad_m(t):
    return Q_rad0*np.exp(-t/tau_rad_m)    # Q_rad(4.5Ga)= 54 TW (I get 2.18 TW)

def Urey(Q_rad,Q_surf):                 # 1/3 for Q_surf = 39 TW and Q_rad = 13 TW (present-day)
    return Q_rad/Q_surf

def heat_loss_cmb(T_cmb,T_LM,nu_LM,A_c):
    # dT_LM = T_cmb-T_LM       T_LM = eta_LM*T_m     T_cmb = eta_c*T_c
    #return A_c*k_LM*dT_LM/delta_LM
    return (A_c*k_LM*(T_cmb-T_LM))/((kappa*nu_LM*Ra_c/alpha*g*(T_cmb-T_LM))**(1./3.))

def calc_delta_LM(nu_LM,dT_LM):
    return (kappa*nu_LM*Ra_c/alpha/g/dT_LM)**(1./3)

Q_rad=calc_Q_rad_m(4.5e9*year)/1e12


# Stagnant vs. mobile-lid

def Q_conv_SL(Q_conv):
    return SL_ML*Q_conv


# Melting

def calc_Q_melt(eps_erupt, Mdot_melt, dT_melt):    
    return eps_erupt*Mdot_melt*(L_melt+c_m*dT_melt)

A_p = 4*np.pi*R**2      # Ask Marine

def calc_Mdot_melt(f_melt, delta_UM):   
    Vdot_up = 1.16*kappa*A_p/delta_UM
    return Vdot_up*rho_solid*f_melt

#Mdot_melt =  calc_Mdot_melt(0.1, 80000)

z = np.linspace(0,300e3,500)

def temperature_profile_melting_region(T_g,dT_UM,delta_UM,z):
    return T_g+dT_UM*erf(2*z/delta_UM)

def calc_T_melt(T_UM):      
    return (T_UM+T_sol0)/2

def temperature_perturbation_surface(T_melt,T_g,T_UM):
    z_2 = (T_UM-T_sol0)/gamma_z
    z_melt = z_2/2
    return T_melt-T_g-z_melt*gamma_ad
    
def calc_T_sol(z):
    return T_sol0+gamma_z*z

def calc_f_vol(T_melt):         
    return f_vol0*(T_melt-1373)    

def calc_f_melt(f_vol):         
    return (f_vol*rho_solid)/(rho_melt+f_vol*(rho_solid-rho_melt))

def calc_Q_ad(dT_D,k):
    return A*k*dT_D/D

def Nusselt_number(Q_conv,Q_ad):
    return Q_conv/Q_ad

def calc_Nu_melt(Nu,eps_erupt,f_melt):
    a_0 = 0.7 # Constant
    return Nu*a_0*eps_erupt*f_melt*(St**(-1)+1)

def calc_Nu_melt22(Nu,Q_melt,Q_conv):
    return Nu*Q_melt/Q_conv

def internal_heat(Mdot_melt,DeltaT_melt):
    return Mdot_melt*c_m* DeltaT_melt

def latent_heat(Mdot_melt):
    return Mdot_melt*L_melt

# Figure 2 a
T_UM = 1630

nu_UM    = calc_nu_UM(T_UM)
dT_UM    = calc_dT_UM(T_UM,T_g)
delta_UM = calc_delta_UM(nu_UM,dT_UM)

T = temperature_profile_melting_region(T_g,calc_dT_UM(T_UM, T_g),calc_delta_UM(calc_nu_UM(T_UM),calc_dT_UM(T_UM, T_g)),z)
plt.figure(2)
plt.plot(z/1000, T,label='$T_{\mathrm{UM}}$=1630 K')
plt.xlim([0,300])

T_UM = 2130

nu_UM    = calc_nu_UM(T_UM)
dT_UM    = calc_dT_UM(T_UM,T_g)
delta_UM = calc_delta_UM(nu_UM,dT_UM)

T = temperature_profile_melting_region(T_g,calc_dT_UM(T_UM, T_g),calc_delta_UM(calc_nu_UM(T_UM),calc_dT_UM(T_UM, T_g)),z)
plt.plot(z/1000, T,label='$T_{\mathrm{UM}}$=2130 K')

T_sol = calc_T_sol(z)
plt.plot(z/1000,T_sol,label='$T_{\mathrm{sol}}$')
plt.legend()
plt.xlabel('z (km)')
plt.ylabel('T (K)')

# Figure 2 b
T_UM = np.linspace(1600,2200,500)

T_melt      = calc_T_melt(T_UM)
Delta_Tmelt = temperature_perturbation_surface(T_melt,T_g,T_UM)
Delta_TUM   = calc_dT_UM(T_UM,T_g)
plt.figure(3)
plt.plot(T_UM,T_melt,T_UM,Delta_Tmelt,T_UM,Delta_TUM)
plt.ylim([500,2000])
plt.ylabel('T (K)')
plt.xlabel('$T_{\mathrm{UM}}$ (K)')
plt.legend()

# Figure 2 c
eps_erupt   = 1.
dT_UM       = calc_dT_UM(T_UM,T_g)
delta_UM    = calc_delta_UM(nu_UM,dT_UM)
T_melt      = calc_T_melt(T_UM)
f_vol       = calc_f_vol(T_melt)
f_melt      = calc_f_melt(f_vol)
Mdot_melt   = calc_Mdot_melt(f_melt, delta_UM)
Delta_Tmelt = temperature_perturbation_surface(T_melt,T_g,T_UM)
Q_melt      = calc_Q_melt(eps_erupt, Mdot_melt, Delta_Tmelt)

int_heat = internal_heat(Mdot_melt,Delta_Tmelt)

lat_heat = latent_heat(Mdot_melt)

T_cmb   = 4104
T_m     = 1630
k = 4
A = 4*np.pi*R**2
eps_erupt = 1 
dT_D    =  T_cmb -T_g
Q_ad    = calc_Q_ad(dT_D,k)
nu_UM   = calc_nu_UM(T_UM)
Q_conv  = calc_Q_conv (T_m, T_g, nu_UM)
Nu      = Nusselt_number(Q_conv,Q_ad)
Nu_melt = calc_Nu_melt(Nu,eps_erupt,f_melt)/Nu
Nu_melt22 = calc_Nu_melt22(Nu,Q_melt,Q_conv)


plt.figure(4)
plt.plot(T_UM,Q_melt/1e12,label='$Q_{\mathrm{melt}}$')
plt.plot(T_UM,int_heat/1e12,label='$Q_{\mathrm{int}}$')
plt.plot(T_UM,lat_heat/1e12,label='$Q_{\mathrm{lat}}$')
#plt.plot(T_UM,Nu_melt,label='$Nu_{\mathrm{melt}}$')
plt.plot(T_UM,Nu_melt22,label='$Nu_{\mathrm{melt22}}$')

plt.semilogy()
plt.ylim([0.1,1e2])
plt.ylabel('$Q_{\mathrm{melt}}$ (TW)')
plt.xlabel('$T_{\mathrm{UM}}$ (K)')
plt.legend()
plt.show()


# Core

# Core cooling

def calc_Q_cmb(Q_core,Q_L, Q_G, Q_rad_c):
    return Q_core+Q_L+Q_G+Q_rad_c

def calc_Q_core(Tdot_c):
    return -M_c*c_c*Tdot_c

def calc_T_cmb(T_c):
    return eta_c*T_c

def calc_Q_rad_c(t):
    return Q_rad0*np.exp(-t/tau_rad_c)    # Q_rad(4.5Ga)= 0.2 TW-1.95 TW

Q_rad_core=calc_Q_rad_c(4.5e9*year)/1e12

def calc_Q_icb(Mdot_ic):
    return Mdot_ic*(L_Fe+E_G)

def calc_T_c(T_cmb,r):
    T_cmb*np.exp((R_c**2-r**2)/D_N**2)
    
def calc_Fe_solidus(r):
    return T_Fe0*np.exp(-2*(1-1/(3*gamma_c))*r**2/D_Fe**2)

# No volatile depression of solidus considered
 
a = (D_N/D_Fe)**2
b = (D_N/R_c)**2
c = 1-1/(3*gamma_c)
def calc_R_ic(T_cmb):  
    return R_c*np.sqrt((b*np.ln(T_Fe0/T_cmb)-1)/(2*c*a-1))
             

def calc_Rdot_ic(Tdot_c,T_c,R_ic):
    return -(D_N**2*Tdot_c)/(2*R_ic*(2*c*a-1)*T_c)

def calc_Mdotic(A_ic,Tdot_cmb,dRic_dTcmb):
    return A_ic*rho_ic*Tdot_cmb*dRic_dTcmb

def calc_dRic_dTcmb(T_cmb):
    return -(R_c/(2*T_cmb)*b)/(b*np.ln(T_Fe0/T_cmb)-1)

def calc_Tdot_c(A_ic,Q_cmb,Q_rad_c,dRic_dTcmb):
    return -(Q_cmb-Q_rad_c)/(M_c*c_c-A_ic*rho_ic*eta_c*dRic_dTcmb*(L_Fe+E_G))

def calc_magnetic_dipole_moment(rho, F_c,D_c):
    return 4*np.pi*R_c**3*gamma_d*np.sqrt(rho/(2*mu_0))*(F_c*D_c)**(1./3)

def calc_Dc(R_ic):     # Dynamo region shell thickness
    return R_c-R_ic

def calc_Fc(F_th,F_X): # Buyoancy flux
    return F_th + F_X

def calc_F_th(qc_conv):
    alpha_c*g_c/(rho_c*c_c)*qc_conv
    
def calc_qc_conv(q_cmb,qc_ad):
    return q_cmb-qc_ad

def calc_qc_ad(T_cmb,k_c):
    return k_c*T_cmb*R_c/(D_N)**2

def calc_kc(T_cmb):
    T_cmb = 4000
    return sigma_c*L_c*T_cmb
kc = calc_kc(4000)

def calc_F_X(Deltarho_X,Rdot_ic,R_ic,g_ic):
    return g_ic*Deltarho_X/rho_c*(R_ic/R_c)**2*Rdot_ic

def calc_g_ic(R_ic):
    return g_c*R_ic/R_c

def calc_Deltarho_X(rho_X):
    return rho_c-rho_X


# Simple thermal history results

# Mantle-only model (Q_cmb=0)

gamma_m = 0.87 # for T_m = 1630 K --> present-day mantle temperature
eta_melt = 0.08
gamma_L = 721 # JK-1
A = 4*np.pi*R**2
Q_cmb = 0

def calc_Q_conv_nocore(a1,beta,nu_UM,T_m):
    return a1*(nu_UM)**(-beta)*(T_m)**(beta+1.)

def calc_a1(beta):
    return A*k_UM*(alpha*g/kappa/Ra_c)**beta*(eta_UM*gamma_m)**(beta+1)

def calc_Q_melt_nocore(eps_e,a2,nu_UM,T_m, beta):
    return eps_e*a2*(nu_UM)**(-beta)*(T_m)**(beta+2.)

def calc_a2(beta,f_melt0):
    return (1.16)*eta_melt*gamma_L*(rho_solid*kappa*A*f_melt0)/(D)*((alpha*g*D**3*eta_UM*gamma_m)/(Ra_c*kappa))**(beta)

def calc_DeltaT_m(T_m):
    return gamma_m*T_m

def calc_dTm(Q_conv,Q_melt,Ur):
    return (-Q_conv-Q_melt+Ur*(Q_conv+Q_melt))/M_m/c_m
    
    

# Thermal catastrophe: T_m > 3000 K

Ur = 0.67
beta = 0.33
eps_e = 0

T_m = 2400
nu_mantle = calc_nu_mantle(T_m)
nu_UM = 1e18#calc_nu_mantle(T_m)#/10.

a1 = calc_a1(beta)#8.4e10 (I get e14)

a2 = calc_a2(0.33,0.1)  


dt = 4.5e7*year
time_end = 4.5e9#*year
time_vector=np.linspace(time_end,0,101)

T_m = np.zeros((len(time_vector),1),dtype=np.float32)
T_m_melt = np.zeros((len(time_vector),1),dtype=np.float32)
dT_m = np.zeros((len(time_vector),1),dtype=np.float32)
Q_conv = np.zeros((len(time_vector),1),dtype=np.float32)
Q_conv_melt = np.zeros((len(time_vector),1),dtype=np.float32)
Q_melt = np.zeros((len(time_vector),1),dtype=np.float32)
dT_m_melt = np.zeros((len(time_vector),1),dtype=np.float32)

T_m[0] = 2400
T_m_melt[0] = 2400


for i in range(len(time_vector)):
    Q_conv[i] = calc_Q_conv_nocore(a1,beta,nu_UM,T_m[i])
    Q_conv_melt[i] = calc_Q_conv_nocore(a1,beta,nu_UM,T_m_melt[i])
    Q_melt[i] = calc_Q_melt_nocore(1,a2,nu_UM,T_m_melt[i], beta)
    dT_m[i] =calc_dTm(Q_conv[i],0,Ur)*-dt
    dT_m_melt[i] = calc_dTm(Q_conv_melt[i],Q_melt[i],Ur)*-dt
    if i<len(time_vector)-1:
        T_m[i+1] = T_m[i] + dT_m [i]
        T_m_melt[i+1] = T_m_melt[i]+ dT_m_melt[i]


plt.plot(time_vector/1e9,T_m,time_vector/1e9,T_m_melt)
plt.xlim(0,5)
plt.ylim(0,6000)
plt.show()

plt.plot(time_vector/1e9,Q_conv/1e12,time_vector/1e9,(Ur*Q_conv)/1e12,time_vector/1e9,Q_conv_melt/1e12,time_vector/1e9,Q_melt/1e12)
#time_vector/1e9,Q_conv/1e12,time_vector/1e9,(Ur*Q_conv)/1e12,
plt.semilogy()
plt.ylim(1,1000)
plt.xlim(0,5)
plt.show()



























