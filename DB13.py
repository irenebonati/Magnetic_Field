#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:01:43 2018

@author: irenebonati
"""

'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
                           Driscoll & Bercovici 2013
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ----------------------------- Model constants ----------------------------- #

year = 365*24*3600  # 1 year (s)

tau_m     = 3200*year                # Vertical melt percolation time (s)
P_w       = 270e5                    # bar
P_sat0    = 610                      # Pa   
Lw        = 2469                     # Jg-1
T_sat0    = 273                      # K
R_g       = 8.314                    # J mol-1 K-1                 
phistar_W = 3.16*1e18                # CO2 m-3yr-1
E_a       = 4.186e4                  # Jmol-1
R         = 6371e3
a         = 1.
ustar_p   = (50e-3)/(year)           # ms-1
rho_w     = 1e3                      # kgm-3
L_p       = 4e6                      # m
b         = 2e21                     # m-1s-1
g         = 9.8                      # ms-2
Rg        = 8.314
# Upper limit to hydrodynamic escape rate
phi_LH    = 2e31                     # s-1
a_L       = 4
tau_L     = 10                       # s
A_cl      = 0.9
tau_cl_E  = 1.4
tau_cl_V  = 21
eps_sa    = 4e-6
mu_0      = 4*np.pi*1e-7
# -----------------------------------------------------------------------------
#                                    CO2 cycle
# -----------------------------------------------------------------------------

# DEGASSING

# Degassing flux of water (w), Eq(9)
def calc_F_D_wm(wm,eps_D):
    return wm*eps_D/tau_m

# Degassing flux of carbon dioxide (c), Eq(9)
def calc_F_D_wc(wc,eps_D):
    return wc*eps_D/tau_m

# Precipitation flux of water from atmosphere to ocean, Eq.(10)
def calc_F_ocw(wa,wsat,tau_pr):
    return (wa-wsat)/tau_pr

def calc_wsat(T_g,P_sat):
    return P_sat/P_w

# Saturation pressure of water, Eq(11)
def calc_P_sat(T_g,mw):
    return P_sat0*np.exp(-mw*Lw/R_g*(1./T_g-1./T_sat0))

# WEATHERING

# CO2 weathering flux (yr-1), Eq(12)
def calc_FW_c(F_W,f_A):
    return F_W*f_A

# Weathering function, Eq.(13)
def calc_FW(P_c,Pstar_c,P_sat,Pstar_sat,Tstar,T):
    return phistar_W*(P_c/Pstar_c)**(0.55)*(P_sat/Pstar_sat)**(0.3)*np.exp(E_a/R_g*(1/Tstar-1/T))
 
A_E = 4*np.pi*R**2    

# Dimensionless weatherable area fraction, normalized to present-day (Eq.(14))
def calc_f_A(A_oc,u_p):
    return (1-A_oc/A_E)*(u_p/ustar_p)**a

def calc_A_oc(r_os):
    return np.pi*r_os**2

def calc_r_os(V_oc,theta_oc):
    return (3*V_oc/np.pi*np.tan(theta_oc))**(1./3.)

def calc_V_oc(M_oc):
    return M_oc/rho_w

def calc_M_oc(w_oc,Mstar_oc):
    return w_oc*Mstar_oc
    

# SUBDUCTION
    
# Subduction flux on ocean plate, Eq.(15)
def calc_Fsc(c_p,tau_p):
    return c_p/tau_p

def calc_tau_p(u_p):
    return L_p/u_p


# DISSOLUTION
    
# Flux of CO2 in the ocean, Eq.(16)
    

# -----------------------------------------------------------------------------
#                                    Escape flux
# -----------------------------------------------------------------------------

# DIFFUSION LIMITED ESCAPE

# Diffusion limited escape flux, Eq.(18)
def calc_FLD(eps_LD,Te,ma,mH2,alpha_t):
    return eps_LD*A_E*b*g/Rg/Te*(ma-mH2)*alpha_t

def calc_eps_LD(Nw):
    return Nw**(-1)


# MAGNETIC LIMITED ESCAPE
    
## Dynamic pressure solar wind
#def calc_Psw(n_sw,m_p,v_sw):
#    return m_p*n_sw*v_sw**2

## Height of magnetopause
#def calc_R_mp(mu_o,M,P_sw):
#    return ((mu_0*M**2)/(8*np.pi**2*P_sw))(1./6)

# Escape flow, Eq.(20)
def calc_F_LM(eps_LM,H_L,A_L,n_L,n_sw,tau_L):
    return eps_LM*H_L*A_L*(n_L-n_sw)/tau_L

def calc_eps_LM(Nw):
    return Nw**(-1)

# Loss area
def calc_A_L(R_mp):
    return 4*np.pi*R_mp**2

# Weak magnetic field (Venus) --> Loss altitude
def calc_z_L(H_g,P_g,P_sw):
    return H_g*np.ln(P_g/P_sw)

# Loss time scale of hydrogen atoms
def calc_tau_L(m_p,e,B,rho):
    return 4*m_p*a_L**2/np.pi/e/B/rho

# Hydrogen number density, Eq.(22)
def calc_n(n_exo,R_exo,H_exo,r):
    return n_exo*np.exp(-R_exo/H_exo*(1-R_exo/r))
    
def calc_n_exo(H_exo,sigma_coll):
    return (H_exo*sigma_coll)**(-1.)

def calc_H_exo(k_b,T_exo,m1):
    return k_b*T_exo/m1/g

def calc_z_exo(H_g,n_g,n_exo):
    return H_g*np.ln(n_g/n_exo)

# Exobase temperature, Eq.(23)
def calc_T_exo(T_0s,eps_X,I_X,k_b,sigma_coll,K_0,m_abs,sigma_abs):
    return T_0s+eps_X*I_X*k_b*sigma_coll/K_0/m_abs/g/sigma_abs


# -----------------------------------------------------------------------------
#                                    Ground temperature
# -----------------------------------------------------------------------------

# Ground temperature solution, Eq.(24)
def calc_Tg4(T_e,tau_g):
    return T_e**4*(1.+3./4*tau_g)

# Effective temperature, Eq.(25)
def calc_Te(A,S,sigma):
    return (1-A)*S/4/sigma

def S(L,a):
    return L/np.pi/4/a**2

# Optical depth at ground, Eq.(26)
def calc_tau_g(P_w,P_kw,P_c,P_kc):
    return P_w/P_kw+P_c/P_kc
    
def calc_P_kw(ma,m_w,mu_w,kappa_w):
    return ma/m_w*g/mu_w/kappa_w

def calc_A(A_g,phi):
    return A_g*(1-phi)+A_cl*phi

def calc_A_g(x_oc,A_oc,x_r,A_r,x_i,A_i):
    return x_oc*A_oc+x_r*A_r+x_i*A_i

# Reflectivity of clouds
def calc_phi(gamma,beta,tau_cl):
    return gamma*(1-np.exp(-2*beta*tau_cl))/(1-gamma**2*np.exp(-2*beta*tau_cl))

# Cloud optical depth
def calc_tau_cl(P_w,P_sa,P_kappawcl,P_kappasacl):
    return P_w/P_sa+P_kappawcl/P_kappasacl

def calc_P_sa(P_c):
    return eps_sa*P_c

def calc_taudot_g(Pdot_c,P_kappac,Pdot_w,P_kappaw):
    return Pdot_c/P_kappac+Pdot_w/P_kappaw

def Tdot_e(T_e,A,Adot):
    return -T_e/(4*(1-A))*Adot

def Adot(phi,Adot_g,DeltaA,A_g,phidot):
    return (1-phi)*Adot_g+(DeltaA-A_g)*phidot

def calc_phidot(gamma,phi,beta,tau_cl,taudot_cl):
    return 2*beta*gamma*(1-gamma*phi)/(1-gamma**2*np.exp(-2*beta*tau_cl))*taudot_cl
                         
def calc_taudot_cl(Pdot_sa,Pkappaclsa,Pdot_w,Pkappaclw):
    return Pdot_sa/Pkappaclsa+Pdot_w/Pkappaclw

def calc_Adot_g(xdot_oc,A_oc,xdot_r,A_r,xdot_i,A_i):
    return xdot_oc*A_oc+xdot_r*A_r+xdot_i*A_i

def calc_xdot_oc(r_os,R,wdot_oc):
    return 1./6*(r_os/R)**2*wdot_oc

def calc_xdot_i(eps_i,xstar_oc,x_oc,xdot_oc):
    return eps_i/xstar_oc*(1-2*x_oc)*xdot_oc

def calc_xdot_r(xdot_oc,xdot_i):
    return -xdot_oc-xdot_i


    

# Height of magnetopause
def calc_R_mp(mu_o,M,P_sw):
    return ((mu_0*M**2)/(8*np.pi**2*P_sw))(1./6)

# Dynamic pressure solar wind
def calc_Psw(n_sw,m_p,v_sw):
    return m_p*n_sw*v_sw**2











