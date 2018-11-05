#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:15:32 2018

@author: irenebonati
"""

'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
                           Driscoll & Bercovici 2014
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

'''

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Model constants ----------------------------- #


T_g  = 300          # Surface temperature (K)
year = 365*24*3600  # 1 year (s)

# Mantle parameters
cm           = 1265           # Specific heat of mantle (Jkg-1K-1)
Mm           = 4.06e24        # Mantle mass (kg)
k_UM         = 4.2            # Upper mantle thermal conductivity (Wm-1K-1)
k_LM         = 10             # Lower mantle thermal conductivity (Wm-1K-1)
R            = 6371e3         # Planetary radius (m)
R_UM         = 410e4          # Radius of upper mantle (m)
R_c          = 3471e3         # Radius of the core(m)
Ra_c         = 660            # Critical Rayleigh number for thermal convection
alpha        = 3e-5           # Thermal expansivity of the mantle (K-1)
g            = 10             # Gravity (ms-2)
kappa        = 1e-6           # Mantle thermal diffusivity (m2s-1)
nu_0         = 7e7            # Reference viscosity (Pa s)
A_nu         = 3e5            # Viscosity activation energy 
R_g          = 8.314          # Ideal gas constant (Jmol-1K-1)
nu_LMvsnu_UM = 10             # Lower mantle vs. upper mantle viscosity
D            = 2891e3         # Mantle depth (m)
eta_UM       = 0.7            # Upper mantle adiabatic temperature drop
eta_LM       = 1.3            # Lower mantle adiabatic temperature drop
eta_c        = 0.79           # Upper core adiabatic temperature drop
Q_rad0       = 13e12          # Present-day mantle radiogenic heat flow (W)
tau_rad      = 2.94*1e9*year  # Mantle radioactive decay time scale (s)

SL_ML        = 1./25          # with T_ref = 2000 K and A_v = 3e5 Jmol-1

L_melt       = 320e3          # Latent heat of mantle melting (Jkg-1)
rho_solid    = 3300           # Mantle upwelling solid density (kgm-3)
rho_melt     = 2700           # Mantle melt density (kgm-3)
T_sol0       = 1244           # Surface melting temperature (K)
gamma_z      = 3.9e-3         # Solidus gradient (Km-1)
gamma_ad     = 1e-3           # Magma adiabatic gradient (Km-1)
f_vol0       = 1e-4           # Reference volumetric melt fraction (K-1)

# Core parameters
Mc          = 1.95e24         # Core mass (kg)
cc          = 840             # Specific heat of core (Jkg-1K-1)
tau_rad_c   = 1.2*1e9*year    
L_Fe        = 750e3        # Latent heat of inner core crystallization (kJkg-1) 
E_G         = 3e5          # Gravitational energy ICB (Jkg-1)
R_c         = 3480e3       # Core radius
D_N         = 6340e3       # Core adiabatic length scale
D_Fe        = 7000e3       # Iron solidus length scale
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
T_Fe0       = 5600             

#----------------------------- Mantle cooling ------------------------------- #

# Conservation of energy in the mantle (Eq.(1))
def calc_Qsurf_mantle(Q_conv,Q_melt):
    return Q_conv+Q_melt

# Mantle secular cooling rate (Eq.(2))
def calc_Q_mantle(Tdot_m):
    return -cm*Mm*Tdot_m

# Mantle thermal evolution (Eq.(3))
def calc_Tdot_mantle(Q_cmb,Q_rad,Q_conv,Q_melt):
    return (Q_cmb+Q_rad-Q_conv-Q_melt)/Mm/cm

# Surface area of the upper mantle
def calc_A_UM(R_UM):
    return 4*np.pi*R_UM**2

A_um = calc_A_UM(R)

# Convective cooling of the mantle (Eq.(4) and (8))
def calc_Qconv_mantle(A_um,DeltaT_m,delta_UM,beta,nu_UM):
    #return A*k_UM*DeltaT_UM/delta_UM
    return A_um*k_UM*(alpha*g/Ra_c/kappa)**(beta)*(eta_UM*DeltaT_m)**(beta+1)*(nu_UM)**(-beta)

# Temperature jump across upper mantle thermal boundary layer (delta_UM) 
def calc_DeltaT_UM(T_UM):
    return T_UM-T_g
    
# Temperature jump across upper mantle thermal boundary layer (delta_UM) 
def calc_DeltaT_UM_approx(DeltaT_m):#(T_UM):
    #return T_UM-T_g
    return eta_UM*DeltaT_m

# Rayleigh number upper mantle (Eq.(5))
def calc_Ra_UM(DeltaT_UM,delta_UM,nu_UM):
    return (alpha*g*DeltaT_UM*delta_UM**3)/(kappa*nu_UM)

# Average mantle viscosity (Arrhenius law, Eq.(6))
def calc_nu_m(T_m):
    return nu_0*np.exp(A_nu/R_g/T_m)

# Upper mantle viscosity
def calc_nu_UM(nu_m):
    return nu_m/10.

# Critical thermal boundary layer thickness (Ra_UM=Ra_c, Eq.(7))
def calc_delta_UM(nu_UM,DeltaT_UM):
    return (Ra_c*nu_UM*kappa/alpha/g/DeltaT_UM)**(1./3.)

# Critical thermal boundary layer thickness (Ra_UM=Ra_c, Eq.(7))
def calc_delta_UM_beta(nu_UM,DeltaT_UM,beta):
    return D*(Ra_c*nu_UM*kappa/alpha/g/DeltaT_UM/(D**3))**beta

def calc_DeltaT_m(T_m):
    return T_m-T_g

# Total mantle radiogenic heat production (Eq.(9))
def calc_Q_rad(t):
    return Q_rad0*np.exp(-t/tau_rad)
    
# URey ratio (Eq.(10))
def calc_Ur(Q_rad,Q_surf):
    return Q_rad/Q_surf

Qrad45 =  calc_Q_rad(-4.5*1e9*year)   

# Surface area of the core
def calc_A_c(R_c):
    return 4*np.pi*R_c**2

A_c = calc_A_c(R_c)

# Core heat loss (Eq.(11))
def calc_Q_cmb(DeltaT_LM,delta_LM):
    return A_c*k_LM*DeltaT_LM/delta_LM

# Lower mantle thermal boundary layer temperature jump
def calc_DeltaT_LM(T_cmb,T_LM):
    return T_cmb-T_LM

# Lower mantle temperature
def calc_T_LM(T_m):
    return eta_LM*T_m
    
# CMB temperature
def calc_T_cmb(T_c):
    return eta_c*T_c

# Lower mantle thermal boundary layer thickness (Eq.(12))
def calc_delta_LM(nu_LM,DeltaT_LM):
    return (kappa*nu_LM*Ra_c/alpha/g/DeltaT_LM)**(1./3.)

def calc_nu_LM(nu_LMvsnu_UM,nu_UM):
    return nu_LMvsnu_UM*nu_UM


##------------------------- Stagnant vs. mobile lid -------------------------- #
#
#def Q_conv_SL(Q_conv):
#    return SL_ML*Q_conv


#------------------------- Heat loss due to melting-------------------------- #

# Mantle melt heat loss (Eq.(13))
def calc_Q_melt(eps_erupt,Mdot_melt,DeltaT_melt):
    return eps_erupt*Mdot_melt*(L_melt+cm*DeltaT_melt)

# Melt mass flux (Eq.(14))
def calc_Mdot_melt(Vdot_up,f_melt):
    return Vdot_up*rho_solid*f_melt

# Total surface area
def calc_Atot(R):
    return 4*np.pi*R**2

A_tot = calc_Atot(R)

# Volumetric upwelling rate (Eq.(15))
def calc_Vdot_up(A_tot,delta_UM):
    return 1.16*kappa*A_tot/delta_UM

Vdot_up   = calc_Vdot_up(A_tot,80e3)
Mdot_melt = calc_Mdot_melt(Vdot_up,0.1)

# Conductive temperature profile in boundary layer (upper mantle, Eq.(16))
def calc_T(DeltaT_UM,z,delta_UM):
    return T_g+DeltaT_UM*erf(2*z/delta_UM)

# Low pressure solidus (Eq.(19))
def calc_T_sol(z):
    return T_sol0+gamma_z*z

# Figure 2 a
T_UM = 1630
z    = np.linspace(0,300e3,500)

nu_UM     = calc_nu_m(T_UM)
DeltaT_UM = calc_DeltaT_UM(T_UM)
delta_UM  = calc_delta_UM_beta(nu_UM,DeltaT_UM,1./3.)
T         = calc_T(DeltaT_UM,z,delta_UM)
#
plt.figure(1)
plt.plot(z/1000, T,label='$T_{\mathrm{UM}}$=1630 K')
plt.xlim([0,300])
plt.ylim([0,2500])


T_UM = 2130

nu_UM      = calc_nu_m(T_UM)
DeltaT_UM  = calc_DeltaT_UM(T_UM)
delta_UM   = calc_delta_UM_beta(nu_UM,DeltaT_UM,1./3.)
T          = calc_T(DeltaT_UM,z,delta_UM)
#
plt.plot(z/1000, T,label='$T_{\mathrm{UM}}$=2130 K')

T_sol = calc_T_sol(z)
plt.plot(z/1000, T_sol,label='$T_{\mathrm{sol}}$')

plt.legend()
plt.xlabel('z (km)')
plt.ylabel('T (K)')
plt.savefig('2a.eps', format='eps', dpi=1000)


# Effective temperature of partial melt (Eq.(17))
def calc_T_melt(T_UM):      
    return (T_UM+T_sol0)/2

# Temperature  perturbation at the surface (Eq.(18))
def calc_DeltaT_melt(T_melt,T_UM):
    z_2 = (T_UM-T_sol0)/gamma_z
    z_melt = z_2/2
    return T_melt-T_g-z_melt*gamma_ad

# Volumetric melt fraction (Eq.(20))
def calc_f_vol(T_melt):
    return f_vol0*(T_melt-1373)

# Melt mass fraction (Eq.(21))
def calc_f_melt(f_vol):
    return f_vol*rho_solid/(rho_melt+f_vol*(rho_solid-rho_melt))

# Figure 2b
T_UM = np.linspace(1600,2200,500)

T_melt      = calc_T_melt(T_UM)
DeltaT_melt = calc_DeltaT_melt(T_melt,T_UM)
DeltaT_UM   = calc_DeltaT_UM(T_UM)

plt.figure(2)
plt.plot(T_UM,T_melt,label='$T_{\mathrm{melt}}$')
plt.plot(T_UM,DeltaT_melt,label='$\Delta T_{\mathrm{melt}}$')
plt.plot(T_UM,DeltaT_UM,label='$\Delta T_{\mathrm{UM}}$')
plt.ylim([500,2000])
plt.ylabel('T (K)')
plt.xlabel('$T_{\mathrm{UM}}$ (K)')
plt.legend()
plt.savefig('2b.eps', format='eps', dpi=1000)



# Nusselt number for convective heat transport
def calc_Nu(Q_conv,Q_ad):
    return Q_conv/Q_ad

def calc_Q_ad(DeltaT_D):
    return A_tot*k_UM*DeltaT_D/D

def calc_DeltaT_D(T_cmb):
    return T_cmb-T_g

# Figure 2c
T_UM = np.linspace(1600,2200,500)

f_vol     = calc_f_vol(T_melt)
f_melt    = calc_f_melt(f_vol)
nu_UM     = calc_nu_m(T_UM)/10.
delta_UM  = calc_delta_UM(nu_UM,DeltaT_UM)
Vdot_up   = calc_Vdot_up(A_tot,delta_UM)
Mdot_melt = calc_Mdot_melt(Vdot_up,f_melt)

eps_erupt     = 1.
Q_melt        = calc_Q_melt(eps_erupt,Mdot_melt,DeltaT_melt)
internal_heat = Mdot_melt*cm*DeltaT_melt
latent_heat   = Mdot_melt*L_melt


plt.figure(3)
plt.plot(T_UM,Q_melt/1e12,label='$Q_{\mathrm{tot}}$')
plt.plot(T_UM,internal_heat/1e12,label='$Q_{\mathrm{int}}$')
plt.plot(T_UM,latent_heat/1e12,label='$Q_{\mathrm{lat}}$')

plt.semilogy()
plt.ylim([0.1,1e2])
plt.ylabel('$Q_{\mathrm{melt}}$ (TW)')
plt.xlabel('$T_{\mathrm{UM}}$ (K)')
plt.legend()
plt.savefig('2c.eps', format='eps', dpi=1000)
plt.show()


# Core

# Core cooling

def calc_Q_cmb_2(Q_core,Q_L, Q_G, Q_rad_c):
    return Q_core+Q_L+Q_G+Q_rad_c

def calc_Q_core(Tdot_c):
    return -Mc*cc*Tdot_c

def calc_T_cmb_2(T_c):
    return eta_c*T_c

def calc_Q_rad_c(t):
    return Q_rad0*np.exp(-t/tau_rad_c)    # Q_rad(4.5Ga)= 0.2 TW-1.95 TW

Q_rad_core=calc_Q_rad_c(4.5e9*year)/1e12

def calc_Q_icb(Mdot_ic):
    return Mdot_ic*(L_Fe+E_G)

def calc_T_c(T_cmb,r):
    return T_cmb*np.exp((R_c**2-r**2)/(D_N**2))
    
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
    return -(Q_cmb-Q_rad_c)/(Mc*cc-A_ic*rho_ic*eta_c*dRic_dTcmb*(L_Fe+E_G))

def calc_magnetic_dipole_moment(rho, F_c,D_c):
    return 4*np.pi*R_c**3*gamma_d*np.sqrt(rho/(2*mu_0))*(F_c*D_c)**(1./3)

def calc_Dc(R_ic):     # Dynamo region shell thickness
    return R_c-R_ic

def calc_Fc(F_th,F_X): # Buyoancy flux
    return F_th + F_X

def calc_F_th(qc_conv):
    alpha_c*g_c/(rho_c*cc)*qc_conv
    
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


def calc_dTm(Q_conv,Q_melt,Ur):
    return (-Q_conv-Q_melt+Ur*(Q_conv+Q_melt))/Mm/cm
    
    

#Ur = 0.67
#beta = 0.33
#eps_e = 0
#
#a1 = calc_a1(beta)#8.4e10 (I get 8.5e14)
#
#f_melt0 = calc_f_melt(f_vol0)
#a2 = calc_a2(beta,0.1)  
#
#
#dt = 4.5e7*year
#time_end = 4.5e9
#time_vector=np.linspace(time_end,0,101)
#time_vector_rev=np.linspace(0,time_end,101)*year
#
#T_m = np.zeros((len(time_vector),1),dtype=np.float32)
#T_m_melt = np.zeros((len(time_vector),1),dtype=np.float32)
#dT_m = np.zeros((len(time_vector),1),dtype=np.float32)
#Q_conv = np.zeros((len(time_vector),1),dtype=np.float32)
#Q_conv_melt = np.zeros((len(time_vector),1),dtype=np.float32)
#Q_melt = np.zeros((len(time_vector),1),dtype=np.float32)
#dT_m_melt = np.zeros((len(time_vector),1),dtype=np.float32)
#nu_mantle = np.zeros((len(time_vector),1),dtype=np.float32)
#nu_UM = np.zeros((len(time_vector),1),dtype=np.float32)
#
#T_m[0] = 2300
#T_m_melt[0] = 2300
#
#for i in range(len(time_vector)):
#    nu_mantle[i] = calc_nu_m(T_m[i])
#    nu_UM[i] = 1e18#nu_mantle[i]/10.
#    Q_conv[i] = calc_Q_conv_nocore(a1,beta,nu_UM[i],T_m[i])
#    Q_conv_melt[i] = calc_Q_conv_nocore(a1,beta,nu_UM[i],T_m_melt[i])
#    Q_melt[i] = calc_Q_melt_nocore(1,a2,nu_UM[i],T_m_melt[i], beta)
#    dT_m[i] =calc_dTm(Q_conv[i],0,Ur)*-dt
#    dT_m_melt[i] = calc_dTm(Q_conv_melt[i],Q_melt[i],Ur)*-dt
#    if i<len(time_vector)-1:
##        dT_m_melt[i] = calc_dTm(Q_conv_melt[i],Q_melt[i],Ur)*-time_vector_rev[i+1]
##        dT_m[i] = calc_dTm(Q_conv[i],0,Ur)*-time_vector_rev[i+1]
#        T_m[i+1] = T_m[i] + dT_m [i]
#        T_m_melt[i+1] = T_m_melt[i]+ dT_m_melt[i]
#
#
#plt.plot(time_vector/1e9,T_m,time_vector/1e9,T_m_melt)
#plt.xlim(0,5)
#plt.ylim(0,6000)
#plt.show()
#
#plt.plot(time_vector/1e9,Q_conv/1e12,time_vector/1e9,(Ur*Q_conv)/1e12,time_vector/1e9,Q_conv_melt/1e12,time_vector/1e9,Q_melt/1e12)
##time_vector/1e9,Q_conv/1e12,time_vector/1e9,(Ur*Q_conv)/1e12,
#plt.semilogy()
#plt.ylim(1,1000)
#plt.xlim(0,5)
#plt.show()
