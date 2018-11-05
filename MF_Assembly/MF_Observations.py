#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:47:35 2018

@author: irenebonati
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from matplotlib import colors


font = {'family' : 'normal',
        'weight' : 'normal',
        'color'  : 'black',
        'size'   : 11}

font2 = {'family' : 'normal',
        'weight' : 'normal',
        'color'  : 'black',
        'size'   : 9}

font3 = {'family' : 'normal',
        'weight' : 'normal',
        'color'  : 'black',
        'size'   : 8}

# Parameters
N0 = 7                                    # m-3
alpha = 7e-6                              # Efficiency factor
mp = 1.6726e-27                           # Proton mass (kg)
dens = (mp*N0)*1e6
v = 4e5                                   # Solar wind speed (ms-1)
d = [1,5.203,9.539,19.18,30.06]           # Orbital distance (AU)
Rmp = [10*6.371e6,65*69.911e6,20*58.232e6,18*25.362e6,23*24.622e6]
omega = [23.9,9.9,10.7,17.2,16.1]
mass = [5.97e24,1898e24,568e24,86.8e24,102e24]
MJ = 1898e24
radius = [6.371e6,69.911e6,58.232e6,25.362e6,24.622e6]
surface_field = [38e-6,550e-6,28e-6,32e-6,27e-6]

for i in range(len(omega)):
    omega [i] = omega[i]* 3600

P = np.zeros((len(d),1),dtype=np.float64)
Pram = np.zeros((len(d)),dtype=np.float64)
Lrad = np.zeros((len(d),1),dtype=np.float64)
M   = np.zeros((len(d),1),dtype=np.float64)
fit = np.zeros((10,1),dtype=np.float64)
P_alt = np.zeros((len(d)),dtype=np.float64)
fit = np.zeros((10,1),dtype=np.float64)
M_mom = np.zeros((len(d)),dtype=np.float64)
black = np.zeros((len(d)),dtype=np.float64)

for i in range(len(d)):
        P[i]    = alpha*(dens/d[i]**2)*v**3*np.pi*Rmp[i]**2
        Pram[i] = P[i]/alpha
        Lrad[i] = P[i]*1e7
        P_alt[i] = 4e11*(omega[i]/(9.9*3600))**(0.79)*(mass[i]/MJ)**(1.33)*(d[i]/5.203)**(-1.6)
        #P_alt[i] = 4e9*(omega[i]/(9.9*3600))**(0.58)*(mass[i]/MJ)**(0.98)*(d[i]/5.203)**(-1.17)
        M_mom[i] = surface_field[i]*radius[i]**3
        black[i] = omega[i]*mass[i]**(5./3.)

       
x = np.linspace(1e11,3e15,10)

for i in range(len(x)):
    fit[i] = x[i]*7e-4

plt.figure(1)
fig, ax = plt.subplots()

text = ['Earth','Jupiter','Saturn','Uranus','Neptune']

for i in range(len(d)):
    plt.scatter(Pram[i],P_alt[i],color='royalblue',s=35)
    if i==1:
        plt.text(Pram[i]-0.5*Pram[i],P_alt[i]-0.4*P_alt[i],text[i],**font)
    
    elif i==len(d)-1:
        plt.text(Pram[i]+0.3*Pram[i],P_alt[i]-0.2*P_alt[i],text[i],**font)

    else:
        plt.text(Pram[i]+0.3*Pram[i],P_alt[i]-0.2*P_alt[i],text[i],**font)
plt.plot(x,fit,color='black',linestyle='--',linewidth=0.8)
  
plt.semilogy()
plt.semilogx()
plt.ylim([1e8,1e12])
plt.xlim([1e11,3e15])
plt.xlabel('Solar wind kinetic power (W)')
plt.ylabel('Radio Power (W)')
plt.title('Emitted radio power')
ax.xaxis.set_ticks_position('both')

ax2 = plt.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Radio Luminosity (erg/s)', color='black')  # we already handled the x-label with ax1
ax2.semilogy()
ax2.set_ylim([1e15, 1e19])
for i in range(len(d)):
    #plt.scatter(Pram[i],P[i]*1e7,color='royalblue',s=35)
    #plt.scatter(Pram[i],P_alt[i]*1e7,color='royalblue',s=35)
    plt.plot(Pram[i],fit[i]*1e7,color='black',linestyle='--')

plt.savefig('Bodes_Law.pdf', bbox_inches='tight',format='pdf') 
             
plt.show()

# Magnetization
for i in range(len(d)):
    M[i] = (Rmp[i])**6.*(2*np.pi*(dens/(d[i]**2))*v**2)
    M[i] = np.sqrt(M[i])

plt.figure(2)
ax1 = plt.subplots()
fig, ax = plt.subplots()

for i in range(len(d)):
    plt.scatter(black[i],M_mom[i],color='royalblue',s=35)
plt.text(black[0]+black[0]/3.5,M_mom[0]-M_mom[0]/4,text[0],**font)
plt.text(black[2]+black[2]/3.7,M_mom[2]-M_mom[2]/4,text[2],**font)
plt.text(black[1]-black[1]/1.4,M_mom[1]-M_mom[1]/4,text[1],**font)
plt.text(black[3]-black[3]/1.34,M_mom[3]-M_mom[3]/4,text[3],**font)
plt.text(black[4]+black[4]/4,M_mom[4]-M_mom[4]/4,text[4],**font)


plt.loglog()

plt.xlabel('$\omega M^{5/3}$')
plt.ylabel('Magnetic moment ($T m^{3}$)')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.title('Magnetic moment')
plt.savefig('Magn_Moment.pdf', bbox_inches='tight',format='pdf') 


T_eff = np.linspace(2500,6000,20)
for i in range(len(T_eff)):
    HZ_ven = 1.776+2.136e-4*(T_eff-5780)+2.533e-8*(T_eff-5780)**2-1.332e-11*(T_eff-5780)**3-3.097e-15*(T_eff-5780)**4
    HZ_mar = 0.32+5.547e-5*(T_eff-5780)+1.526e-9*(T_eff-5780)**2-2.874e-12*(T_eff-5780)**3-5.011e-16*(T_eff-5780)**4
    HZ_run = 1.107+1.332e-4*(T_eff-5780)+1.58e-8*(T_eff-5780)**2-8.308e-12*(T_eff-5780)**3-1.931e-15*(T_eff-5780)**4
    HZ_mg = 0.356+6.171e-5*(T_eff-5780)+1.698e-9*(T_eff-5780)**2-3.198e-12*(T_eff-5780)**3-5.575e-16*(T_eff-5780)**4


# Earth position
HZ_planets = [1.92,1,0.42]
T_eff_planets = [5780,5780,5780]
MF_planets = [0,1,0]


# MF calculation
beta = 0.1
mu_0 = 1.256*1e-6
rho_0 = 11e3
F_Earth = 2e-13
#omega_Earth = 7.29e-5
D_Earth = 2.26e6
Ro_Earth = 0.09
Ro_planet = 0.1
R_Earth = 6371000
M_Earth = 6e24
omega_Earth = 86400
MF_Earth = 7.8e22*5.

# Trappist 1 d
omega = 4.049959*86400
M = 0.32*M_Earth
R = 0.782*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_t1d = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth

# Trappist 1 e
omega = 6.099043*86400
M = 0.772*M_Earth
R = 0.910*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_t1e = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth


# Trappist 1 f
omega = 9.206690*86400
M = 0.7*M_Earth
R = 0.97*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_t1f = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth


# Trappist 1 g
omega = 12.35*86400
M = 1.34*M_Earth
R = 1.127*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_t1g = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth

# Proxima b
omega = 11.186*86400
M = 1.27*M_Earth
R = 1*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_proxb = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth

pos_Prox = 0.0017/(0.0485**2)

# Gliese 667Cc
omega = 28.155*86400
M = 3.709*M_Earth
R = 1.5*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_gliese = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth

pos_Gliese = 0.0137/(0.1251**2)

# Kepler 1229 b
omega = 86*86400
M = 2.7*M_Earth
R = 1.39*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_k1229b = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth

pos_k1229b=0.04784/(0.2896**2)


#K2-72e
omega = 24.1699*86400
M = 2.7*M_Earth
R = 0.82*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_K272e = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth

pos_K272 = 0.014/(0.2086**2)


#Luyten b
omega = 18.6498*86400
M = 2.89*M_Earth
R = 1.35*R_Earth

m_core = 1./0.21*(1.07-((R/R_Earth)/((M/M_Earth)**0.27)))
r_core = (np.sqrt(m_core))*R
D = 0.65*r_core
F = (0.1/0.09)**(2)*(D/D_Earth)**(2./3.)*(omega/omega_Earth)**(7./3.)*2e-13
MF_Luytb = (4*np.pi*(r_core)**(3.)*beta*(rho_0/mu_0)**(0.5)*(F*D)**(1./3.))/MF_Earth


HZ_planets = [1,1.15,0.72,0.39,0.27,0.65,0.94,0.57,pos_K272,1.23]
T_eff_planets = [5780,2550,2550,2550,2550,3042,3350,3784,3497,3382]
MF_planets = [1,MF_t1d,MF_t1e,MF_t1f,MF_t1g,MF_proxb,MF_gliese,MF_k1229b,MF_K272e,MF_Luytb]


plt.figure(3)
plt.plot(HZ_ven,T_eff,'white',HZ_mar,T_eff,'white')
plt.plot(HZ_run,T_eff,'black',linestyle=':')
plt.plot(HZ_mg,T_eff,'black',linestyle=':')
plt.fill_betweenx(T_eff, HZ_mar, HZ_ven,color='cornflowerblue',alpha=0.15)
MF = plt.scatter(HZ_planets,T_eff_planets,c=MF_planets,norm=colors.LogNorm(),s=55,edgecolors='black',linewidth=0.6)
cb = plt.colorbar(MF, ticks=[1e-2,1e-1,1,10,1e2])
cb.set_label('Magnetic moment (M$\oplus$)')
plt.xlim([0,2])
plt.ylim([2480,6020])
plt.xlabel('Stellar flux (F$\oplus$)')
plt.ylabel('Stellar effective temperature (K)')
plt.gca().invert_xaxis()
plt.text(1.22,2650,'T-1d',**font2)
plt.text(0.79,2650,'T-1e',**font2)
plt.text(0.46,2650,'T-1f',**font2)
plt.text(0.2,2530,'T-1g',**font2)
plt.text(1.09,5880,'Earth',**font2)
plt.text(0.8,3142,'Prox Cen b',**font2)
plt.text(0.93,3450,'GJ 667Cc',**font2)
plt.text(0.72,3884,'K 1229b',**font2)
plt.text(pos_K272+0.02,3597,'K2-72e',**font2)
plt.text(1.33,3482,'Luyt b',**font2)
plt.title('Magnetic moment of tidally-locked exoplanets',**font)
plt.savefig('HZ.pdf', bbox_inches='tight',format='pdf')


# Radio emission
MF_planets = [MF_t1d,MF_t1e,MF_t1f,MF_t1g,MF_proxb,MF_gliese,MF_k1229b,MF_K272e,MF_Luytb]
M_planets = [0.32,0.772,0.7,1.34,1.27,3.709,2.7,2.7,2.89] #Masses
R_planets = [0.782,0.910,0.97,1.127,1,1.5,1.39,0.82,1.35]
omega_planets = [4.049959*86400,6.099043*86400,9.206690*86400,12.35*86400,11.186*86400,28.155*86400,86*86400,24.1699*86400,18.6498*86400]
distance_planets = [12,12,12,12,1.3,6.8,236,69.8,3.74] #pc
a_planets = [0.0215,0.0282,0.0371,0.0451,0.0485,0.1251,0.2896,0.2086,0.091] #pc

P_rad = np.zeros((len(distance_planets),1),dtype=np.float64)
fc_max = np.zeros((len(distance_planets),1),dtype=np.float64)
sens = np.zeros((len(distance_planets),1),dtype=np.float64)

for i in range(len(distance_planets)):
    distance_planets[i]=distance_planets[i]*3.08e16
    M_planets[i]=M_planets[i]*M_Earth
    R_planets[i]=R_planets[i]*R_Earth
    MF_planets[i] = MF_planets[i]*7.8e22
    
for i in range(len(distance_planets)):
    P_rad[i]  = 4e11*(omega_planets[i]/(9.9*3600))**(0.79)*(M_planets[i]/MJ)**(1.33)*(a_planets[i]/5.203)**(-1.6)
    fc_max[i] = 24e6*(MF_planets[i]/(1.56e27))/((R_planets[i]**3)/(71492000**3))
    fc_max[i] = (1.60217662*1e-19*1.256*1e-6*MF_planets[i])/(4*(np.pi**2)*9.10*1e-31*R_planets[i]**3)
    sens[i] = P_rad[i]/(4*np.pi*distance_planets[i]**2*0.5*fc_max[i])*1e29

f_maxplot = np.zeros((len(distance_planets),1),dtype=np.float64)

for i in range(len(distance_planets)):
    f_maxplot[i] = fc_max[i]/1e6

MF_planets = [MF_t1d,MF_t1e,MF_t1f,MF_t1g,MF_proxb,MF_gliese,MF_k1229b,MF_K272e,MF_Luytb]

plt.figure(6)
plt.scatter(f_maxplot,sens,s=47,edgecolors='black',linewidth=0.6,color='royalblue')

plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux density (mJy)')
plt.axvline(10,1e-7,300,color='black',linestyle=':')
#plt.text(15,1e-2,'Ionospheric cutoff frequency',rotation=90,**font3)
plt.semilogy()
plt.semilogx()

plt.ylim([1e-5,4e2])
plt.xlim([1e-3,400])

plt.title('Flux density of tidally-locked exoplanets',**font)
plt.text(80,1e-4,'K2-72e',**font2)
plt.text(6,12,'Prox Cen b',**font2)
plt.text(f_maxplot[0]+2e-3,sens[0]-5,'T-1d',**font2)
plt.text(f_maxplot[1]-2.5e-1,sens[1]+0.2,'T-1e',**font2)

plt.text(f_maxplot[2]+2e-4,sens[2]-10,'T-1f',**font2)
plt.text(f_maxplot[3]+0.5e-1,sens[3],'T-1g',**font2)
plt.text(f_maxplot[5]-0.7e-1,sens[5]+7,'GJ 667Cc',**font2)
plt.text(f_maxplot[6]-0.3e-1,sens[6]+5e-3,'K 1229b',**font2)
plt.text(f_maxplot[8]+0.5,sens[8]-3,'Luyt b',**font2)


plt.savefig('Sensit_MF.pdf', bbox_inches='tight',format='pdf')
plt.show()
























 
