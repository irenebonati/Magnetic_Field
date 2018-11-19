#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Labrosse 2001/2003/2015
# Age and evolution of growth rate of the inner core


import numpy as np
import scipy.integrate as scODE
import matplotlib.pyplot as plt


# Physical parameters

R_OC = 3480.e3  # m
R_IC_P = 1221.e3  # m

QCMB = 10e12  # watts
H = 0  # radiactivity  !!! Not implemented yet

# from fitting PREM
K0 = 1403.e9  # Pa
RHO_0 = 12451  # kg/m-3
K_PRIM0 = 3.567  # no units

G = 6.67384e-11  # m3/kg/s2

DELTA_S = 127  # J/K/kg
M_P = 9e-9  # K/Pa
M_X = -21e3  # K
TL_RIC = 5700  # K
X0 = 5.6e-2
GAMMA = 1.5
CP = 750  # J/K/kg
BETA = 0.83


DTS_DTAD = 1.65
k = 200.  # W/K/m
KAPPA = k / RHO_0 / CP
print "kappa = ", KAPPA

G_PRIM = 4. / R_IC_P

### Lenghts and constants
L_RHO = np.sqrt(3. * K0 / (2. * np.pi * G * RHO_0**2.))
A_RHO = (5. * K_PRIM0 - 13.) / 10.


# Numerical parameters

N = 100  # number of steps in c (radius IC)


# Useful functions

def density(radius):
    """ Calculate the density at the radius r """
    return RHO_0 * (1. - radius**2. / L_RHO**2. - A_RHO * radius**4. / L_RHO**4.)


def gravity(radius): #not used
    """ Calculate the gravity at the radius r """
    return 4. * np.pi / 3. * G * RHO_0 * radius * \
      (1 - 3. / 5. * radius**2. / L_RHO**2. - 3. * A_RHO / 7. * radius**4. / L_RHO**4.)


def fC(radius, gamma):
    """ Calculate the approximate value of function fC(radius,gamma)
    (as defined in Labrosse 2015 eq. A.1.) """
    return radius**3. * (1 - 3. / 5. * (gamma + 1) * radius**2.
                    - 3. / 14. * (gamma + 1) * (2 * A_RHO - gamma) * radius**4.)


def f_chi(radius, ric):
    """ return the function f_chi, equation A.15 in Labrosse 2015"""
    return radius**3. * (-ric**2 / 3. / L_RHO**2.
                    + 1. / 5. * (1 + ric**2 / L_RHO**2.) * radius**2. - 13. / 70. * radius**4.)


def melting_temperature(radius):
    """ Calculate the melting temperature at the IC boundary at rIC """
    return TL0 - K0 * M_P * radius**2. / L_RHO**2. \
      + M_X * X0 * radius**3. / (L_RHO**3. * fC(R_OC / L_RHO, 0.))


def dTL_dr_ic(radius):
    """ Calculate the diff of melting temperature at the IC boundary at rIC """
    return -K0 * M_P * 2. * radius / L_RHO**2. \
      + 3. * M_X * X0 * radius**2. / (L_RHO**3. * fC(R_OC / L_RHO, 0.))


def A7int_function(radius):
    """ Voir equation A.7 in Labrosse 2015, this is the function inside the integrale """
    return radius**2. * density(radius)**(GAMMA + 1.)


def A13int_function(radius, ric): 
    """ Voir equation A.13 in Labrosse 2015, this is the function inside the integrale """
    A = radius**2. * (1 - radius**2. - A_RHO * radius**4.)\
       * (radius**2 - ric**2. /L_RHO**2) * (1. - 3. / 10. * (radius**2. + ric**2. / L_RHO**2.))
    return A


# Functions for the P (PL, PC, PX)


def power_secular_cooling(radius):
    """ from equation A7 (Labrosse 2015) """
    result, foo = scODE.quad(A7int_function, radius, R_OC)
    Pc = -4. * np.pi * CP / density(radius)**GAMMA\
        * (dTL_dr_ic(radius) + 2. * GAMMA * RHO_0 * melting_temperature(radius)
           * radius / (density(radius) * L_RHO**2.)
           * (1 + 2. * A_RHO * radius**2. / L_RHO**2.)) * result
    return Pc


def power_secular_cooling2(radius):
    """ from equation A.8 (Labrosse 2015) """
    Pc2 = -4. * np.pi / 3. * RHO_0 * CP * L_RHO**3. \
        * (1 - radius**2. / L_RHO**2 - A_RHO * radius**4. / L_RHO**4.)**(-GAMMA)\
        * (dTL_dr_ic(radius) + 2. * GAMMA * melting_temperature(radius) * radius / L_RHO**2. *
           (1 + 2. * A_RHO * radius**2. / L_RHO**2)
           / (1 - radius**2. / L_RHO**2. - A_RHO * radius**4. / L_RHO**4.)) \
        * (fC(R_OC / L_RHO, GAMMA) - fC(radius / L_RHO, GAMMA))
    return Pc2


def power_latent_heat(radius):
    """ from equation A.5 (Labrosse 2015) """
    return 4. * np.pi * radius**2. * melting_temperature(radius) * density(radius) * DELTA_S


def power_gravitational_heat(radius):
    """ from equation A.13 (Labrosse 2015)"""
    result, err = scODE.quad(A13int_function, radius /
                             L_RHO, R_OC / L_RHO, args=radius)
    return 8. * np.pi**2. * X0 * G * RHO_0**2. * BETA * radius**2.\
        * L_RHO**2 / fC(R_OC / L_RHO, 0) * result


def power_gravitational_heat2(radius):
    """ from equation A.14 (Labrosse 2015) """
    return 8 * np.pi**2 * X0 * G * RHO_0**2 * BETA * radius**2. \
      * L_RHO**2. / fC(R_OC / L_RHO, 0) \
      * (f_chi(R_OC / L_RHO, radius) - f_chi(radius / L_RHO, radius))


# Functions pour mathcal L, C, X (valeurs les plus precises)

def latent_heat(radius):
    results, err = scODE.quad(power_latent_heat, 0, radius)
    return results


def secular_cooling(radius):
    results, err = scODE.quad(power_secular_cooling, 0, radius)
    return results


def gravitiational_heat(radius):
    results, err = scODE.quad(power_gravitational_heat, 0, radius)
    return results

#########
#########


if __name__ == '__main__':

    r = R_IC_P  # Calcul pour r=r_ic

    fcOC = fC(R_OC / L_RHO, 0.)

    # liquidus temperature for P0, xi_0, P0 is pressure in the center and xi_0 the bulk composition.
    # need to be redefine (the expression below is not valid as it is TL(rOC).
    # - K0*M_P*R_IC_P**2./L_RHO**2. + M_X*X0*R_IC_P**3./(L_RHO**3.*fcOC)
    TL0 = TL_RIC
    print "TL0", TL0

    # Latent heat
    mathcalL_approx = 4. * np.pi / 3. * RHO_0 * TL0 * DELTA_S * r**3. \
        * (1 - 3. / 5. * (1 + K0 / TL0 * M_P) * r**2. / L_RHO**2.
           * X0 / (2 * fC(R_OC / L_RHO, 0.) * TL0) * M_X * r**3. / L_RHO**3.)
    mathcalL, err = scODE.quad(power_latent_heat, 0, R_IC_P)
    print "latent heat", mathcalL, mathcalL_approx

    # Secular cooling
    mathcalC_approx = 4. * np.pi / 3. * RHO_0 * CP * L_RHO * r**2 * fC(R_OC / L_RHO, GAMMA)\
        * (M_P * K0 - GAMMA * TL0 - M_X * X0 / fC(R_OC / L_RHO, 0.) * r / L_RHO)
    mathcalC, err = scODE.quad(power_secular_cooling, 0, R_IC_P)
    print "secular cooling", mathcalC, mathcalC_approx

    # Compositional energy
    mathcalX, err = scODE.quad(power_gravitational_heat, 0, R_IC_P)
    mathcalX2, err = scODE.quad(power_gravitational_heat2, 0, R_IC_P)
    print "compositional energy", mathcalX, mathcalX2

    print "Total energy", mathcalX + mathcalC + mathcalL

    # Age IC
    Aic = (mathcalL + mathcalX + mathcalC) / QCMB
    print 'Qcmb = ', QCMB, ' Watts'
    print Aic / (np.pi * 1e7) / 1e9, ' Gyrs'
    print QCMB / (power_latent_heat(r) + power_secular_cooling(r) + power_gravitational_heat(r)), ' m/s'

    plt.plot(np.linspace(7e12, 15e12, 20), (mathcalL + mathcalX +
                                            mathcalC) / np.linspace(7e12, 15e12, 20)
                                            / (np.pi * 1e7) / 1e9)


# Graphs

    Qcmb = QCMB + np.arange(-3, 7, 2) * 1e12
    print Qcmb

    f, axarr = plt.subplots(2, 1)
    f2, axarr2 = plt.subplots(2, 1)
    f3, axarr3 = plt.subplots(2, 1)

    c = np.linspace(0.1 * R_IC_P, R_IC_P, 200)

    for Q in Qcmb:

        dcdt = np.zeros(200)
        t = np.zeros(200)
        Tic = np.zeros(200)
        S = np.zeros(200)

        for i in range(0, 200):
            dcdt[i] = Q / (power_latent_heat(c[i]) + power_secular_cooling(c[i]) + power_gravitational_heat(c[i]))
            t[i] = (latent_heat(c[i]) + gravitiational_heat(c[i]) +
                    secular_cooling(c[i])) / Q / (np.pi * 1.e7 * 1.e6)
            Tic[i] = 3. * KAPPA / (DTS_DTAD - 1) * (power_latent_heat(c[i]) +
                                                    power_secular_cooling(c[i]) + power_gravitational_heat(c[i])) / Q / c[i]
            S[i] = 3 * KAPPA * RHO_0 * G_PRIM * \
                GAMMA * TL0 / K0 * (1. / Tic[i] - 1)

        tauIC = (mathcalL + mathcalX + mathcalC) / Q / (np.pi * 1.e7 * 1.e6)

        axarr[0].plot(c / 1.e3, dcdt)
        axarr[1].plot(t, c / 1.e3)
        axarr[1].plot(t, (R_IC_P) * (t / tauIC)**0.4 / 1.e3, '+')
        axarr2[0].plot(c / 1.e3, Tic)
        axarr2[0].plot(c / 1.e3, np.ones(200))
        axarr2[1].plot(c / 1.e3, S * np.pi * 1e7 * 1e9)
        axarr2[1].plot(c / 1.e3, np.zeros(200))

    axarr[0].set_title('dcdt as fn of c')

    axarr[1].set_title('c as fn of t')


    axarr2[0].set_title('Tic and S as fn of radius for different values of Q')

    for tauIC in (np.arange(200, 1800, 200) * 1.e6 * 1.e7 * np.pi):

        Tic = np.zeros(200)
        S = np.zeros(200)

        for i in range(0, 200):
            Tic[i] = 3. * KAPPA / (DTS_DTAD - 1) * (power_latent_heat(c[i]) + power_secular_cooling2(
                c[i]) + power_gravitational_heat2(c[i])) / (mathcalX + mathcalC + mathcalL) / c[i] * tauIC
            S[i] = 3 * KAPPA * RHO_0 * G_PRIM * \
                GAMMA * TL_RIC / K0 * (1. / Tic[i] - 1)

        axarr3[0].plot(c / 1.e3, Tic)
        axarr3[0].plot(c / 1.e3, np.ones(200))
        axarr3[1].plot(c / 1.e3, S * np.pi * 1e7 * 1e9)
        axarr3[1].plot(c / 1.e3, np.zeros(200))

    axarr3[0].set_title('Tic et S as fn of radius for diff age of IC')
    print TL0, melting_temperature(R_IC_P)

    Pl = np.zeros(200)
    Pc = np.zeros(200)
    Px = np.zeros(200)

    for i in range(0, 200):
        Pl[i] = latent_heat(c[i])
        Px[i] = gravitiational_heat(c[i])
        Pc[i] = secular_cooling(c[i])

    t = (Pl + Px + Pc) / QCMB / (np.pi * 1.e7 * 1.e6)

    f4, axarr4 = plt.subplots(1, 2)
    axarr4[0].plot(c / 1.e3, Pl, label="P_L")
    axarr4[0].plot(c / 1.e3, Px, label="P_x")
    axarr4[0].plot(c /
                   1.e3, Pc, label="P_c")
    axarr4[0].plot(c / 1.e3, Pl + Px + Pc, label="total")
    axarr4[1].plot(t, Pl, t, Px, t, Pc, t, Pl + Px + Pc)
    axarr4[0].set_title('energy as fn of radius (km)')
    axarr4[1].set_title('energy as fn of time (Ma)')
    axarr4[0].legend()

    plt.show()
