import numpy as np
from math import *
from macronova2py import macronova2py as m2p


def Make_Rosswog_SEDS(KNE_parameters, separated=False):
    Nt = 2000
    n = len(KNE_parameters) - 2
    MNE_parameters = KNE_parameters[0:n]
    read_hrate = KNE_parameters[n]
    heating_rates_file = KNE_parameters[n+1]
    luminosity = m2p.calculate_luminosity(n, MNE_parameters, read_hrate,
                                          heating_rates_file, Nt)
    return SED_timeseries(luminosity, separated)


def SED_timeseries(luminosity, separated=False):
    # --------------------------------------------------------------------------
    # Planck distribution
    # normalized to sigma*T**4/pi

    #####################
    # physics constants #
    #####################
    hplanck = 6.62607004e-27     # Planck constant [erg*s]
    kB = 1.38064852e-16     # Boltzmann constant [erg/K]
    parsec = 3.08567758e+18     # parsec [cm]
    clight = 2.99792458e10      # speed of light [cm/s]
    sigma = 5.67051e-5         # Stefan-Boltzmann constant [erg/(s*cm2*K4)]
    msol = 1.9891e+33         # solar mass
    day_in_s = 8.64e+4            # one day [s]
    delta_t = 0.5*day_in_s       # time step
    Ang = 1.e-8              # angstrom [cm]
    Robs = 10.0*parsec        # distance for absolute magnitudes

    ####################
    # other parameters #
    ####################
    x_cut = 100.              # cut value for Planck function argument
    t_start_d = 1.e-4             # start time [d]
    t_end_d = 30.               # end time [d]
    delta_dex = 0.05              # spacing of time grid in log10 space

    # time grid parameters
    # starting and ending times
    (t_begin, t_end) = (t_start_d*day_in_s, t_end_d*day_in_s)
    itmax = int(log10(t_end/t_begin)/delta_dex)               # of iterations

    ti = luminosity[:, 0]          # times
    Li = luminosity[:, 1]          # luminosities
    Ti = luminosity[:, 2]          # Temperatures
    Ri = luminosity[:, 3]          # radii
    vi = Ri[:]/(clight*(ti[:]+1e-10))          # velocities
    nl = len(ti)
    # allocate regridded arrays
    tim = np.zeros(itmax+1)
    lum = np.zeros(itmax+1)
    tef = np.zeros(itmax+1)
    rad = np.zeros(itmax+1)

    # interpolation loop
    t = t_begin
    i = 0
    dfac = 10.0**delta_dex
    for it in range(0, itmax+1):
        tim[it] = t
        while (ti[i] < t and i < nl-1):
            i = i + 1
        (t1, t2) = (ti[i-1], ti[i])
        (L1, L2) = (Li[i-1], Li[i])
        (T1, T2) = (Ti[i-1], Ti[i])
        (R1, R2) = (Ri[i-1], Ri[i])
        (v1, v2) = (vi[i-1], vi[i])
        #
        lum[it] = L1 + (t-t1)/(t2-t1)*(L2-L1)
        tef[it] = T1 + (t-t1)/(t2-t1)*(T2-T1)
        rad[it] = R1 + (t-t1)/(t2-t1)*(R2-R1)
        #
        t = t*dfac

    # output
    wavelengths = range(1000, 25010, 10)
    sed_data_struct = np.zeros(((itmax+1)*len(wavelengths), 3))

    for it in range(0, itmax+1):
        # compute spectral normalization coefficient
        Coef = lum[it] / (4*Robs**2 * sigma*tef[it]**4)

        # output spectral flux f_lambda [erg s^-1 cm^-2 A^-1]
        for i, lam_A in enumerate(wavelengths):
            lam_cm = lam_A * Ang
            f_lm = Coef * Blam(lam_cm, tef[it]) * Ang
            sed_data_struct[it*len(wavelengths)+i, :] = tim[it]/day_in_s, \
                lam_A, f_lm

    if separated is True:
        phase = np.unique(sed_data_struct[:, 0])
        wave = np.unique(sed_data_struct[:, 1])
        flux = np.zeros((len(phase), len(wave)))
        for i, phz in enumerate(phase):
            for j, wv in enumerate(wave):
                flux[i, j] = sed_data_struct[i*len(wave) + j, 2]

        return phase, wave, flux
    else:
        return sed_data_struct


def Blam(lam, T):
    #####################################
    # Planck function, lambda[cm] input #
    #####################################
    # argument of Planck function
    #####################
    # physics constants #
    #####################
    hplanck = 6.62607004e-27     # Planck constant [erg*s]
    kB = 1.38064852e-16     # Boltzmann constant [erg/K]
    clight = 2.99792458e10      # speed of light [cm/s]
    x_cut = 100.0

    x = hplanck*clight/(kB*T*lam)

    if(x > x_cut):
        return 0.0
    else:
        return ((2.0*hplanck*clight**2)/lam**5) / (exp(x) - 1.0)
