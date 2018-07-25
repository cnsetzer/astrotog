import os
import re
import datetime
import csv
import numpy as np
import sncosmo
import pandas as pd
from scipy.integrate import simps
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import opsimsummary as oss
import seaborn as sns
# from astrotog import macronovae_wrapper as mw
import macronovae_wrapper as mw

# font = {'size': 14}
# matplotlib.rc('font', **font)
sns.set_style('whitegrid')  # I personally like this style.
sns.set_context('talk')  # Easy to change context from `talk`, `notebook`, `poster`, `paper`. though further fine tuning is human.
# set seed
np.random.seed(12345)


class simulation(object):
    """
    Top-level class that represents the desired run_simulation
    """
    def __init__(self):
        self.cadence_path =
        self.throughputs_path =
        self.reference_path =
        self.output_path =
        self.cadence_flags =
        self.z_min =
        self.z_max =
        self.z_bin_size =
        self.multiprocess =


class LSST(survey):
    """
    Top-level class for the LSST instrument and survey.
    """
    def __init__(self, cadence_path, throughputs_path, reference_path,
                 cadence_flags, instrument_params=None):
        if instrument_params is None:
            self.FOV_radius = np.deg2rad(1.75)
            self.instrument = 'lsst'
            self.magsys = 'ab'
            self.filters = ['lsstu', 'lsstg', 'lsstr',
                            'lssti', 'lsstz', 'lssty']
        else:
            self.FOV_radius = instrument_params['fov_rad']
            self.instrument = instrument_params['instrument']
            self.magsys = instrument_params['magsys']
            self.filters = instrument_params['filters']

        super().__init__(cadence_path, throughputs_path, reference_path,
                         cadence_flags)


class rosswog_kilonovae(kilonovae):
    """
    Top-level class for kilonovae transients based on Rosswog, et. al 2017
    semi-analytic model for kilonovae spectral energy distributions.
    """
    def __init__(self, mej=None, vej=None, kappa=None, bounds=None,
                 uniform_v=False, KNE_parameters=None):
        if (mej and vej and kappa):
            self.m_ej = mej
            self.v_ej = vej
            self.kappa = kappa
        elif KNE_parameters:
            pass
        else:
            draw_parameters(bounds, uniform_v)
        make_sed(KNE_parameters)
        self.subtype = 'rosswog semi-analytic'
        super().__init__()

    def draw_parameters(self, bounds=None, uniform_v=False):
        if bounds is not None:
            kappa_min = min(bounds['kappa'])
            kappa_max = max(bounds['kappa'])
            mej_min = min(bounds['m_ej'])
            mej_max = max(bounds['m_ej'])
            vej_min = min(bounds['v_ej'])
            vej_max = max(bounds['v_ej'])
        else:
            # Set to default values from Rosswog's paper
            kappa_min = 1.0
            kappa_max = 10.0
            mej_min = 0.01
            mej_max = 0.2
            vej_min = 0.01
            vej_max = 0.5

        self.kappa = np.random.uniform(low=kappa_min, high=kappa_max)
        self.m_ej = np.random.uniform(low=mej_min, high=mej_max)
        if uniform_v is False:
            self.v_ej = np.random.uniform(low=vej_min, high=vej_max*pow(p[key]['m_ej']
                                          / mej_min, np.log10(0.25/vej_max) / np.log10(mej_max/mej_min)))
        else:
            self.v_ej = np.random.uniform(low=vej_min, high=vej_max)

    def make_sed(self, KNE_parameters=None):
        if KNE_parameters is None:
            KNE_parameters = []
            KNE_parameters.append(0.00001157)
            KNE_parameters.append(50.0)
            KNE_parameters.append(self.m_ej)
            KNE_parameters.append(self.v_ej)
            KNE_parameters.append(1.3)
            KNE_parameters.append(0.25)
            KNE_parameters.append(1.0)
            KNE_parameters.append(self.kappa)
            KNE_parameters.append(150.0)
            # Not reading heating rates from file so feed fortran dummy
            # variables
            KNE_parameters.append(False)
            KNE_parameters.append('dummy string')
        self.phase, self.wave, self.flux = mw.Make_Rosswog_SEDS(KNE_parameters,
                                                                separated=True)


class rosswog_numerical_kilonovae(kilonovae):
    """
    Top-level class for kilonovae transients based on Rosswog, et. al 2017
    numerically generated kilonovae spectral energy distributions.
    """
    def __init__(self, path):
        make_sed(path)
        self.subtype = 'rosswog numerical'
        super().__init__()

    def make_sed(self, path_to_seds):
        # Get the list of SED files
        fl = os.listdir(path_to_seds)
        # Randomly select SED
        rindex = np.random.randint(low=0, high=len(fl))
        filei = fl[rindex]
        filename = path_to_seds + '/' + filei
        fileio = open(filename, 'r')
        SED_header_params(fileio)
        # Read in SEDS data with sncosmo tools
        self.phase, self.wave, self.flux = sncosmo.read_griddata_ascii(filename)

    def SED_header_params(self, fileio):
        # Read header for parameter data for model (Specific for Rosswog)
        for headline in fileio:
            if headline.strip().startswith("#"):
                if re.search("kappa =", headline):
                    self.kappa = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                elif re.search("m_ej = |m_w =", headline):
                    self.m_ej = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                elif re.search("v_ej = |v_w =", headline):
                    self.v_ej = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                else:
                    continue
            else:
                fileio.close()
                break
