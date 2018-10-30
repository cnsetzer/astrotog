import os
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import sncosmo
from astropy.cosmology import Planck15 as cosmo
from LSSTmetrics.efficiencyTable import EfficiencyTable as eft
from .macronovae_wrapper import make_rosswog_seds as mw
from .classes import kilonova
from .classes import survey


class simulation(object):
    """
    Top-level class that represents the desired run_simulation
    """
    def __init__(self, cadence_path, dither_path, throughputs_path,
                 reference_path, z_max, output_path='.',
                 cadence_flags='combined', z_min=0.0,
                 z_bin_size=0.01, batch_size='all', cosmology=cosmo,
                 rate_gpc=1000, dithers=True, simversion='lsstv4',
                 add_dithers=False, t_before=30.0, t_after=30.0,
                 response_path=None, instrument=None, ra_col='_ra',
                 dec_col='_dec', filter_null=False, desc_dithers=False,
                 same_dist=False, min_dec=-np.pi/2.0,
                 max_dec=np.pi/6.0, trans_duration=30.0):
        self.cadence_path = cadence_path
        self.throughputs_path = throughputs_path
        self.reference_path = reference_path
        self.output_path = output_path
        self.cadence_flags = cadence_flags
        self.z_min = z_min
        self.z_max = z_max
        self.z_bin_size = z_bin_size
        self.batch_size = batch_size
        self.cosmology = cosmology
        self.rate = rate_gpc
        self.dithers = dithers
        self.version = simversion
        self.add_dithers = add_dithers
        self.t_before = t_before
        self.t_after = t_after
        self.response_path = response_path
        self.instrument = instrument
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.filter_null = filter_null
        self.desc_dithers = desc_dithers
        self.dither_path = dither_path
        self.same_dist = same_dist
        self.min_dec = min_dec
        self.max_dec = max_dec
        self.transient_duration = trans_duration


class lsst(survey):
    """
    Top-level class for the LSST instrument and survey.
    """
    def __init__(self, simulation):

        self.FOV_radius = np.deg2rad(1.75)
        self.instrument = 'lsst'
        self.magsys = 'ab'
        self.filters = ['u', 'g', 'r',
                        'i', 'z', 'y']
        self.dust_corrections = {'u': 4.145, 'g': 3.237, 'r': 2.273,
                                 'i': 1.684, 'z': 1.323, 'y': 1.088}
        self.utc_offset = -3.0 #Specify the UTC offset in hours INCLUDE the +/-
        self.response_function(simulation.response_path)
        super().__init__(simulation)

    def response_function(self, response_path):
        self.detect_table = eft.fromDES_EfficiencyFile(response_path)


class wfirst(survey):
    """
    Top-level class for the LSST instrument and survey.
    """
    def __init__(self, simulation):

        self.FOV_radius = np.deg2rad(1.75)
        self.instrument = 'wfirst'
        self.magsys = 'ab'
        self.filters = ['u', 'g', 'r',
                        'i', 'z', 'y']
        self.dust_corrections = {'u': 4.145, 'g': 3.237, 'r': 2.273,
                                 'i': 1.684, 'z': 1.323, 'y': 1.088}

        self.utc_offset = 0.0 #Specify the UTC offset in hours INCLUDE the +/-
        self.response_function(simulation.response_path)
        super().__init__(simulation)

    def response_function(self, response_path):
        self.detect_table = eft.fromDES_EfficiencyFile(response_path)


class ztf(survey):
    """
    Top-level class for the LSST instrument and survey.
    """
    def __init__(self, simulation):

        self.FOV_radius = np.deg2rad(1.75)
        self.instrument = 'wfirst'
        self.magsys = 'ab'
        self.filters = ['u', 'g', 'r',
                        'i', 'z', 'y']
        self.dust_corrections = {'u': 4.145, 'g': 3.237, 'r': 2.273,
                                 'i': 1.684, 'z': 1.323, 'y': 1.088}

        self.utc_offset = -3.0 #Specify the UTC offset in hours INCLUDE the +/-
        self.response_function(simulation.response_path)
        super().__init__(simulation)

    def response_function(self, response_path):
        self.detect_table = eft.fromDES_EfficiencyFile(response_path)


class saee_nsns(kilonova):
    """
    Top-level class for kilonovae transients based on Rosswog, et. al 2017
    semi-analytic model for kilonovae spectral energy distributions.
    """
    def __init__(self, mej=None, vej=None, kappa=None, bounds=None,
                 uniform_v=False, KNE_parameters=None, parameter_dist=False,
                 num_samples=1, probability=0.5):
        self.num_params = 3
        if parameter_dist is True:
            if num_samples > 1:
                self.pre_dist_params = True
                self.number_of_samples = num_samples
                self.draw_parameters(bounds, uniform_v, probability)
            else:
                print('To generate a parameter distribution you need to supply\
                        a number of samples greater than one.')
                exit()
            self.subtype = 'rosswog semi-analytic'
            self.type = 'parameter distribution'
        else:
            self.number_of_samples = num_samples
            if (mej and vej and kappa):
                self.param1 = mej
                self.param1_name = 'm_ej'
                self.param2 = vej
                self.param2_name = 'v_ej'
                self.param3 = kappa
                self.param3_name = 'kappa'
            elif KNE_parameters:
                pass
            else:
                self.draw_parameters(bounds, uniform_v, probability)
            self.make_sed(KNE_parameters)
            self.subtype = 'rosswog semi-analytic'
            super().__init__()

    def draw_parameters(self, bounds=None, uniform_v=False, probability=0.5):
        # Set the parameter names
        self.param1_name = 'm_ej'
        self.param2_name = 'v_ej'
        self.param3_name = 'kappa'

        if bounds is not None:
            kappa_min = min(bounds['kappa'])
            kappa_max = max(bounds['kappa'])
            mej_min = min(bounds['m_ej'])
            mej_max = max(bounds['m_ej'])
            vej_min = min(bounds['v_ej'])
            vej_max = max(bounds['v_ej'])
            probability = bounds['orientation_probability']
        else:
            # Set to default values from Rosswog's paper
            kappa_min = 1.0
            kappa_max = 10.0
            mej_min = 0.01
            mej_max = 0.2
            vej_min = 0.01
            vej_max = 0.5

        # Determine output shape for parameters based on size
        if self.number_of_samples > 1:
            out_shape = (self.number_of_samples, 1)
        else:
            out_shape = self.number_of_samples

        self.param3 = np.random.binomial(1, probability, size=out_shape)*(kappa_max - kappa_min) + kappa_min
        self.param1 = np.random.uniform(low=mej_min, high=mej_max,
                                        size=out_shape)
        if uniform_v is False:
            self.param2 = 1.1*np.ones(shape=out_shape)
            ind = np.where(self.param2 > vej_max*pow(self.param1/mej_min, np.log10(0.25/vej_max)/np.log10(mej_max/mej_min)))
            while len(ind) > 0:
                int_shape = shape(self.param2[ind])
                self.param2[ind] = np.random.uniform(low=vej_min, high=vej_max,
                                                size=int_shape)
                ind = np.where(self.param2 > vej_max*pow(self.param1/mej_min, np.log10(0.25/vej_max)/np.log10(mej_max/mej_min)))
        else:
            self.param2 = np.random.uniform(low=vej_min, high=vej_max,
                                            size=out_shape)

    def make_sed(self, KNE_parameters=None):
        if KNE_parameters is None:
            KNE_parameters = []
            KNE_parameters.append(0.00001157)
            KNE_parameters.append(60.0)
            KNE_parameters.append(self.param1)
            KNE_parameters.append(self.param2)
            KNE_parameters.append(1.3)
            KNE_parameters.append(0.25)
            KNE_parameters.append(1.0)
            KNE_parameters.append(self.param3)
            KNE_parameters.append(150.0)
            # Not reading heating rates from file so feed fortran dummy
            # variables
            KNE_parameters.append(False)
            KNE_parameters.append('dummy string')
        self.phase, self.wave, self.flux = mw(KNE_parameters,
                                              separated=True)


class rosswog_numerical_kilonova(kilonova):
    """
    Top-level class for kilonovae transients based on Rosswog, et. al 2017
    numerically generated kilonovae spectral energy distributions.
    """
    def __init__(self, path=None, singleSED=None, parameter_dist=False,
                 num_samples=1):
        self.number_of_samples = num_samples
        self.num_params = 3
        self.pre_dist_params = False
        self.param1_name = 'm_ej'
        self.param2_name = 'v_ej'
        self.param3_name = 'kappa'

        if parameter_dist is True:
            self.subtype = 'rosswog numerical'
            self.type = 'parameter distribution'

        else:
            self.make_sed(path, singleSED)
            self.subtype = 'rosswog numerical'
            super().__init__()

    def make_sed(self, path, singleSED):
        if not singleSED:
            # Get the list of SED files
            fl = os.listdir(path)
            # Randomly select SED
            rindex = np.random.randint(low=0, high=len(fl))
            filei = fl[rindex]
            filename = path + '/' + filei
            fileio = open(filename, 'r')
        else:
            filename = path
            fileio = open(filename, 'r')
        self.SED_header_params(fileio)
        fileio.close()
        # Read in SEDS data with sncosmo tools
        self.phase, self.wave, self.flux = sncosmo.read_griddata_ascii(filename)

    def SED_header_params(self, fileio):
        # Read header for parameter data for model (Specific for Rosswog)
        for headline in fileio:
            if headline.strip().startswith("#"):
                if re.search("kappa =", headline):
                    self.param3 = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                elif re.search("m_ej = |m_w =", headline):
                    self.param1 = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                elif re.search("v_ej = |v_w =", headline):
                    self.param2 = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                else:
                    continue
            else:
                break


class metzger_kilonova(kilonova):
    """
    Top-level class for kilonovae transients based on Scolnic, et. al 2017
    model for kilonovae spectral energy distribution mimicing the GW170817
    event.
    """
    def __init__(self, parameter_dist=False, num_samples=1):
        self.number_of_samples = num_samples
        self.num_params = 0
        self.make_sed()
        self.subtype = 'rosswog semi-analytic'
        super().__init__()

    def make_sed(self):
        self.phase, self.wave, self.flux = SED_FUNCTION()


class cowperthwaite_kilonova(kilonova):
    """
    Top-level class for kilonovae transients based on Scolnic, et. al 2017
    model for kilonovae spectral energy distribution mimicing the GW170817
    event.
    """
    def __init__(self, parameter_dist=False, num_samples=1):
        self.number_of_samples = num_samples
        self.num_params = 0
        self.make_sed()
        self.subtype = 'rosswog semi-analytic'
        super().__init__()

    def make_sed(self):
        self.phase, self.wave, self.flux = SED_FUNCTION()


class kasen_kilonova(kilonova):
    """
    Top-level class for kilonovae transients based on Scolnic, et. al 2017
    model for kilonovae spectral energy distribution mimicing the GW170817
    event.
    """
    def __init__(self, parameter_dist=False, num_samples=1):
        self.number_of_samples = num_samples
        self.num_params = 0
        self.make_sed()
        self.subtype = 'rosswog semi-analytic'
        super().__init__()

    def make_sed(self):
        self.phase, self.wave, self.flux = SED_FUNCTION()


class desgw_kne(kilonova):
    """
    Top-level class for kilonovae transients based on Scolnic, et. al 2017
    model for kilonovae spectral energy distribution mimicing the GW170817
    event.
    """
    def __init__(self, path=None, parameter_dist=False, num_samples=1):
        self.number_of_samples = num_samples
        self.num_params = 0
        self.pre_dist_params = False
        self.subtype = 'scolnic empirical'
        if parameter_dist is True:
            self.type = 'parameter distribution'
        else:
            self.make_sed(path)
            super().__init__()

    def make_sed(self, path):
        self.phase, self.wave, self.flux = sncosmo.read_griddata_ascii(path)


class saee_nsbh(kilonova):
    """
    Top-level class for kilonovae transients based on Rosswog, et. al 2017
    semi-analytic model for kilonovae spectral energy distributions.
    """
    def __init__(self, mej=None, vej=None, kappa=None, bounds=None,
                 uniform_v=True, KNE_parameters=None, parameter_dist=False,
                 num_samples=1, probability=0.5):
        self.num_params = 3
        if parameter_dist is True:
            if num_samples > 1:
                self.pre_dist_params = True
                self.number_of_samples = num_samples
                self.draw_parameters(bounds, uniform_v, probability)
            else:
                print('To generate a parameter distribution you need to supply\
                        a number of samples greater than one.')
                exit()
            self.subtype = 'saee'
            self.type = 'parameter distribution'
        else:
            self.number_of_samples = num_samples
            if (mej and vej and kappa):
                self.param1 = mej
                self.param1_name = 'm_ej'
                self.param2 = vej
                self.param2_name = 'v_ej'
                self.param3 = kappa
                self.param3_name = 'kappa'
            elif KNE_parameters:
                pass
            else:
                self.draw_parameters(bounds, uniform_v, probability)
            self.make_sed(KNE_parameters)
            self.subtype = 'saee'
            super().__init__()

    def draw_parameters(self, bounds=None, uniform_v=False, probability=0.5):
        # Set the parameter names
        self.param1_name = 'm_ej'
        self.param2_name = 'v_ej'
        self.param3_name = 'kappa'

        if bounds is not None:
            kappa_min = min(bounds['kappa'])
            kappa_max = max(bounds['kappa'])
            mej_min = min(bounds['m_ej'])
            mej_max = max(bounds['m_ej'])
            vej_min = min(bounds['v_ej'])
            vej_max = max(bounds['v_ej'])
            probability = bounds['orientation_probability']
        else:
            kappa = 10.0
            mej_min = 0.05
            mej_max = 0.2
            vej_min = 0.1
            vej_max = 0.25

        # Determine output shape for parameters based on size
        if self.number_of_samples > 1:
            out_shape = (self.number_of_samples, 1)
        else:
            out_shape = self.number_of_samples

        self.param3 = np.ones(shape=out_shape)*kappa
        self.param1 = np.random.uniform(low=mej_min, high=mej_max,
                                        size=out_shape)
        if uniform_v is False:
            self.param2 = 1.1*np.ones(shape=out_shape)
            ind = np.where(self.param2 > vej_max*pow(self.param1/mej_min, np.log10(0.25/vej_max)/np.log10(mej_max/mej_min)))
            while len(ind) > 0:
                int_shape = shape(self.param2[ind])
                self.param2[ind] = np.random.uniform(low=vej_min, high=vej_max,
                                                size=int_shape)
                ind = np.where(self.param2 > vej_max*pow(self.param1/mej_min, np.log10(0.25/vej_max)/np.log10(mej_max/mej_min)))
        else:
            self.param2 = np.random.uniform(low=vej_min, high=vej_max,
                                            size=out_shape)

    def make_sed(self, KNE_parameters=None):
        if KNE_parameters is None:
            KNE_parameters = []
            KNE_parameters.append(0.00001157)
            KNE_parameters.append(60.0)
            KNE_parameters.append(self.param1)
            KNE_parameters.append(self.param2)
            KNE_parameters.append(1.3)
            KNE_parameters.append(0.25)
            KNE_parameters.append(1.0)
            KNE_parameters.append(self.param3)
            KNE_parameters.append(150.0)
            # Not reading heating rates from file so feed fortran dummy
            # variables
            KNE_parameters.append(False)
            KNE_parameters.append('dummy string')
        self.phase, self.wave, self.flux = mw(KNE_parameters,
                                              separated=True)
