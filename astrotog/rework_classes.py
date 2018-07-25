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


class observations(object):
    """
    Class for the observations of simulated sources as made by a survey.
    """

    def __init__(self, transients):
        pass

    def overlap(self):

    def snr(self, pandas_table, survey):
        """

        """
        for iter, row in pandas_table.iterrows():
            for band in survey.filters:
                fluxes = np.vstack(row[band][flux_key])
                errs = np.vstack(row[band][err_key])
                Observations[key][obs_key][band]['SNR'] = np.divide(fluxes, errs)
        return Observations


class transient(object):
    """
    Base class for transients
    """
    def __init__(self):
        source = sncosmo.TimeSeriesSource(self.phase, self.wave, self.flux)
        self.model = sncosmo.Model(source=source)

    def put_in_universe(self, ra, dec, z, cosmo):
        self.ra = ra
        self.dec = dec
        redshift(z, cosmo)

    def redshift(self, z, cosmo):
        self.z = z
        lumdist = cosmo.luminosity_distance(z).value * 1e6  # in pc
        # Note that it is necessary to scale the amplitude relative to the 10pc
        # (i.e. 10^2 in the following eqn.) placement of the SED currently
        self.model.set(z=z, amplitude=pow(np.divide(10.0, lumdist), 2))


class kilonovae(transient):
    """
    Base class for kilonovae transients
    """
    def __init__(self):
        self.type = 'kne'
        super().__init__()


class survey(object):
    """ Class for instrument parameters and the simulated cadence which
    together form the survey.

    Attributes
    ----------
    cadence : Pandas dataframe
        The survey strategy which defines how the sky is observed.
    throughputs : dict of np.arrays
        Dictionary with an entry for each band filter of the instrument, where
        the entries consist of two np.arrays. One array is the sequence of
        wavelengths for which the throughput is defined, and the second array
        is the fractional throughput at each wavelength.
    reference_flux : dict of np.scalars
        Dictionary with an entry for each band filter of the intsrument, where
        each entry is the bandflux for the reference system chosen given the
        response of that band filter.
    survey_parameters : Pandas dataframe
        Dataframe containing basic information about the survey needed to
        simulate transient observations.

    """

    def __init__(self, cadence_path, throughputs_path, reference_path,
                 cadence_flags):
        """
        Builds the survey object calling the methods to set the base attributes
        of the class.
        """
        get_cadence(cadence_path, cadence_flags)
        get_throughputs(throughputs_path)
        get_reference_flux(reference_path)
        get_survey_params()

    def get_cadence(self, path, flag='combined'):
        """
        Method to get the survey cadence from an OpSim database.


        Parameters
        ----------
        path : str
            The path to the database file for the simulated survey.
        flag: str
            Flag for OpSimSummary to return specificed subset of the survey.
            The default is combined which returns the whole survey.
        Returns
        -------
        cadence : Pandas dataframe
            Dataframe containing the summary elements of each visit that is
            simulated in OpSim for the chosen survey.
        """
        self.cadence = oss.OpSimOutput.fromOpSimDB(path, subset=flag).summary

    def get_throughputs(self, path):
        """
        Method to obtain the throughput response for each band filter of the
        instrument, as selected by the provided path.


        Parameters
        ----------
        path : str
            The path to the directory containing the throughputs files for the
            instrument.

        Returns
        -------
        throughputs : dict of two np.array
            List of the throughput efficiency and associated wavelengths for
            each band filter in the directory.

        """
        self.throughputs = {}
        tp_filelist = os.listdir(path)
        for band_file_name in tp_filelist:
            band = band_file_name.strip('.dat')
            self.throughputs[band] = {}
            throughput_file = path + '/' + band_file_name
            band_wave = list()
            band_throughput = list()
            # Get the particular band throughput
            band_file = open(throughput_file, 'r')
            for line in band_file:
                # Strip header comments
                if line.strip().startswith("#"):
                    # check to see what units the wavelengths are in
                    nano_match = re.search('nm|nanometer|Nanometer|Nm', line)
                    ang_match = re.search(u'\u212B|Ang|Angstrom|angstrom', line)
                    mic_match = re.search(u'\u03BC|um|micrometer|micron|', line)
                    if nano_match:
                        conversion = 10.0  # conversion for nanometers to Angstrom
                    elif ang_match:
                        conversion = 1.0
                    elif mic_match:
                        conversion = 1e4
                    continue
                else:
                    # Strip per line comments
                    comment_match = re.match(r'^([^#]*)#(.*)$', line)
                    if comment_match:  # The line contains a hash / comment
                        line = comment_match.group(1)
                    line = line.strip()
                    split_fields = re.split(r'[ ,|;"]+', line)
                    band_wave.append(conversion*float(split_fields[0]))
                    band_throughput.append(float(split_fields[1]))
            band_file.close()
            band_wave = np.asarray(band_wave)
            band_throughput = np.asarray(band_throughput)
            self.throughputs[band]['wavelengths'] = band_wave
            self.throughputs[band]['throughput'] = band_throughput

    def get_reference_flux(self, path, magsys='ab'):
        """
        Method to get the reference flux response for a given magnitude system
        in the filter bands that are defined for chosen survey instrument, as
        defined by the given file path.

        Parameters
        ----------
        path : str
            The path to the directory containing the throughputs files for the
            instrument.
        magsys : str
            This is the magnitude system used for the instrument. Assumed to be
            the AB magnitude system.

        Returns
        -------
        throughputs : dict of two np.array
            List of the throughput efficiency and associated wavelengths for
            each band filter in the directory.

        """
        ref_wave = list()
        ref_flux_per_wave = list()
        # Add a line to the phase object too to use with sncosmo
        phase_for_ref = np.arange(0.5, 5.0, step=0.5)
        ref_filepath = os.path.join(path, '{0}.dat'.format(magsys))
        ref_file = open(ref_filepath, 'r')
        for line in ref_file:
            # Strip header comments
            if line.strip().startswith("#"):
                continue
            else:
                # Strip per line comments
                comment_match = re.match(r'^([^#]*)#(.*)$', line)
                if comment_match:  # The line contains a hash / comment
                    line = comment_match.group(1)
                line = line.strip()
                split_fields = re.split(r'[ ,|;"]+', line)
                ref_wave.append(float(split_fields[0]))
                ref_flux_per_wave.append(float(split_fields[1]))
        ref_file.close()
        # Convert to arrays for use with the sncosmo model
        phase_for_ref = np.asarray(phase_for_ref)
        ref_wave = np.asarray(ref_wave)
        ref_flux_per_wave = np.asarray(ref_flux_per_wave)
        ref_flux_per_wave_for_model = np.empty([len(phase_for_ref),
                                                len(ref_wave)])

        # Fake multi-phase observations to exploit the sncosmo model object
        for i, phase in enumerate(phase_for_ref):
                ref_flux_per_wave_for_model[i, :] = ref_flux_per_wave
        # Put throughput and reference on the same wavelength grid
        # Exploit sncosmo functionality to do this
        ref_source = sncosmo.TimeSeriesSource(phase_for_ref, ref_wave,
                                              ref_flux_per_wave_for_model)
        ref_model = sncosmo.Model(source=ref_source)
        self.reference_flux_response = {}
        for band in self.throughputs.keys():
            self.reference_flux_response[band] = \
                Compute_Bandflux(band=band, throughputs=self.throughputs,
                                 ref_model=ref_model)

    def get_survey_params(self):
        """
        Method to obtain summary features of the given cadence.
        """
        # Given a prescribed survey simulation get basic properties of the
        # simulation. Currently assume a rectangular (in RA,DEC) solid angle on
        # the sky
        # Note that the values are assumed to be in radians
        min_db = self.cadence.min()
        max_db = self.cadence.max()
        # Need to extend by FOV
        self.min_ra = min_db['ditheredRA']
        self.max_ra = max_db['ditheredRA']
        self.min_dec = min_db['ditheredDec']
        self.max_dec = max_db['ditheredDec']
        # Survey area in degrees squared
        self.survey_area = np.rad2deg(np.sin(self.max_dec) - np.sin(self.min_dec))*np.rad2deg(self.max_ra - self.min_ra)
        self.min_mjd = min_db['expMJD']
        self.max_mjd = max_db['expMJD']
        self.survey_time = self.max_mjd - self.min_mjd  # Survey time in days


class transient_distribution(object):
    """
    Class representing the cosmological distribution of events happening
    in the universe.
    """
    def __init__(self, survey, rate, cosmo):
        self.rate = lambda x: rate/pow(1000, 3)
        redshift_dist(survey, cosmo)
        sky_location_dist(survey)
        time_dist(survey)

    def redshift_dist(self, survey, cosmology):
        # Internal funciton to generate a redshift distribution
        # Given survey parameters, a SED rate, and a cosmology draw from a Poisson
        # distribution the distribution of the objects vs. redshift.
        zlist = list(sncosmo.zdist(zmin=param_priors['zmin'], zmax=param_priors['zmax'],
                                   time=survey.survey_time, area=survey.survey_area,
                                   ratefunc=self.rate, cosmo=cosmology))
        self.redshift_list = zlist
        self.number_simulated = len(SED_zlist)

    def sky_location_dist(self, survey):
        # For given survey paramters distribute random points within the (RA,DEC)
        # space. Again assuming a uniform RA,Dec region
        self.ra_dist = np.random.uniform(survey.min_ra, survey.max_ra,
                                         self.number_simulated)
        self.dec_dist = np.arcsin(np.random.uniform(low=np.sin(survey.min_dec),
                                  high=np.sin(survey.max_dec),
                                  size=self.number_simulated))

    def time_dist(self, survey):
        self.time_dist = np.random.uniform(low=survey.min_mjd,
                                           high=survey.max_mjd,
                                           self.number_simulated)

    # def transient_instances(self):
    #     self.transients = []
    #     for i in range(self.number_process):


class detections(object):
    """
    Class for collecting information about transients that are observed and
    pass the criteria for detection.
    """
    def __init__(self):
        pass
