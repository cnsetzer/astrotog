__all__ = ["transient_distribution"]
import os
import re
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from astropy.constants import c as speed_of_light_ms
import sncosmo
from copy import deepcopy
import opsimsummary as oss
from .functions import bandflux

speed_of_light_kms = speed_of_light_ms.to("km/s").value  # Convert m/s to km/s


class transient(object):
    """
    Base class for transients
    """

    def __init__(self):
        self.extend_sed_waves()
        source = sncosmo.TimeSeriesSource(self.phase, self.wave, self.flux)
        self.model = deepcopy(
            sncosmo.Model(source=source)
        )  # Use deepcopy to make sure full class is saved as attribute of new class

    def put_in_universe(self, id, t, ra, dec, z, cosmo):
        self.id = int(id)
        self.t0 = t
        self.ra = ra
        self.dec = dec
        self.z = z
        self.peculiar_velocity()
        self.redshift(cosmo)
        self.tmax = t + self.model.maxtime()

        print(
            "Debug: The peak apparent magnitude in lsstz is {}".format(
                self.model.source_peakmag("lsstz", "ab", sampling=0.1)
            )
        )

        return self

    def redshift(self, cosmo):
        lumdist = cosmo.luminosity_distance(self.z).value * 1e6  # in pc
        amp = pow(np.divide(10.0, lumdist), 2)
        # Note that it is necessary to scale the amplitude relative to the 10pc
        # (i.e. 10^2 in the following eqn.) placement of the SED currently
        self.model.set(z=self.obs_z)
        self.model.set(amplitude=amp)

        # Current working around for issue with amplitude...
        # mapp = cosmo.distmod(self.z).value + self.model.source_peakmag(
        #    "lsstz", "ab", sampling=0.1
        # )
        # self.model.set_source_peakmag(m=mapp, band="lsstz", magsys="ab", sampling=0.1)

    def peculiar_velocity(self):
        # Draw from gaussian peculiar velocity distribution with width 300km/s
        # Hui and Greene (2006)
        state = np.random.get_state()
        np.random.seed(seed=self.id)
        self.peculiar_vel = np.random.normal(loc=0, scale=300)
        np.random.set_state(state)
        self.obs_z = (1 + self.z) * (
            np.sqrt(
                (1 + (self.peculiar_vel / speed_of_light_kms))
                / ((1 - (self.peculiar_vel / speed_of_light_kms)))
            )
        ) - 1.0

    def extend_sed_waves(self):
        min_wave = np.min(self.wave)
        max_wave = np.max(self.wave)
        delta_wave = abs(self.wave[0] - self.wave[1])
        extension_flux = 0.0
        new_max = 50000.0
        new_min = 50.0

        # insert new max and min
        if max_wave < new_max:
            for wave in np.arange(
                start=max_wave + delta_wave, stop=new_max + delta_wave, step=delta_wave
            ):
                self.wave = np.insert(self.wave, self.wave.size, wave, axis=0)
                self.flux = np.insert(
                    self.flux, self.flux.shape[1], extension_flux, axis=1
                )

        if min_wave > new_min:
            for wave in np.arange(
                start=min_wave - delta_wave, stop=new_min - delta_wave, step=-delta_wave
            ):
                self.wave = np.insert(self.wave, 0, wave, axis=0)
                self.flux = np.insert(self.flux, 0, extension_flux, axis=1)


class kilonova(transient):
    """
    Base class for kilonovae transients
    """

    def __init__(self):
        self.type = "kne"
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

    def __init__(self, simulation):
        """
        Builds the survey object calling the methods to set the base attributes
        of the class.
        """
        self.dithered = simulation.dithers
        self.get_cadence(simulation)
        self.get_throughputs(simulation)
        self.get_reference_flux(simulation.reference_path)
        self.get_survey_params(simulation)

    def get_cadence(self, simulation):
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
        path = simulation.cadence_path
        flag = simulation.cadence_flags
        vers = simulation.version
        add_dith = simulation.add_dithers
        null_option = simulation.filter_null

        if simulation.dithers is True:
            if simulation.desc_dithers is True:
                dither_table = pd.read_csv(simulation.dither_path)
                dither_table.rename(
                    inplace=True,
                    axis="columns",
                    mapper={
                        "observationId": "obsHistID",
                        "descDitheredRA": "ditheredRA",
                        "descDitheredDec": "ditheredDec",
                    },
                )
                dither_table.set_index("obsHistID", inplace=True)
            else:
                dither_table = pd.read_csv(simulation.dither_path, index_col=0)

            self.cadence = oss.OpSimOutput.fromOpSimDB(
                path,
                subset=flag,
                opsimversion=vers,
                add_dithers=add_dith,
                dithercolumns=dither_table,
                filterNull=null_option,
            ).summary
        else:
            self.cadence = oss.OpSimOutput.fromOpSimDB(
                path,
                subset=flag,
                opsimversion=vers,
                add_dithers=add_dith,
                filterNull=null_option,
            ).summary

    def get_throughputs(self, simulation):
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
        path = simulation.throughputs_path
        self.throughputs = {}
        tp_filelist = os.listdir(path)
        for band_file_name in tp_filelist:
            band = os.path.splitext(band_file_name)[0]
            band = band.replace("{}".format(simulation.instrument), "")
            self.throughputs[band] = {}
            throughput_file = path + "/" + band_file_name
            band_wave = list()
            band_throughput = list()
            # Get the particular band throughput
            band_file = open(throughput_file, "r")
            for line in band_file:
                # Strip header comments
                if line.strip().startswith("#"):
                    # check to see what units the wavelengths are in
                    nano_match = re.search("nm|nanometer|Nanometer|Nm", line)
                    ang_match = re.search(u"\u212B|Ang|Angstrom|angstrom", line)
                    mic_match = re.search(u"\u03BC|um|micrometer|micron|", line)
                    if nano_match:
                        conversion = 10.0  # conversion for nanometers to Angs
                    elif ang_match:
                        conversion = 1.0
                    elif mic_match:
                        conversion = 1e4
                    continue
                else:
                    # Strip per line comments
                    comment_match = re.match(r"^([^#]*)#(.*)$", line)
                    if comment_match:  # The line contains a hash / comment
                        line = comment_match.group(1)
                    line = line.strip()
                    split_fields = re.split(r'[ ,|;"]+', line)
                    band_wave.append(conversion * float(split_fields[0]))
                    band_throughput.append(float(split_fields[1]))
            band_file.close()
            band_wave = np.asarray(band_wave)
            band_throughput = np.asarray(band_throughput)
            self.throughputs[band]["wavelengths"] = band_wave
            self.throughputs[band]["throughput"] = band_throughput

    def get_reference_flux(self, path, magsys="ab"):
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
        ref_filepath = os.path.join(path, "{0}.dat".format(magsys))
        ref_file = open(ref_filepath, "r")
        for line in ref_file:
            # Strip header comments
            if line.strip().startswith("#"):
                continue
            else:
                # Strip per line comments
                comment_match = re.match(r"^([^#]*)#(.*)$", line)
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
        ref_flux_per_wave_for_model = np.empty([len(phase_for_ref), len(ref_wave)])

        # Fake multi-phase observations to exploit the sncosmo model object
        for i, phase in enumerate(phase_for_ref):
            ref_flux_per_wave_for_model[i, :] = ref_flux_per_wave
        # Put throughput and reference on the same wavelength grid
        # Exploit sncosmo functionality to do this
        ref_source = sncosmo.TimeSeriesSource(
            phase_for_ref, ref_wave, ref_flux_per_wave_for_model
        )
        ref_model = sncosmo.Model(source=ref_source)
        self.reference_flux_response = {}
        for band in self.throughputs.keys():
            self.reference_flux_response[band] = bandflux(
                band_throughput=self.throughputs[band], ref_model=ref_model
            )

    def get_survey_params(self, simulation):
        """
        Method to obtain summary features of the given cadence.
        """
        self.col_dec = simulation.dec_col
        self.col_ra = simulation.ra_col
        min_db = self.cadence.min()
        max_db = self.cadence.max()
        if simulation.same_dist is True:
            self.min_ra = 0.0
            self.max_ra = 2.0 * np.pi
            if min_db[self.col_dec] < simulation.min_dec:
                if min_db[self.col_dec] < np.deg2rad(-90.0) + self.FOV_radius:
                    self.min_dec = min_db[self.col_dec]
                else:
                    self.min_dec = min_db[self.col_dec] - self.FOV_radius
            else:
                if simulation.min_dec < np.deg2rad(-90.0) + self.FOV_radius:
                    self.min_dec = simulation.min_dec
                else:
                    self.min_dec = simulation.min_dec - self.FOV_radius
            if max_db[self.col_dec] > simulation.max_dec:
                if max_db[self.col_dec] > np.deg2rad(90.0) - self.FOV_radius:
                    self.max_dec = max_db[self.col_dec]
                else:
                    self.max_dec = max_db[self.col_dec] + self.FOV_radius
            else:
                if simulation.max_dec > np.deg2rad(90.0) - self.FOV_radius:
                    self.max_dec = simulation.max_dec
                else:
                    self.max_dec = simulation.max_dec + self.FOV_radius
        else:
            # Given a prescribed survey simulation get basic properties of the
            # simulation. Currently assume a rectangular (in RA,DEC) solid angle on
            # the sky
            # Note that the values are assumed to be in radians
            self.min_ra = min_db[self.col_ra]
            if self.min_ra > 0.0 + self.FOV_radius:
                self.min_ra = self.min_ra - self.FOV_radius
            self.max_ra = max_db[self.col_ra]
            if self.max_ra < 2.0 * np.pi - self.FOV_radius:
                self.max_ra = self.max_ra + self.FOV_radius
            self.min_dec = min_db[self.col_dec]
            if self.min_dec > -np.pi / 2.0 + self.FOV_radius:
                self.min_dec = self.min_dec - self.FOV_radius
            self.max_dec = max_db[self.col_dec] + self.FOV_radius
            if self.max_dec < np.pi / 2.0 - self.FOV_radius:
                self.max_dec = self.max_dec + self.FOV_radius

        # Survey area in degrees squared
        self.survey_area = np.rad2deg(
            np.sin(self.max_dec) - np.sin(self.min_dec)
        ) * np.rad2deg(self.max_ra - self.min_ra)
        self.min_mjd = min_db["expMJD"] - (
            (1 + simulation.z_max) * simulation.transient_duration
        )
        self.max_mjd = max_db["expMJD"]
        self.survey_time = self.max_mjd - self.min_mjd  # Survey time in days


class transient_distribution(object):
    """
    Class representing the cosmological distribution of events happening
    in the universe.
    """

    def __init__(self, survey, sim, parameter_distribution=None):
        self.rate = lambda x: sim.rate / pow(1000.0, 3)
        self.redshift_distribution(survey, sim)
        self.sky_location_dist(survey)
        self.time_dist(survey)
        self.ids_for_distribution()

    def redshift_distribution(self, survey, sim):
        # Internal funciton to generate a redshift distribution
        # Given survey parameters, a SED rate, and a cosmology draw from a
        # Poisson distribution the distribution of the objects vs. redshift.
        zlist = np.asarray(
            list(
                sncosmo.zdist(
                    zmin=sim.z_min,
                    zmax=sim.z_max,
                    time=survey.survey_time,
                    area=survey.survey_area,
                    ratefunc=self.rate,
                    cosmo=sim.cosmology,
                )
            )
        )
        self.redshift_dist = zlist.reshape((len(zlist), 1))
        self.number_simulated = len(zlist)

    def sky_location_dist(self, survey):
        # For given survey paramters distribute random points within the
        # (RA,DEC) space. Again assuming a uniform RA,Dec region
        self.ra_dist = np.random.uniform(
            low=survey.min_ra, high=survey.max_ra, size=(self.number_simulated, 1)
        )
        self.dec_dist = np.arcsin(
            np.random.uniform(
                low=np.sin(survey.min_dec),
                high=np.sin(survey.max_dec),
                size=(self.number_simulated, 1),
            )
        )

    def time_dist(self, survey):
        self.time_dist = np.random.uniform(
            low=survey.min_mjd, high=survey.max_mjd, size=(self.number_simulated, 1)
        )

    def ids_for_distribution(self):
        self.ids = np.arange(
            start=1, stop=self.number_simulated + 1, dtype=np.int
        ).reshape((self.number_simulated, 1))
