import os
import re
import numpy as np
import sncosmo
from scipy.integrate import simps
from copy import deepcopy
import opsimsummary as oss


class observations:
    """
    This is a class for obsevations from a cadence.

    This handles




    """

    def __init__(self):

        pass

    def match(self, var):

        return self

#
# class database:
#     """
#     Base class for cadence, observation, and source database
#     """
#     def __init__(self, type, path, flags):
#         self.type = type
#         self.path = path
#         self.flags = flags
#         if type == 'cadence':
#             read_cadence()
#         elif type == 'observation':
#             build_obs_db()
#         elif type == 'source':
#             build_source_db()
#         else:
#             raise ValueError('{0} is not a supported database type.'.format(type))


class transient:
    """
    Base class for transients
    """
    def __init__(self):
        pass


class survey(object):
    """ Class for instrument parameters and the simulated cadence which together
    form the survey.

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


class distribution(object):
    """
    Class representing the cosmological distribution of events happening
    in the universe
    """
    def __init__(self):
        pass


def Compute_Bandflux(band, throughputs, SED=None, phase=None, ref_model=None):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Parameters`` section.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    If ``*args`` or ``**kwargs`` are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name : type
            description

            The description may span multiple lines. Following lines
            should be indented to match the first line of the description.
            The ": type" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : :obj:`str`, optional
        The second parameter.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    bool
        True if successful, False otherwise.

        The return type is not optional. The ``Returns`` section may span
        multiple lines and paragraphs. Following lines should be indented to
        match the first line of the description.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    """
    band_wave = throughputs[band]['wavelengths']
    band_throughput = throughputs[band]['throughput']
    # Get 'reference' SED
    if ref_model:
        flux_per_wave = ref_model.flux(time=2.0, wave=band_wave)

    # Get SED flux
    if SED is not None and phase is not None:
        # For very low i.e. zero registered flux, sncosmo sometimes returns
        # negative values so use absolute value to work around this issue.
        flux_per_wave = abs(SED['model'].flux(phase, band_wave))

    # Now integrate the combination of the SED flux and the bandpass
    response_flux = flux_per_wave*band_throughput
    bandflux = simps(response_flux, band_wave)
    return np.asscalar(bandflux)
