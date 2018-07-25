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


class survey:
    """
    Class to represent the instrumental features that will transform the source
    fluxes into what we observe.
    """

    def __init__(self, cadence_path, throughputs_path, reference_path,
                 cadence_flags):
        get_cadence(cadence_path, cadence_flags)
        get_throughputs(throughputs_path)
        get_reference_flux(reference_path)

    def get_cadence(self, path, flags):
        self.cadence = oss.OpSimOutput.fromOpSimDB(path, subset=flags).summary

    def get_throughputs(self, path):
        """
        Note
        ----

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

    def get_reference_flux(self, path):
        self.reference_flux_response


class distribution(object):
    """
    Class representing the cosmological distribution of events happening
    in the universe
    """
    def __init__(self):
        pass
