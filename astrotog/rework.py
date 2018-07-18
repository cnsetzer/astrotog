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


class database:
    """
    Base class for cadence, observation, and source database
    """
    def __init__(self, type, path, flags):
        self.type = type
        self.path = path
        self.flags = flags
        if type == 'cadence':
            read_cadence()
        elif type == 'observation':
            build_obs_db()
        elif type == 'source':
            build_source_db()
        else:
            raise ValueError('{0} is not a supported database type.'.format(type))


class transient:
    """
    Base class for transient
    """
    def __init__(self):
        pass

class instrument:
    """
    Class to represent the instrumental features that will transform the source
    fluxes into what we observe.
    """

    def __init__(self, cadence_path,):
        get_cadence(cadence_path)
        get_throughputs()
        get_reference_flux()


    def get_cadence(self, cadence_path, flags='wfd'):
        self.cadence = oss.OpSimOutput.fromOpSimDB(cadence_path, subset=flags).summary
    def get_throughputs(self):
        self.throughputs =
    def get_reference_flux(self):
        self.reference_flux_response

class distribution(object):
    """
    Class representing the cosmological distribution of events happening
    in the universe
    """
    def __init__(self):
        pass
