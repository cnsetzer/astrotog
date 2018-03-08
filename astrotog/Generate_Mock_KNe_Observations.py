import os
import numpy as np
from math import inf
import matplotlib.pyplot as plt
import astrotog
from astrotog import transient_obs_dist as tod
import seaborn
# Import a cosmology, comment out if you want to define your own per the
# astropy cosmology class
from astropy.cosmology import Planck15 as cosmo

# Define directory for locations of SEDS, references, throughputs
paths = {}
paths['seds'] = '/Users/cnsetzer/Documents/LSST/astrotog/sedb/rosswog/NSNS/winds'
paths['survey'] = '/Users/cnsetzer/Documents/LSST/surveydbs/minion_1016_sqlite.db'
paths['throughputs'] = '/Users/cnsetzer/Documents/LSST/astrotog/throughputs'
paths['references'] = '/Users/cnsetzer/Documents/LSST/astrotog/throughputs/references'
# Relevant fields in the survey database
fields = ['fieldID', 'fieldRA', 'fieldDec', 'filter', 'expMJD', 'fiveSigmaDepth']
# Flag for the survey database retreival to only get a subset of the whole.
db_flag = 'wfd'
# Parameter prior for generating the transient KNe distribution
param_priors = {'zmin': 0.0, 'zmax': 0.1, 'z_bin_size': 0.01, 'rate': 1000.0,
                'cosmology': cosmo, 'kappa_min': 1, 'kappa_max': 10,
                'm_ej_min': 0.01, 'm_ej_max': 0.2, 'v_ej_min': 0.01, 'v_ej_max': 0.5}
instrument_params = {'Instrument': 'lsst', 'FOV_rad': np.deg2rad(1.75), 'Mag_Sys': 'ab'}
# Different selections cuts and corresponding limits
Cuts = {'SNR': {'upper': inf, 'lower': 5, 'limit': 0.5}}
# Flag for SED generation to just cycle through SEDs in the database
gen_flag = 'cycle'

# Initialize the figure number for iterative, functional plotting
fig_num = 1
# Setup the basic running structure
print(' ')
obs_database = tod.Get_ObsStratDB_Summary(paths['survey'], db_flag)
print(' Done reading in observation databse: {}'.format(paths['survey']))
print('\n Getting survey paramters...')
survey_params = tod.Get_Survey_Params(obs_database)
print(' Done retreiving survey paramters.')
# Generate the all mock KNe SEDs
print('\n Generating mock KNe sources...')
SEDs = tod.Gen_SED_dist(paths['seds'], survey_params, param_priors, gen_flag)
print(' Done generating mock KNe sources.')
print(' Getting the LSST throughputs and computing the reference fluxes...')
instrument_params = tod.Get_Throughputs(instrument_params, paths)
instrument_params = tod.Get_Reference_Flux(instrument_params, paths)
print(' Done computing instrument parameters.')
# Apply observation to all mock SEDs
print('\n Applying simulated observations to mock sources...')
All_Source_Observations = tod.Gen_Observations(SEDs, obs_database, instrument_params)
print(' Done generating simulated observations of mock sources.')

# Add quality of observation information, currently just SNR
All_Source_Observations = tod.Assign_SNR(All_Source_Observations)
# Using this quality assignment and predefined selection cut criteria determine
# 'detections'
All_Source_Observations, Detections, n_detect, efficiency = tod.Get_Detections(All_Source_Observations, Cuts)
print('\n The number of detected KNe is {2} for a {0} cut of {1}.\n This is an efficiency of {3:.3f}%'
      .format('SNR', Cuts['SNR']['lower'], n_detect, 100*efficiency))
print('The number of mock sources is {0}, and the number of observed sources is \
        {1}'.format(len(SEDs.keys()), len(Detections.keys())))
# Plot histogram of detected vs genereated mock KNe
N_z_fig, fig_num = tod.Get_N_z(SEDs, Detections, param_priors, fig_num)

# Plot the lightcurve results
Lightcurve_fig, fig_num = tod.Plot_Observations(Detections, fig_num)
# For the first run show only one plot
plt.show()
