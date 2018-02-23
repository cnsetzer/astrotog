import numpy as np
from math import inf
import matplotlib.pyplot as plt
import kne_dist_functions as kdf
import seaborn
# Import a cosmology, comment out if you want to define your own per the
# astropy cosmology class
from astropy.cosmology import Planck15 as cosmo

# Define directory for location of SEDS
sedsdb_path = "./sedb/rosswog/NSNS/winds"
surveydb_path = '/Users/cnsetzer/Documents/LSST/surveydbs/minion_1016_sqlite.db'
fields = ['fieldID', 'fieldRA', 'fieldDec', 'filter', 'expMJD', 'fiveSigmaDepth']
param_priors = {'zmin': 0.0, 'zmax': 0.1, 'cosmology': cosmo,
                'kappa_min': 1, 'kappa_max': 10, 'm_ej_min': 0.01,
                'm_ej_max': 0.2, 'v_ej_min': 0.01, 'v_ej_max': 0.5}
instrument_params = {'FOV_rad': np.deg2rad(1.75), 'Mag_Sys': 'ab'}
Cuts = {'SNR': {'upper': inf, 'lower': 5}}
gen_flag = 'cycle'
db_flag = 'wfd'
# Setup the basic running structure
obs_database = kdf.Get_ObsStratDB_Summary(surveydb_path, db_flag)
print(' Done reading in observation databse: {}'.format(surveydb_path))
print('\n Getting survey paramters...')
survey_params = kdf.Get_Survey_Params(obs_database)
print(' Done retreiving survey paramters.')
# Generate the all mock KNe SEDs
print('\n Generating mock KNe sources...')
SEDs = kdf.Gen_SED_dist(sedsdb_path, survey_params, param_priors, gen_flag)
print(' Done generating mock KNe sources.')
# Apply observation to all mock SEDs
print('\n Applying simulated observations to mock sources...')
All_Source_Observations = kdf.Gen_Observations(SEDs, obs_database, instrument_params)
print(' Done generating simulated observations of mock sources.')

# Plot the lightcurve results
figure = kdf.Plot_Observations(Detections)
# For the first run show only one plot
plt.show()

exit()
# Add quality of observation information, currently just SNR
All_Source_Observations = kdf.Assign_SNR(All_Source_Observations)
# Using this quality assignment and predefined selection cut criteria determine
# 'detections'
All_Source_Observations, Detections, n_detect, efficiency = kdf.Get_Detections(All_Source_Observations, Cuts)
print('\n The number of detected KNe for a {} cut of {} is {}\n This is an efficiency of {.2f}%'
      .format('SNR', Cuts['SNR']['lower'], n_detect, 100*efficiency))

# Plot histogram of detected vs genereated mock KNe
N_z_fig = kdf.Get_N_z(All_Source_Observations, Detections)

# Plot the lightcurve results
figure = kdf.Plot_Observations(Detections)
# For the first run show only one plot
plt.show()
