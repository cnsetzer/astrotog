import numpy as np
import matplotlib.pyplot as plt
import kne_dist_functions as kdf
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
gen_flag = 'cycle'
db_flag = 'wfd'
# Setup the basic running structure
obs_database = kdf.Get_ObsStratDB(surveydb_path, db_flag)
survey_params = kdf.Get_Survey_Params(obs_database)
SEDs = kdf.Gen_SED_dist(sedsdb_path, survey_params, param_priors, gen_flag)
Observations = kdf.Gen_Observations(SEDs, obs_database, instrument_params)

# Plot the results
figure = Plot_Observations(Observations)
# For the first run show only one plot
plt.show()
