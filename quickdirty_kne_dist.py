import os
import re
import sncosmo
import numpy as np
import pandas as pd
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import opsimsummary as oss

# Import a cosmology, comment out if you want to define your own per the
# astropy cosmology class
from astropy.cosmology import Planck15 as cosmo

# Define directory for location of SEDS
seds_path = "./sedb/rosswog/winds"
surveydb_path = '/Users/cnsetzer/Documents/LSST/surveydbs/minion_1016_sqlite.db'
fields = ['fieldID', 'fieldRA', 'fieldDec', 'filter', 'expMJD', 'fiveSigmaDepth']
# Test import with plots of lc
# sncosmo.plot_lc(model=model, bands=['lsstr'])
# plt.show()

# # Test Grid with Scatter plot in the 3dimensions
# pkappa, pm_ej, pv_ej = [], [], []
# for key in key_list:
#         pkappa.append(int(seds_data[key]['kappa']))
#         pm_ej.append(seds_data[key]['m_ej'])
#         pv_ej.append(seds_data[key]['v_ej'])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pkappa, pm_ej, pv_ej)
# ax.set_xlabel('Kappa')
# ax.set_ylabel('M_ej')
# ax.set_zlabel('V_ej')
# plt.show()

# # Slice the parameters space based on Kappa
# pm_ej_k1, pm_ej_k10, pm_ej_k100 = [], [], []
# pv_ej_k1, pv_ej_k10, pv_ej_k100 = [], [], []
# for i in np.arange(len(key_list)):
#     if pkappa[i] == 1:
#         pv_ej_k1[i] = pv_ej[i]
#         pm_ej_k1[i] = pm_ej[i]
#     elif pkappa[i] == 10:
#         pv_ej_k10[i] = pv_ej[i]
#         pm_ej_k10[i] = pm_ej[i]
#     else:
#         pv_ej_k100[i] = pv_ej[i]
#         pm_ej_k100[i] = pm_ej[i]


def Get_SEDdb(path_to_seds):
    # Import SEDs into a dictionary structure
    # supported format is currently that of Rosswog's SEDS .data
    # TODO convert all files to json and read from json format
    seds_data = {}
    key_list = []
    # Get the list of SED files
    fl = os.listdir(path_to_seds)
    # Read in all  SEDS
    for filei in fl:
        filename = seds_path + filei
        fileio = open(filename, 'r')
        # Initialize dicts for sedsdb
        seds_key = filei.split(".", 1)[0]
        key_list.append(seds_key)
        seds_data[seds_key] = {}

        kappa, m_ej, v_ej = Get_SED_header_info(fileio)
    # Debug Print of seds_key to find problematic sed
        # print(seds_key)

        # Read in SEDS data with sncosmo tools
        phase, wave, flux = sncosmo.read_griddata_ascii(filename)
        source = sncosmo.TimeSeriesSource(phase, wave, flux, zero_before=True)
        model = sncosmo.Model(source=source)
        # Construct the full sed db
        seds_data[seds_key]['model'] = model
        seds_data[seds_key]['parameters']['kappa'] = kappa
        seds_data[seds_key]['parameters']['m_ej'] = m_ej
        seds_data[seds_key]['parameters']['v_ej'] = v_ej
    return seds_data


def Get_SED_header_info(fileio):
        # Read header for parameter data for model (Specific for Rosswog)
        for headline in fileio:
            if headline.strip().startswith("#"):
                if re.search("kappa =", headline):
                    kappa = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                elif re.search("m_ej = |m_w =", headline):
                    m_ej = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                elif re.search("v_ej = |v_w =", headline):
                    v_ej = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
                else:
                    continue
            else:
                fileio.close()
                break
        return kappa, m_ej, v_ej


def Get_ObsStratDB(db_path, flag):
    # Import Observing Strategy Database
    return opsimdb = oss.OpSimOutput.fromOpSimDB(surveydb_path, subset=flag)


def Gen_SED(params, SEDdb_loc=None):
    # Given a set of parameters generate an SED
    if SEDdb_loc:
        return interpolate_SED(params, SEDdb_loc)
    else:
        # Space for implementation of a parametric model for SEDs
        return generated_SED


def Interpolate_SED(params, SEDdb_loc):
    # Given a parameter space of SEDs from numerical simulations
    # interpolate a new SED that falls within this region.
    SEDdb = Get_SEDdb(SEDdb_loc)
    sub_SEDdb = Interpolation_Subspace(SEDdb, params, param_limits)
    # Find the nearest neighbors for interpolation from the parameters.
    Neighbor_SEDs = Get_SEDdb_Neighbors(sub_SEDdb)
    return iSED


def Get_SEDdb_Neighbors(sub_SEDdb):
    # For a n dimensional parameter space spanned by data
    # return the nearest neighbors
    Neighbor_SEDs = {}
    p_space = Build_Param_Space_From_SEDdb(sub_SEDdb)

    return Neighbor_SEDs


def Interpolation_Subspace(SEDdb, params):
    # Take in the full parameter space and boundary conditions and select the
    # relevant subspace for interpolation
    p_space = Build_Param_Space_From_SEDdb(SEDdb)
    for p in params:

    return sub_SEDdb


def Build_Param_Space_From_SEDdb(SEDdb):
    # Given the SEDdb build the parameter space for interpolation
    keys = SEDdb.keys()

    return p_space


def Gen_zDist_SEDs(zrange, survey_time, survey_area, cosmology, param_limits, seds_path):
    # Internal funciton to generate a redshift distribution
    Dist_SEDs = {}
    # Given survey parameters, a SED rate, and a cosmology draw from a Poisson
    # distribution the distribution of the objects vs. redshift.
    SED_zlist = list(sncosmo.zdist(zmin=zrange[0], zmax=zrange[1],
                                   time=survey_time, area=survey_area,
                                   rate=SED_Rate(), cosmo=cosmology))
    N_SEDs = len(SED_zlist)
    for i in np.arange(N_SEDs):
        SED_params = Draw_SED_Params(param_limits)
        Dist_SEDs[i] = Gen_SED(SED_params, seds_path)
        # Place the SED at the redshift from the redshift distribution calc.
        Dist_SEDs[i]['model'].set(z=SED_zlist[i],
                                  amplitude=1./pow(cosmology.luminosity_distance(SED_zlist[i]).value, 2))
    return Dist_SEDs


def SED_Rate():
    # Intermediate wrapper for generic SED rates, currently set to return KNe
    # Rates
    return KNe_Rate()


def KNe_Rate():
    # Rate definition for Kilonovae, defined for future exploration
    rate = 500/pow(1000, 3)  # per yer per Mpc^3
    return rate


def Draw_SED_Params(param_limits):
    # Wrapper for generic SED parameters, note this is dependent on Model
    # Currently returns for KNe
    return Draw_KNe_Params(param_limits)


def Draw_KNe_Params(param_limits):
    # Sample the parameter priors
    p = {}
    for key in param_limits.keys():
        if key = 'kappa':
            p[key] = param_limits[key][0]+int(param_limits[key][1] -
            param_limits[key][0])*int(np.random.binomial(1, 0.5, size=None))
        elif key = 'm_ej' | key = 'v_ej':
            p['m_ej'] = np.random.uniform(low=param_limits['m_ej'][0],
                                          high=param_limits['m_ej'][1])
            p['v_ej'] = np.random.uniform(low=param_limits['v_ej'][0],
                                     high=param_limits['v_ej'][1]*pow(p['m_ej']
                                          / param_limits['m_ej'][0],
                                          np.log10(0.25/param_limits['v_ej'][1])
                                          / np.log10(param_limits['m_ej'][1]
                                          / param_limits['m_ej'][0])))
    return p


def uniform_pdf(a, b, x):
    # Log probability for the uniform distribution
    if a <= x <= b:
        return (1/b-a)
    else:
        return 0


def SEDs_to_Lightcurves(SEDs, instrument):
    # Go from SED to multi-band lightcurves for a given instrument
    for key in SEDs.key_list():

    return lightcurves


def ObsSample_Light_Curve(Lightcurve, Band):
    # Small function to take in light

    return Sampled_LC


def Get_Survey_Params(obs_db):
    # Given a prescribed survey simulation get basic properties of the
    # simulation. Currently assume a rectangular (in RA,DEC) solid angle on
    # the sky
    # Note that the values are assumed to be in radians
    min_db = obs_db.summary.min().deepcopy
    max_db = obs_db.summary.max().deepcopy
    survey_params = {}
    survey_params['min_ra'] = min_db['fieldRA']
    survey_params['max_ra'] = max_db['fieldRA']
    survey_params['min_dec'] = min_db['fieldDec']
    survey_params['max_dec'] = max_db['fieldDec']
    survey_params['survey_area'] = np.rad2deg(np.cos(min_dec) - np.cos(max_dec))*np.rad2deg(max_ra - min_ra)
    survey_params['min_mjd'] = min_db['expMJD']
    survey_params['max_mjd'] = max_db['expMJD']
    survey_params['survey_time'] = max_mjd - min_mjd  # Survey time in days
    return survey_params, obs_db


def Ra_Dec_Dist(n, survey_params):
    # For given survey paramters distribute random points within the (RA,DEC)
    # space. Again assuming a uniform RA,Dec region
    RA_dist = np.random.uniform(survey_params['min_ra'], survey_params['max_ra'], n)
    Dec_dist = np.arcsin((1 - np.sin(survey_params['min_dec'])*np.sin(survey_params['max_dec'])
               - pow(np.sin(survey_params['min_dec']), 2))/((np.sin(survey_params['max_dec']
               - np.sin(survey_params['min_dec'])))*np.random.uniform(low=0.0, high=1.0, size=n)))
    return RA_dist, Dec_dist0


def Time_Dist(n, survey_params):
    time_dist = np.random.uniform(survey_params['min_mjd'], survey_params['max_mjd'], n)
    return time_dist


def Build_Sub_SurveyDB(obsdb_path, fields, flag):
    # For a desired set fields create smaller subset database for queries
    obs_db = Get_ObsStratDB(obsdb_path, flag)
    sub_survey_db = obs_db.summary[fields].deepcopy
    return sub_survey_db
