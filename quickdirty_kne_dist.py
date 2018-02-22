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
seds_path = "./sedb/rosswog/NSNS/winds"
surveydb_path = '/Users/cnsetzer/Documents/LSST/surveydbs/minion_1016_sqlite.db'
fields = ['fieldID', 'fieldRA', 'fieldDec', 'filter', 'expMJD', 'fiveSigmaDepth']
param_priors = {'zmin': 0.0, 'zmax': 0.7, 'cosmology': cosmo,
                'kappa_min': 1, 'kappa_max': 10, 'm_ej_min': 0.01,
                'm_ej_max': 0.2, 'v_ej_min': 0.01, 'v_ej_max': 0.5}
instrument_params = {'FOV_rad': np.deg2rad(1.75), 'Mag_Sys': 'ab'}
gen_flag = 'cycle'
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


def Gen_SED(params, SEDdb_loc=None, gen_flag=None):
    # Given a set of parameters generate an SED
    if SEDdb_loc:
        if gen_flag == 'cycle':
            return Pick_Rand_dbSED(SEDdb_loc)
        else:
            return interpolate_SED(params, SEDdb_loc)
    else:
        # Space for implementation of a parametric model for SEDs
        return generated_SED


def Pick_Rand_dbSED(database_path):
    # Temporary function to pick random SED
    SEDdb = Get_SEDdb(SEDdb_loc)
    # unpacks keys object into a indexable list
    unpacked_key_list = *SEDdb.keys(),
    # Number of available seds
    N_SEDs = len(unpacked_key_list)
    Random_Draw = np.random.randint(low=0, high=N_SEDs)
    Rand_SED = SEDdb[unpacked_key_list[Random_Draw]]
    return Rand_SED


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
    print('This isn\'t working yet')
    return Neighbor_SEDs


def Interpolation_Subspace(SEDdb, params):
    # Take in the full parameter space and boundary conditions and select the
    # relevant subspace for interpolation
    p_space = Build_Param_Space_From_SEDdb(SEDdb)
    for p in params:
        print('This isn\'t working yet')
    return sub_SEDdb


def Build_Param_Space_From_SEDdb(SEDdb):
    # Given the SEDdb build the parameter space for interpolation
    keys = SEDdb.keys()
    print('This isn\'t working yet')
    return p_space


def Gen_zDist_SEDs(seds_path, survey_params, param_priors, gen_flag=None):
    # Internal funciton to generate a redshift distribution
    Dist_SEDs = {}
    # Given survey parameters, a SED rate, and a cosmology draw from a Poisson
    # distribution the distribution of the objects vs. redshift.
    SED_zlist = list(sncosmo.zdist(zmin=param_priors['zmin'], zmax=param_priors['zmax'],
                                   time=survey_params['survey_time'], area=survey_params['survey_area'],
                                   rate=SED_Rate(), cosmo=param_priors['cosmology']))
    N_SEDs = len(SED_zlist)
    for i in np.arange(N_SEDs):
        SED_params = Draw_SED_Params(param_priors)
        Dist_SEDs[i] = Gen_SED(SED_params, seds_path, gen_flag)
        # Place the SED at the redshift from the redshift distribution calc.
        Dist_SEDs[i]['model'].set(z=SED_zlist[i],
                                  amplitude=1./pow(param_priors['cosmology'].luminosity_distance(SED_zlist[i]).value, 2))
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


def SED_to_Sample_Lightcurves(SED, matched_db, instrument_params,):
    # Go from SED to multi-band lightcurves for a given instrument
    lc_samples = {}
    # Gather observations by band to build the separate lightcurves
    bands = matched_db['filter'].unique()
    for band in bands:
        mags, magnitude_errors = [], []
        lsst_band = 'lsst{}'.format(band)
        times = matched_db.query('filter == \'{}\''.format(band))['expMJD'].unique()
        for time, i in times, np.arange(len(times)):
            single_obs_db = matched_db.query('filter == \'{}\' and expMJD == {}'.format(band, time))
            mags[i] = Get_Obs_Magnitudes(SED, single_obs_db, instrument_params)
            bandflux = Get_BandFlux(SED, single_obs_db)
            fiveSigmaDepth = single_obs_db['fiveSigmaDepth'].unique()
            bandflux_error = Get_Band_Flux_Error(fiveSigmaDepth)
            magnitude_errors[i] = Get_Magnitude_Error(bandflux, bandflux_error)
        # Assemble the per band dictionary of lightcurve observations
        lc_samples[lsst_band] = {'times': times, 'magnitudes': mags, 'mag_errors': magnitude_errors}
    return lc_samples


def Get_BandFlux(SED, single_obs_db):
    # Get the bandflux for the given filter and phase
    obs_phase = single_obs_db['expMJD'] - SED['parameters']['min_MJD']
    band = 'lsst{}'.format(single_obs_db['filter'].unique())
    bandflux = SED['model'].bandflux(band, obs_phase)
    return bandflux


def Get_Obs_Magnitudes(SED, single_obs_db, instrument_params):
    # Get the observed magnitudes for the given band
    obs_phase = single_obs_db['expMJD'].unique() - SED['parameters']['min_MJD']
    band = 'lsst{}'.format(single_obs_db['filter'].unique())
    magsys = instrument_params['Mag_Sys']
    bandmag = SED['model'].bandmag(band, magsys, obs_phase)
    return bandmag


def Get_Magnitude_Error(bandflux, bandflux_error):
    # Compute the per-band magnitude errors
    magnitude_error = abs(-2.5*bandflux_error/(bandflux*np.log(10)))
    return magnitude_error


def Get_Band_Flux_Error(fiveSigmaDepth):
    # Compute the integrated bandflux error
    # Note this is trivial since the five sigma depth incorporates the
    # integrated time of the exposures.
    Flux_five_sigma = pow(10, -0.4*fiveSigmaDepth)
    bandflux_error = Flux_five_sigma/5
    return bandflux_error


def Gen_Observations(SEDs, obs_database, instrument_params):
    key_list = SEDs.keys()
    for key in key_list:
        matched_obs_db = Match_Event_to_Obs(SEDs[key], obs_database, instrument_params)
        mock_lc_obs = SED_to_Sample_Lightcurves(SEDs[key], matched_obs_db, instrument_params)
        SEDs[key]['observations'] = mock_lc_obs
    return SEDs


def Match_Event_to_Obs(SED, obs_database, instrument_params):
    # Function to find  "observations"
    survey_field_hw = instrument_params['FOV_rad']
    field_rahalf_width = survey_field_hw
    field_dechalf_width = survey_field_hw
    t_overlaps = obs_database.query('%d < expMJD < %d' % (SED['parameters']['min_MJD'], SED['parameters']['max_MJD']))
    ra_t_overlaps = t_overlaps.query('fieldRA - %d < %d < fieldRA + %d' % (field_rahalf_width, SED['parameters']['ra'], field_rahalf_width))
    # Full overlaps (note this doesn't take into account dithering)
    full_overlap_db = ra_t_overlaps.query('fieldDec - %d < %d < fieldDec + %d' % (field_dechalf_width, SED['parameters']['ra'], field_dechalf_width))
    return full_overlap_db


def Get_Survey_Params(obs_db):
    # Given a prescribed survey simulation get basic properties of the
    # simulation. Currently assume a rectangular (in RA,DEC) solid angle on
    # the sky
    # Note that the values are assumed to be in radians
    min_db = obs_db.summary.min().deepcopy
    max_db = obs_db.summary.max().deepcopy
    survey_params = {}
    survey_params['min_ra'] = min_db['fieldRA'].unique()
    survey_params['max_ra'] = max_db['fieldRA'].unique()
    survey_params['min_dec'] = min_db['fieldDec'].unique()
    survey_params['max_dec'] = max_db['fieldDec'].unique()
    survey_params['survey_area'] = np.rad2deg(np.cos(min_dec) - np.cos(max_dec))*np.rad2deg(max_ra - min_ra)
    survey_params['min_mjd'] = min_db['expMJD'].unique()
    survey_params['max_mjd'] = max_db['expMJD'].unique()
    survey_params['survey_time'] = max_mjd - min_mjd  # Survey time in days
    return survey_params, obs_db


def Ra_Dec_Dist(n, survey_params):
    # For given survey paramters distribute random points within the (RA,DEC)
    # space. Again assuming a uniform RA,Dec region
    RA_dist = np.random.uniform(survey_params['min_ra'], survey_params['max_ra'], n)
    Dec_dist = np.arcsin((1 - np.sin(survey_params['min_dec'])*np.sin(survey_params['max_dec'])
               - pow(np.sin(survey_params['min_dec']), 2))/((np.sin(survey_params['max_dec']
               - np.sin(survey_params['min_dec'])))*np.random.uniform(low=0.0, high=1.0, size=n)))
    return RA_dist, Dec_dist


def Time_Dist(n, survey_params):
    time_dist = np.random.uniform(survey_params['min_mjd'], survey_params['max_mjd'], n)
    return time_dist


def Build_Sub_SurveyDB(obsdb_path, fields, flag):
    # For a desired set fields create smaller subset database for queries
    obs_db = Get_ObsStratDB(obsdb_path, flag)
    sub_survey_db = obs_db.summary[fields].deepcopy
    return sub_survey_db

def Gen_SED_dist(SEDdb_path, survey_params, param_priors, gen_flag=None):
    # Compile the full parameter space of the generate SEDS
    # First compute the z_dist based on the survey parameters as this sets the
    # Number of SEDs
    SEDs = Gen_zDist_SEDs(SEDdb_path, survey_params, param_priors, gen_flag)
    key_list = SEDs.keys()
    N_SEDs = len(SEDs)
    RA_dist, Dec_dist = Ra_Dec_Dist(N_SEDs, survey_params)
    t_dist = Time_Dist(N_SEDs, survey_params)
    for key, i in key_list, np.arange(N_SEDs):
        #SEDs[key]['parameters']['z'] = SEDs[key]['model'].get('z')
        SEDs[key]['parameters']['ra'] = RA_dist[i]
        SEDs[key]['parameters']['dec'] = Dec_dist[i]
        SEDs[key]['parameters']['min_MJD'] = t_dist[i]
        SEDs[key]['parameters']['max_MJD'] = t_dist[i] + SEDs[key]['model'].maxtime()
    return SEDs
