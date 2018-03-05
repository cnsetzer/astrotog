import os
import re
import sncosmo
import numpy as np
from scipy.integrate import simps
import pandas as pd
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import opsimsummary as oss


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
        filename = path_to_seds + '/' + filei
        fileio = open(filename, 'r')
        # Initialize dicts for sedsdb
        seds_key = filei.split(".", 1)[0]
        key_list.append(seds_key)
        seds_data[seds_key] = {}

        kappa, m_ej, v_ej = Get_SED_header_info(fileio)
    # Debug Print of seds_key to find problematic sed
        # print(seds_key)

        # Read in SEDS data with sncosmo tools
        phase, wave, flux = deepcopy(sncosmo.read_griddata_ascii(filename))
        source = sncosmo.TimeSeriesSource(phase, wave, flux, zero_before=True)
        model = sncosmo.Model(source=source)
        # Construct the full sed db
        seds_data[seds_key]['model'] = model
        seds_data[seds_key]['parameters'] = {}
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


def Get_ObsStratDB_Summary(surveydb_path, flag):
    # Import Observing Strategy Database
    return oss.OpSimOutput.fromOpSimDB(surveydb_path, subset=flag).summary


def Gen_SED(N_SEDs, new_sed_keys, params, SEDdb_loc=None, gen_flag=None):
    # Given a set of parameters generate an SED
    if SEDdb_loc:
        if gen_flag == 'cycle':
            return Pick_Rand_dbSED(N_SEDs, new_sed_keys, SEDdb_loc)
        else:
            return interpolate_SED(N_SEDs, new_sed_keys, params, SEDdb_loc)
    else:
        # Space for implementation of a parametric model for SEDs
        return generated_SED


def Pick_Rand_dbSED(N_SEDs, new_sed_keys, SEDdb_loc):
    # Get the SED db
    SEDdb = Get_SEDdb(SEDdb_loc)
    # unpacks keys object into a indexable list
    unpacked_key_list = *SEDdb.keys(),
    # Number of available seds
    N_dbSEDs = len(unpacked_key_list)
    print('The number of SEDs in Database is {}'.format(N_dbSEDs))
    Random_Draw = np.random.randint(low=0, high=N_dbSEDs, size=N_SEDs)

    Rand_SED = {}

    for i, key in enumerate(new_sed_keys):
        Rand_SED[key] = deepcopy(SEDdb[unpacked_key_list[Random_Draw[i]]])
    return Rand_SED


def Interpolate_SED(params, SEDdb_loc):
    # Given a parameter space of SEDs from numerical simulations
    # interpolate a new SED that falls within this region.
    SEDdb = Get_SEDdb(SEDdb_loc)
    sub_SEDdb = Interpolation_Subspace(SEDdb, params, param_priors)
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
                                   ratefunc=SED_Rate(), cosmo=param_priors['cosmology']))
    N_SEDs = len(SED_zlist)
    new_keys = list()
    print('The number of mock SEDs being genereated is {}'.format(N_SEDs))
    for i in np.arange(N_SEDs):
        new_keys.append('Mock {}'.format(str(i)))

    SED_params = Draw_SED_Params(param_priors, N_SEDs)
    Dist_SEDs = Gen_SED(N_SEDs, new_keys, SED_params, seds_path, gen_flag)
    # Place the SED at the redshift from the redshift distribution calc.
    Dist_SEDs = Set_SED_Redshift(Dist_SEDs, SED_zlist, param_priors['cosmology'])
    return Dist_SEDs


def Set_SED_Redshift(SEDs, redshifts, cosmology):
    # Wrapper to set the redshift for the provided SEDs by sncosmo model method
    for i, key in enumerate(SEDs.keys()):
        redshift = redshifts[i]
        lumdist = cosmology.luminosity_distance(redshift).value * 1e6  # in pc
        # Note that it is necessary to scale the amplitude relative to the 10pc
        # (i.e. 10^2 in the following eqn.) placement of the SED currently
        SEDs[key]['model'].set(z=redshift, amplitude=pow(np.divide(10.0, lumdist), 2))
    return SEDs


def SED_Rate():
    # Intermediate wrapper for generic SED rates, currently set to return KNe
    # Rates
    return KNe_Rate()


def KNe_Rate():
    # Rate definition for Kilonovae, defined for future exploration
    rate = 500/pow(1000, 3)  # per yer per Mpc^3
    return lambda x: rate


def Draw_SED_Params(param_priors, n):
    # Wrapper for generic SED parameters, note this is dependent on Model
    # Currently returns for KNe
    return Draw_KNe_Params(param_priors, n)


def Draw_KNe_Params(param_priors, n):
    # Sample the parameter priors
    # Build empty param dicts
    p = {}
    for i in np.arange(n):
        p['{}'.format(i)] = {}
    # Set bounds
    kappa_min = param_priors['kappa_min']
    kappa_max = param_priors['kappa_max']
    kappa_diff = kappa_max-kappa_min
    mej_min = param_priors['m_ej_min']
    mej_max = param_priors['m_ej_max']
    vej_min = param_priors['v_ej_min']
    vej_max = param_priors['v_ej_max']
    for key in p.keys():
        p[key]['kappa'] = kappa_min+int(kappa_diff)*int(np.random.binomial(1, 0.5))
        p[key]['m_ej'] = np.random.uniform(low=mej_min, high=mej_max)
        p[key]['v_ej'] = np.random.uniform(low=vej_min, high=vej_max*pow(p[key]['m_ej']
                / mej_min, np.log10(0.25/vej_max) / np.log10(mej_max/mej_min)))
    return p


def SED_to_Sample_Lightcurves(SED, matched_db, instrument_params):
    # Go from SED to multi-band lightcurves for a given instrument
    lc_samples = {}
    # Gather observations by band to build the separate lightcurves
    ref_bandflux = deepcopy(instrument_params['Bandflux_References'])
    bands = deepcopy(matched_db['filter'].unique())
    for band in bands:
        mags, magnitude_errors = [], []
        lsst_band = 'lsst{}'.format(band)
        times = deepcopy(matched_db.query('filter == \'{}\''.format(band))['expMJD'].unique())
        for time in times:
            single_obs_db = deepcopy(matched_db.query('expMJD == {}'.format(time)))
            obs_phase = single_obs_db['expMJD'] - SED['parameters']['min_MJD']
            bandflux = Compute_Bandflux(band=lsst_band, throughputs=throughputs, SED=SED, phase=obs_phase)
            mags.append(Compute_Obs_Magnitudes(bandflux, ref_bandflux[lsst_band]))
            fiveSigmaDepth = deepcopy(single_obs_db['fiveSigmaDepth'].unique())
            bandflux_error = Compute_Band_Flux_Error(fiveSigmaDepth)
            magnitude_errors.append(Get_Magnitude_Error(bandflux, bandflux_error))
        # Assemble the per band dictionary of lightcurve observations
        lc_samples[lsst_band] = {'times': times, 'magnitudes': mags, 'mag_errors': magnitude_errors}
    return deepcopy(lc_samples)


def Compute_Obs_Magnitudes(bandflux, bandflux_ref):
    # Definte the flux reference based on the magnitude system reference to
    # compute the associated maggi
    maggi = bandflux/bandflux_ref
    magnitude = -2.5*np.log10(maggi)
    return magnitude


def Compute_Bandflux(band, throughputs, SED=None, phase=None, ref_model=None):
    band_wave = deepcopy(throughputs[band]['wavelengths'])
    band_throughput = deepcopy(throughputs[band]['throughput'])
    # Get 'reference' SED
    if ref_model:
        flux_per_wave = ref_model.flux(time=1.0, wave=band_wave)

    # Get SED flux
    if SED and phase:
        flux_per_wave = deepcopy(SED['model'].flux(phase, band_wave))

    # Now integrate the convolution of the SED and the bandpass
    convolution = flux_per_wave*band_throughput
    bandflux = simps(convolution, band_wave)
    return bandflux


def Get_Reference_Flux(instrument_params):
    magsys = instrument_params['Mag_Sys']
    ref_wave = list()
    ref_flux_per_wave = list()
    phase_for_ref = list()
    ref_filepath = '../throughputs/references/{0}.dat'.format(magsys)
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
            # Add a line to the phase object too to use with sncosmo
            phase_for_ref.append(1.0)
    ref_file.close()
    ref_wave = np.asarray(ref_wave)
    ref_flux_per_wave = np.asarray(ref_flux_per_wave)
    phase_for_ref = np.asarray(phase_for_ref)
    # Put throughput and reference on the same wavelength grid
    # Exploit sncosmo functionality to do this
    ref_source = sncosmo.TimeSeriesSource(phase_for_ref, ref_wave, ref_flux_per_wave, zero_before=True)
    ref_model = sncosmo.Model(source=ref_source)
    ref_bandflux = {}
    for band in instrument_params['Throughputs'].keys():
        ref_bandflux[band] = Compute_Bandflux(band=band, throughputs=instrument_params['Throughputs'], ref_model=ref_model)
    instrument_params['Bandflux_References'] = ref_bandflux
    return instrument_params


def Get_Throughputs(instrument_params):
    throughputs = {}
    instrument = instrument_params['Instrument']
    throughputs_path = '../throughputs/{0}'.format(instrument)
    tp_filelist = os.listdir(throughputs_path)
    for band_file_name in tp_filelist:
        band = band_file.strip('.dat')
        throughputs[band] = {}
        conversion = 1.0  # Conversion factor for the wavelength unit to Angstrom
        throughput_file = throughputs_path + '/' + band_file_name
        band_wave = list()
        band_throughput = list()
        # Get the particular band throughput
        band_file = open(throughput_file, 'r')
        for line in band_file:
            # Strip header comments
            if line.strip().startswith("#"):
                nano_match = re.match(r'nm|nanometer', line)
                if nano_match:
                    conversion = 10.0  # conversion for nanometers to Angstrom
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
        band_file_file.close()
        band_wave = np.asarray(band_wave)
        band_throughput = np.asarray(band_throughput)
        throughputs[band]['wavelengths'] = band_wave
        throughputs[band]['throughput'] = band_throughput
        instrument_params['Throughputs'] = throughputs
    return instrument_params


# def Get_BandFlux(SED, single_obs_db):
#     # Get the bandflux for the given filter and phase
#     obs_phase = single_obs_db['expMJD'] - SED['parameters']['min_MJD']
#     band = 'lsst{}'.format(single_obs_db['filter'].unique()[0])
#     bandflux = deepcopy(SED['model'].bandflux(band, obs_phase))
#     return np.asscalar(bandflux)
#
#
# def Get_Obs_Magnitudes_from_Model(SED, single_obs_db, instrument_params):
#     # Get the observed magnitudes for the given band
#     obs_phase = single_obs_db['expMJD'].unique() - SED['parameters']['min_MJD']
#     band = 'lsst{}'.format(single_obs_db['filter'].unique()[0])
#     magsys = instrument_params['Mag_Sys']
#     bandmag = deepcopy(SED['model'].bandmag(band, magsys, obs_phase))
#     return np.asscalar(bandmag)


def Get_Magnitude_Error(bandflux, bandflux_error):
    # Compute the per-band magnitude errors
    if bandflux > 0:
        magnitude_error = abs(-2.5*bandflux_error/(bandflux*np.log(10)))
    else:
        magnitude_error = np.asarray(100.00)
    return np.asscalar(magnitude_error)


def Compute_Band_Flux_Error(fiveSigmaDepth):
    # Compute the integrated bandflux error
    # Note this is trivial since the five sigma depth incorporates the
    # integrated time of the exposures.
    Flux_five_sigma = pow(10, -0.4*fiveSigmaDepth)
    bandflux_error = Flux_five_sigma/5
    return np.asscalar(bandflux_error)


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
    field_ra_hw = survey_field_hw
    field_dec_hw = survey_field_hw
    min_time = deepcopy(SED['parameters']['min_MJD'])
    max_time = deepcopy(SED['parameters']['max_MJD'])
    ra = deepcopy(SED['parameters']['ra'])
    dec = deepcopy(SED['parameters']['dec'])
    t_overlaps = deepcopy(obs_database.query('{0} < expMJD < {1}'.format(min_time, max_time)))
    ra_t_overlaps = deepcopy(t_overlaps.query('fieldRA - {0} < {1} < fieldRA + {0}'.format(field_ra_hw, ra)))
    # Full overlaps (note this doesn't take into account dithering)
    full_overlap_db = deepcopy(ra_t_overlaps.query('fieldDec - {0} < {1} < fieldDec + {0}'.format(field_dec_hw, dec)))
    return full_overlap_db


def Get_Survey_Params(obs_db):
    # Given a prescribed survey simulation get basic properties of the
    # simulation. Currently assume a rectangular (in RA,DEC) solid angle on
    # the sky
    # Note that the values are assumed to be in radians
    min_db = obs_db.min()
    max_db = obs_db.max()
    survey_params = {}
    survey_params['min_ra'] = min_db['fieldRA']
    survey_params['max_ra'] = max_db['fieldRA']
    survey_params['min_dec'] = min_db['fieldDec']
    survey_params['max_dec'] = max_db['fieldDec']
    min_ra, max_ra = survey_params['min_ra'], survey_params['max_ra']
    min_dec, max_dec = survey_params['min_dec'], survey_params['max_dec']
    survey_params['survey_area'] = np.rad2deg(np.sin(max_dec) - np.sin(min_dec))*np.rad2deg(max_ra - min_ra)
    survey_params['min_mjd'] = min_db['expMJD']
    survey_params['max_mjd'] = max_db['expMJD']
    min_mjd, max_mjd = survey_params['min_mjd'], survey_params['max_mjd']
    survey_params['survey_time'] = max_mjd - min_mjd  # Survey time in days
    return survey_params


def Ra_Dec_Dist(n, survey_params):
    # For given survey paramters distribute random points within the (RA,DEC)
    # space. Again assuming a uniform RA,Dec region
    RA_dist = np.random.uniform(survey_params['min_ra'], survey_params['max_ra'], n)
    Dec_dist = np.arcsin((np.random.uniform(low=0.0, high=1.0, size=n))*(np.sin(survey_params['max_dec'])
                - np.sin(survey_params['min_dec'])) + np.sin(survey_params['min_dec']))
    return RA_dist, Dec_dist


def Time_Dist(n, survey_params):
    time_dist = np.random.uniform(survey_params['min_mjd'], survey_params['max_mjd'], n)
    return time_dist


def Build_Sub_SurveyDB(obsdb_path, fields, flag):
    # For a desired set fields create smaller subset database for queries
    obs_db = Get_ObsStratDB(obsdb_path, flag)
    sub_survey_db = deepcopy(obs_db[fields])
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
    for i, key in enumerate(key_list):
        SEDs[key]['parameters']['z'] = SEDs[key]['model'].get('z')
        SEDs[key]['parameters']['ra'] = RA_dist[i]
        SEDs[key]['parameters']['dec'] = Dec_dist[i]
        SEDs[key]['parameters']['min_MJD'] = t_dist[i]
        SEDs[key]['parameters']['max_MJD'] = t_dist[i] + SEDs[key]['model'].maxtime()
    return SEDs


def Plot_Observations(Observations):
    # Function to take the specific observation data structure and plot the
    # mock observations.
    obs_key = 'observations'
    source_list = Observations.keys()
    for key in source_list:
        band_keys = Observations[key][obs_key].keys()
        n_plots = len(band_keys)
        # Max 6-color lightcurves
        f, axes = plt.subplots(n_plots)
        for i, band in enumerate(band_keys):
            times = deepcopy(Observations[key][obs_key][band]['times'])
            mags = deepcopy(Observations[key][obs_key][band]['magnitudes'])
            errs = deepcopy(Observations[key][obs_key][band]['mag_errors'])
            axes[i].errorbar(times, mags, errs)
            axes[i].legend(['{}'.format(band)])
            axes[i].set(xlabel='MJD', ylabel=r'$m_{ab}$')
        axes[0].set_title('{}'.format(key))
        # Break to only do one plot at the moment
        break
    return f


def Get_Detections(All_Observations, Selection_Cuts):
    # Given Cuts (Here this will be assumed to be SNR)
    Detections = {}
    n_detections = 0
    obs_key = 'observations'
    mocks_keys = All_Observations.keys()
    n_mocks = len(mocks_keys)
    Cut_keys = Selection_Cuts.keys()
    for mkey in mocks_keys:
        band_keys = All_Observations[mkey][obs_key].keys()
        # Initialize detection as false
        All_Observations[mkey]['Detected'] = False
        for band in band_keys:
            # Initialize as false detection
            All_Observations[mkey][obs_key][band]['Detected'] = False
            obs_in_band = deepcopy(All_Observations[mkey][obs_key][band]['times'])
            n_obs = len(obs_in_band)
            for cuts in Cut_keys:
                for i in np.arange(n_obs):
                    cut_comparison = deepcopy(All_Observations[mkey][obs_key][band][cuts][i])
                    if cut_comparison >= Selection_Cuts[cuts]['lower'] and cut_comparison <= Selection_Cuts[cuts]['upper']:
                        All_Observations[mkey][obs_key][band]['Detected'] = True
                        if All_Observations[mkey]['Detected'] is False:
                            All_Observations[mkey]['Detected'] = True
                            Detections[mkey] = deepcopy(All_Observations[mkey])
                            n_detections += 1
    efficiency = n_detections / n_mocks
    return All_Observations, Detections, n_detections, efficiency


def Assign_SNR(Observations):
    obs_key = 'observations'
    mags_key = 'magnitudes'
    err_key = 'mag_errors'
    key_list = Observations.keys()
    for key in key_list:
        band_keys = Observations[key][obs_key].keys()
        for band in band_keys:
            mags = deepcopy(Observations[key][obs_key][band][mags_key])
            errs = deepcopy(Observations[key][obs_key][band][err_key])
            Observations[key][obs_key][band]['SNR'] = np.divide(mags, errs)
    return Observations


def Get_N_z(All_Source_Obs, Detections):
    param_key = 'parameters'
    all_zs, detect_zs = [], []
    mock_all_keys = All_Source_Obs.keys()
    mock_detect_keys = Detections.keys()
    for key in mock_all_keys:
        all_zs.append(All_Source_Obs[key][param_key]['z'])
    for key in mock_detect_keys:
        detect_zs.append(Detections[key][param_key]['z'])

    all_z_hist, all_z_bins = np.histogram(a=all_zs, bins=10)
    bin_size = abs(all_z_bins[1]-all_z_bins[0])
    detect_z_hist, detect_z_bins = np.histogram(a=detect_zs, bins=all_z_bins)
    N_z_dist_fig = plt.figure()
    plt.hist(x=all_z_hist, bins=all_z_bins, histtype='step', color='red', label='All')
    plt.hist(x=detect_z_hist, bins=detect_z_bins, histtype='step', color='black', label='Detected')
    plt.xlabel('z')
    plt.ylabel(r'$N(z)$')
    plt.title('Number per {} redshift bin'.format(bin_size))
    return N_z_dist_fig
