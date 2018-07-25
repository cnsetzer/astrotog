import os
import re
import datetime
import csv
import numpy as np
import sncosmo
import pandas as pd
from scipy.integrate import simps
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import opsimsummary as oss
import seaborn as sns
# from astrotog import macronovae_wrapper as mw
import macronovae_wrapper as mw

# font = {'size': 14}
# matplotlib.rc('font', **font)
sns.set_style('whitegrid')  # I personally like this style.
sns.set_context('talk')  # Easy to change context from `talk`, `notebook`, `poster`, `paper`. though further fine tuning is human.
# set seed
np.random.seed(12345)

def Get_SEDdb(path_to_seds):
    # Import SEDs into a dictionary structure
    # supported format is currently that of Rosswog's SEDS .data
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

        # Read in SEDS data with sncosmo tools
        phase, wave, flux = deepcopy(sncosmo.read_griddata_ascii(filename))
        source = sncosmo.TimeSeriesSource(phase, wave, flux)
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

#
# def Get_ObsStratDB_Summary(surveydb_path, flag):
#     # Import Observing Strategy Database
#     print(' Using OpSimOutput tool to get the database of simulated survey observations.')
#     return oss.OpSimOutput.fromOpSimDB(surveydb_path, subset=flag, opsimversion='lsstv3').summary


def Gen_SED(N_SEDs, new_sed_keys, param_priors, SEDdb_loc=None, gen_flag=None):
    # Given a set of parameters generate an SED
    if gen_flag == 'cycle':
        if SEDdb_loc:
            return Pick_Rand_dbSED(N_SEDs, new_sed_keys, SEDdb_loc)
    elif gen_flag == 'parametric':
        generating_parameters = Draw_KNe_Params(param_priors, new_sed_keys)
        return Semi_Analytic_KNe(generating_parameters, new_sed_keys)


def Semi_Analytic_KNe(generating_parameters, new_sed_keys):
    seds_data = {}
    for key in new_sed_keys:
        seds_data[key] = {}
        KNE_parameters = []
        KNE_parameters.append(generating_parameters[key]['t_0'])
        KNE_parameters.append(generating_parameters[key]['t_f'])
        KNE_parameters.append(generating_parameters[key]['m_ej'])
        KNE_parameters.append(generating_parameters[key]['v_ej'])
        KNE_parameters.append(generating_parameters[key]['heatingrate_exp'])
        KNE_parameters.append(generating_parameters[key]['thermalization_factor'])
        KNE_parameters.append(generating_parameters[key]['DZ_enhancement_factor'])
        KNE_parameters.append(generating_parameters[key]['kappa'])
        KNE_parameters.append(generating_parameters[key]['Initial_Temp'])
        KNE_parameters.append(False)
        KNE_parameters.append('dummy string')
        phase, wave, flux = mw.Make_Rosswog_SEDS(KNE_parameters, separated=True)
        source = sncosmo.TimeSeriesSource(phase, wave, flux)
        model = sncosmo.Model(source=source)
        seds_data[key]['model'] = model
        seds_data[key]['parameters'] = generating_parameters[key]
    return seds_data


def Pick_Rand_dbSED(N_SEDs, new_sed_keys, SEDdb_loc):
    # Get the SED db
    SEDdb = Get_SEDdb(SEDdb_loc)
    # unpacks keys object into a indexable list
    unpacked_key_list = *SEDdb.keys(),
    # Number of available seds
    N_dbSEDs = len(unpacked_key_list)
    print(' The number of SEDs in Database is {}'.format(N_dbSEDs))
    Random_Draw = np.random.randint(low=0, high=N_dbSEDs, size=N_SEDs)

    Rand_SED = {}

    for i, key in enumerate(new_sed_keys):
        Rand_SED[key] = deepcopy(SEDdb[unpacked_key_list[Random_Draw[i]]])
    return Rand_SED


def Gen_zDist_SEDs(seds_path, survey_params, param_priors, gen_flag=None):
    # Internal funciton to generate a redshift distribution
    Dist_SEDs = {}
    # Given survey parameters, a SED rate, and a cosmology draw from a Poisson
    # distribution the distribution of the objects vs. redshift.
    SED_zlist = list(sncosmo.zdist(zmin=param_priors['zmin'], zmax=param_priors['zmax'],
                                   time=survey_params['survey_time'], area=survey_params['survey_area'],
                                   ratefunc=SED_Rate(param_priors), cosmo=param_priors['cosmology']))
    N_SEDs = len(SED_zlist)
    new_keys = list()
    print(' The number of mock SEDs being genereated is {}'.format(N_SEDs))
    for i in np.arange(N_SEDs):
        new_keys.append('Mock_{}'.format(str(i)))

    Dist_SEDs = Gen_SED(N_SEDs, new_keys, param_priors, seds_path, gen_flag)
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


def SED_Rate(param_priors):
    # Intermediate wrapper for generic SED rates, currently set to return KNe
    # Rates
    rate = param_priors['rate']/pow(1000, 3)  # per yer per Mpc^3 for sncosmo
    return lambda x: rate


def Draw_SED_Params(param_priors, new_sed_keys):
    # Wrapper for generic SED parameters, note this is dependent on Model
    # Currently returns for KNe
    return Draw_KNe_Params(param_priors, new_sed_keys)


def Draw_KNe_Params(param_priors, new_sed_keys):
    # Sample the parameter priors
    # Build empty param dicts
    p = {}
    # Set bounds
    kappa_min = param_priors['kappa_min']
    kappa_max = param_priors['kappa_max']
    mej_min = param_priors['m_ej_min']
    mej_max = param_priors['m_ej_max']
    vej_min = param_priors['v_ej_min']
    vej_max = param_priors['v_ej_max']
    for key in new_sed_keys:
        p[key] = {}
        p[key]['kappa'] = np.random.uniform(low=kappa_min, high=kappa_max)
        p[key]['m_ej'] = np.random.uniform(low=mej_min, high=mej_max)
        p[key]['v_ej'] = np.random.uniform(low=vej_min, high=vej_max*pow(p[key]['m_ej']
                / mej_min, np.log10(0.25/vej_max) / np.log10(mej_max/mej_min)))
        p[key]['t_0'] = 0.00001157
        p[key]['t_f'] = 50.0
        p[key]['heatingrate_exp'] = 1.3
        p[key]['thermalization_factor'] = 0.3
        p[key]['DZ_enhancement_factor'] = 1.0
        p[key]['Initial_Temp'] = 150.0
    return p


def SED_to_Sample_Lightcurves(SED, matched_db, instrument_params):
    # Go from SED to multi-band lightcurves for a given instrument
    lc_samples = {}
    # Gather observations by band to build the separate lightcurves
    ref_bandflux = deepcopy(instrument_params['Bandflux_References'])
    throughputs = instrument_params['throughputs']
    bands = deepcopy(matched_db['filter'].unique())
    for band in bands:
        mags, magnitude_errors, fiveSigmaDepth = [], [], []
        true_bandflux, obs_bandflux, bandflux_error = [], [], []
        true_mags = []
        lsst_band = 'lsst{}'.format(band)
        times = deepcopy(matched_db.query('filter == \'{}\''.format(band))['expMJD'].unique())
        for i, time in enumerate(times):
            # Get the matched sed for a single observation
            single_obs_db = deepcopy(matched_db.query('expMJD == {}'.format(time)))
            obs_phase = np.asscalar(single_obs_db['expMJD'].values - SED['parameters']['min_MJD'])
            true_bandflux.append(Compute_Bandflux(band=lsst_band, throughputs=throughputs, SED=SED, phase=obs_phase))
            fiveSigmaDepth.append(deepcopy(single_obs_db['fiveSigmaDepth'].values))
            bandflux_error.append(Compute_Band_Flux_Error(fiveSigmaDepth[i], ref_bandflux[lsst_band]))
            obs_bandflux.append(Add_Flux_Noise(true_bandflux[i], bandflux_error[i]))
            mags.append(Compute_Obs_Magnitudes(obs_bandflux[i], ref_bandflux[lsst_band]))
            true_mags.append(Compute_Obs_Magnitudes(true_bandflux[i], ref_bandflux[lsst_band]))
            magnitude_errors.append(Get_Magnitude_Error(obs_bandflux[i], bandflux_error[i], ref_bandflux[lsst_band]))

        # Assemble the per band dictionary of lightcurve observations
        lc_samples[lsst_band] = {'times': times, 'magnitudes': mags, 'mag_errors': magnitude_errors,
                                 'band_flux': obs_bandflux, 'flux_error': bandflux_error,
                                 'five_sigma_depth': fiveSigmaDepth, 'true_bandflux': true_bandflux,
                                 'true_magnitude': true_mags}
    return deepcopy(lc_samples)


def Compute_Obs_Magnitudes(bandflux, bandflux_ref):
    # Definte the flux reference based on the magnitude system reference to
    # compute the associated maggi
    maggi = bandflux/bandflux_ref
    magnitude = -2.5*np.log10(maggi)
    return np.asscalar(magnitude)


# def Compute_Bandflux(band, throughputs, SED=None, phase=None, ref_model=None):
#     band_wave = throughputs[band]['wavelengths']
#     band_throughput = throughputs[band]['throughput']
#     # Get 'reference' SED
#     if ref_model:
#         flux_per_wave = ref_model.flux(time=2.0, wave=band_wave)
#
#     # Get SED flux
#     if SED is not None and phase is not None:
#         # For very low i.e. zero registered flux, sncosmo sometimes returns
#         # negative values so use absolute value to work around this issue.
#         flux_per_wave = abs(deepcopy(SED['model'].flux(phase, band_wave)))
#
#     # Now integrate the convolution of the SED and the bandpass
#     convolution = flux_per_wave*band_throughput
#     bandflux = simps(convolution, band_wave)
#     return np.asscalar(bandflux)

#
# def Get_Reference_Flux(instrument_params, paths):
#     magsys = instrument_params['Mag_Sys']
#     ref_wave = list()
#     ref_flux_per_wave = list()
#     # Add a line to the phase object too to use with sncosmo
#     phase_for_ref = np.arange(0.5, 5.0, step=0.5)
#     ref_filepath = os.path.join(paths['references'], '{0}.dat'.format(magsys))
#     ref_file = open(ref_filepath, 'r')
#     for line in ref_file:
#         # Strip header comments
#         if line.strip().startswith("#"):
#             continue
#         else:
#             # Strip per line comments
#             comment_match = re.match(r'^([^#]*)#(.*)$', line)
#             if comment_match:  # The line contains a hash / comment
#                 line = comment_match.group(1)
#             line = line.strip()
#             split_fields = re.split(r'[ ,|;"]+', line)
#             ref_wave.append(float(split_fields[0]))
#             ref_flux_per_wave.append(float(split_fields[1]))
#
#     ref_file.close()
#     # Convert to arrays for use with the sncosmo model
#     phase_for_ref = np.asarray(phase_for_ref)
#     ref_wave = np.asarray(ref_wave)
#     ref_flux_per_wave = np.asarray(ref_flux_per_wave)
#     ref_flux_per_wave_for_model = np.empty([len(phase_for_ref), len(ref_wave)])
#
#     # Fake multi-phase observations to exploit the sncosmo model
#     for i, phase in enumerate(phase_for_ref):
#             ref_flux_per_wave_for_model[i, :] = ref_flux_per_wave
#     # Put throughput and reference on the same wavelength grid
#     # Exploit sncosmo functionality to do this
#     ref_source = sncosmo.TimeSeriesSource(phase_for_ref, ref_wave, ref_flux_per_wave_for_model)
#     ref_model = sncosmo.Model(source=ref_source)
#     ref_bandflux = {}
#     for band in instrument_params['throughputs'].keys():
#         ref_bandflux[band] = Compute_Bandflux(band=band, throughputs=instrument_params['throughputs'], ref_model=ref_model)
#     instrument_params['Bandflux_References'] = ref_bandflux
#     return instrument_params

#
# def Get_Throughputs(instrument_params, paths):
#     throughputs = {}
#     instrument = instrument_params['Instrument']
#     throughputs_path = os.path.join(paths['throughputs'], '{0}'.format(instrument))
#     tp_filelist = os.listdir(throughputs_path)
#     for band_file_name in tp_filelist:
#         band = band_file_name.strip('.dat')
#         throughputs[band] = {}
#         conversion = 1.0  # Conversion factor for the wavelength unit to Angstrom
#         throughput_file = throughputs_path + '/' + band_file_name
#         band_wave = list()
#         band_throughput = list()
#         # Get the particular band throughput
#         band_file = open(throughput_file, 'r')
#         for line in band_file:
#             # Strip header comments
#             if line.strip().startswith("#"):
#                 nano_match = re.search(r'nm|nanometer', line)
#                 if nano_match:
#                     conversion = 10.0  # conversion for nanometers to Angstrom
#                 continue
#             else:
#                 # Strip per line comments
#                 comment_match = re.match(r'^([^#]*)#(.*)$', line)
#                 if comment_match:  # The line contains a hash / comment
#                     line = comment_match.group(1)
#                 line = line.strip()
#                 split_fields = re.split(r'[ ,|;"]+', line)
#                 band_wave.append(conversion*float(split_fields[0]))
#                 band_throughput.append(float(split_fields[1]))
#         band_file.close()
#         band_wave = np.asarray(band_wave)
#         band_throughput = np.asarray(band_throughput)
#         throughputs[band]['wavelengths'] = band_wave
#         throughputs[band]['throughput'] = band_throughput
#         instrument_params['throughputs'] = throughputs
#     return instrument_params


def Add_Flux_Noise(bandflux, bandflux_error):
    # Add gaussian noise to the true bandflux
    new_bandflux = np.random.normal(loc=bandflux, scale=bandflux_error)
    if new_bandflux < 0.0:
        new_bandflux = 1.0e-30
    return new_bandflux


def Get_Magnitude_Error(bandflux, bandflux_error, bandflux_ref):
    # Compute the per-band magnitude errors
    magnitude_error = abs(-2.5/(bandflux*np.log(10)))*bandflux_error
    return np.asscalar(magnitude_error)


def Compute_Band_Flux_Error(fiveSigmaDepth, bandflux_ref):
    # Compute the integrated bandflux error
    # Note this is trivial since the five sigma depth incorporates the
    # integrated time of the exposures.
    Flux_five_sigma = bandflux_ref*pow(10, -0.4*fiveSigmaDepth)
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
    min_time = deepcopy(SED['parameters']['min_MJD'])
    max_time = deepcopy(SED['parameters']['max_MJD'])
    ra = deepcopy(SED['parameters']['ra'])
    dec = deepcopy(SED['parameters']['dec'])
    t_overlaps = deepcopy(obs_database.query('{0} < expMJD < {1}'.format(min_time, max_time)))
    ra_t_overlaps = deepcopy(t_overlaps.query('ditheredRA - 1.25*{0} < {1} < ditheredRA + 1.25*{0}'.format(survey_field_hw, ra)))
    full_overlaps = deepcopy(ra_t_overlaps.query('ditheredDec - 1.25*{0} < {1} < ditheredDec + 1.25*{0}'.format(survey_field_hw, dec)))
    overlapped_indexes = []
    for index, row in full_overlaps.iterrows():
        pointing_ra = row['ditheredRA']
        pointing_dec = row['ditheredDec']
        angdist = np.arccos(np.sin(pointing_dec)*np.sin(dec) +
                            np.cos(pointing_dec) *
                            np.cos(dec)*np.cos(ra - pointing_ra))
        if angdist < survey_field_hw:
            overlapped_indexes.append(index)
        else:
            continue

    if not overlapped_indexes:
        full_overlap_db = pd.DataFrame(columns=list(full_overlaps.columns.values))
    else:
        full_overlap_db = t_overlaps.loc[overlapped_indexes]

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
        SEDs[key]['parameters']['z'] = deepcopy(SEDs[key]['model'].get('z'))
        SEDs[key]['parameters']['ra'] = RA_dist[i]
        SEDs[key]['parameters']['dec'] = Dec_dist[i]
        SEDs[key]['parameters']['min_MJD'] = t_dist[i]
        SEDs[key]['parameters']['max_MJD'] = t_dist[i] + SEDs[key]['model'].maxtime()
    return SEDs


def Plot_Observations(Observations, fig_num):
    # Function to take the specific observation data structure and plot the
    # mock observations.
    obs_key = 'observations'
    source_list = Observations.keys()

    # Plot max lc for talk
    max_lc_p = 0
    for key in source_list:
        band_keys = Observations[key][obs_key].keys()
        for i, band in enumerate(band_keys):
            num_p_lc = len(Observations[key][obs_key][band]['times'])
            if num_p_lc > max_lc_p:
                max_lc_p = num_p_lc
                max_band = band
                max_source_key = key

    f = plt.figure(fig_num)
    axes = f.add_subplot(1, 1, 1)
    t_0 = Observations[max_source_key]['parameters']['min_MJD']
    times = deepcopy(Observations[max_source_key][obs_key][max_band]['times'])
    times = times - t_0
    mags = deepcopy(Observations[max_source_key][obs_key][max_band]['magnitudes'])
    errs = deepcopy(Observations[max_source_key][obs_key][max_band]['mag_errors'])
    axes.errorbar(x=times, y=mags, yerr=errs, fmt='ro')
    axes.legend(['{0}'.format(max_band)])
    axes.set(xlabel=r'$t - t_{0}$ MJD', ylabel=r'$m_{ab}$')
    axes.set_ylim(bottom=np.ceil(max(mags)/10.0)*10.0, top=np.floor(min(mags)/10.0)*10.0)
    axes.set_title('Simulated Source: {}'.format(max_source_key))
    #
    # for key in source_list:
    #     band_keys = Observations[key][obs_key].keys()
    #     n_plots = len(band_keys)
    #     # Max 6-color lightcurves
    #     f = plt.figure(fig_num)
    #     for i, band in enumerate(band_keys):
    #         axes = f.add_subplot(n_plots, 1, i+1)
    #         times = deepcopy(Observations[key][obs_key][band]['times'])
    #         mags = deepcopy(Observations[key][obs_key][band]['magnitudes'])
    #         errs = deepcopy(Observations[key][obs_key][band]['mag_errors'])
    #         axes.errorbar(x=times, y=mags, yerr=errs, fmt='kx')
    #         axes.legend(['{}'.format(band)])
    #         axes.set(xlabel='MJD', ylabel=r'$m_{ab}$')
    #         axes.set_ylim(bottom=np.ceil(max(mags)/10.0)*10.0, top=np.floor(min(mags)/10.0)*10.0)
    #
    #     axes.set_title('{}'.format(key))
    #     # Break to only do one plot at the moment
    #     fig_num += 1
    #     break
    return f, fig_num


def Get_Detections(All_Observations, Selection_Cuts):
    # Given Cuts (Here this will be assumed to be SNR)
    Detections = {}
    n_detections = 0
    obs_key = 'observations'
    det_key = 'Detected'
    mocks_keys = All_Observations.keys()
    n_mocks = len(mocks_keys)
    Cut_keys = Selection_Cuts.keys()
    for mkey in mocks_keys:
        band_keys = All_Observations[mkey][obs_key].keys()
        # Initialize detection as false
        All_Observations[mkey][det_key] = False
        for band in band_keys:
            # Initialize as false detection
            All_Observations[mkey][obs_key][band][det_key] = False
            obs_in_band = deepcopy(All_Observations[mkey][obs_key][band]['times'])
            n_obs = len(obs_in_band)
            for cuts in Cut_keys:
                for i in np.arange(n_obs):
                    cut_comparison = deepcopy(All_Observations[mkey][obs_key][band][cuts][i])
                    if cut_comparison >= Selection_Cuts[cuts]['lower'] and cut_comparison <= Selection_Cuts[cuts]['upper']:
                        All_Observations[mkey][obs_key][band][det_key] = True
                        if All_Observations[mkey][det_key] is False:
                            All_Observations[mkey][det_key] = True
                            n_detections += 1
                            # Build Detections dict base, to add only the
                            # observations which pass the limits
                            Detections[mkey] = {}
                            for otherkeys in All_Observations[mkey].keys():
                                if otherkeys == obs_key:
                                    Detections[mkey][otherkeys] = {}
                                else:
                                    Detections[mkey][otherkeys] = deepcopy(All_Observations[mkey][otherkeys])
            # Limit the amount of observations to those that also pass some limiting SNR
            if All_Observations[mkey][obs_key][band][det_key] is True:
                # Initialize dictionaries for transferring detections from
                # all observations to dedicated detection object.
                Detections[mkey][obs_key][band] = {}
                dkey_items = {}
                for key in All_Observations[mkey][obs_key][band].keys():
                    if key != det_key:
                        dkey_items[key] = []
                # Cycle over observations to assemble the invidiual detections
                # above some lower limit
                for i in np.arange(n_obs):
                    for cuts in Cut_keys:
                        if All_Observations[mkey][obs_key][band][cuts][i] >= Selection_Cuts[cuts]['limit']:
                            for key in All_Observations[mkey][obs_key][band].keys():
                                if key == det_key:
                                    continue
                                else:
                                    key_value = deepcopy(All_Observations[mkey][obs_key][band][key][i])
                                    dkey_items[key].append(key_value)
                Detections[mkey][obs_key][band] = dkey_items

    efficiency = n_detections / n_mocks
    return All_Observations, Detections, n_detections, efficiency


def Assign_SNR(Observations):
    obs_key = 'observations'
    flux_key = 'band_flux'
    err_key = 'flux_error'
    key_list = Observations.keys()
    for key in key_list:
        band_keys = Observations[key][obs_key].keys()
        for band in band_keys:
            fluxes = np.vstack(deepcopy(Observations[key][obs_key][band][flux_key]))
            errs = np.vstack(deepcopy(Observations[key][obs_key][band][err_key]))
            Observations[key][obs_key][band]['SNR'] = np.divide(fluxes, errs)
    return Observations


def Get_N_z(All_Sources, Detections, param_priors, fig_num):
    param_key = 'parameters'
    z_min = param_priors['zmin']
    z_max = param_priors['zmax']
    bin_size = param_priors['z_bin_size']
    n_bins = int(round((z_max-z_min)/bin_size))
    all_zs, detect_zs = [], []
    mock_all_keys = All_Sources.keys()
    mock_detect_keys = Detections.keys()
    for key in mock_all_keys:
        all_zs.append(All_Sources[key][param_key]['z'])
    for key in mock_detect_keys:
        detect_zs.append(Detections[key][param_key]['z'])
    print('The redshift range of all sources is {0:.4f} to {1:.4f}.'.format(min(all_zs),max(all_zs)))
    print('The redshift range of the detected sources is {0:.4f} to {1:.4f}.'.format(min(detect_zs),max(detect_zs)))
    # Create the histogram
    N_z_dist_fig = plt.figure(fig_num)
    plt.hist(x=all_zs, bins=n_bins, range=(z_min, z_max), histtype='step', color='red', label='All Sources', linewidth=3.0)
    plt.hist(x=detect_zs, bins=n_bins, range=(z_min, z_max), histtype='stepfilled', edgecolor='blue', color='blue', alpha=0.3, label='Detected Sources', )
    # plt.tick_params(which='both', length=10, width=1.5)
    plt.yscale('log')
    plt.legend(loc=2)
    plt.xlabel('z')
    plt.ylabel(r'$N(z)$')
    plt.title('Number of sources per {0:.3f} redshift bin'.format(bin_size))
    fig_num += 1
    return N_z_dist_fig, fig_num


def Output_Observations(Detections):
    run_dir = 'run_' + datetime.datetime.now().strftime('%d%m%y_%H%M%S')
    save_path = '/Users/cnsetzer/Documents/LSST/astrotog_output/' + run_dir + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, name in enumerate(Detections.keys()):
        for j, band in enumerate(Detections[name]['observations'].keys()):
            df = pd.DataFrame.from_dict(data=Detections[name]['observations'][band])
            file_str = save_path + name + '_' + band + '.dat'
            df.to_csv(file_str)
    return
