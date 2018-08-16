import numpy as np
import pandas as pd
from scipy.integrate import simps
from copy import deepcopy
from sfdmap import SFDMap as sfd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# font = {'size': 14}
# matplotlib.rc('font', **font)
sns.set_style('whitegrid')  # I personally like this style.
# Easy to change context from `talk`, `notebook`, `poster`, `paper`.
sns.set_context('talk')
# set seed
sfdmap = sfd()


def bandflux(band_throughput, SED_model=None, phase=None, ref_model=None):
    """This is wrapper function to compute either the reference system bandflux
       or the bandflux of a source.


    Parameters
    ----------
    band_throughput : dict of two np.arrays
        This is a dictionary of the wavelengths and transmission fractions that
        characterize the throughput for the given bandfilter.
    SED_model : :obj:, optional
        The sncosmo model object that represents the spectral energy
        distribution of the simulated source.
    phase : float, optional
        This the phase of transient lifetime that is being calculated.
    ref_model : :obj:, optional
        The source model used to calculated the reference bandflux.

    Returns
    -------
    bandflux : float
        The computed bandflux that is measured by the instrument for a source.

    """
    band_wave = band_throughput['wavelengths']
    band_tp = band_throughput['throughput']
    # Get 'reference' SED
    if ref_model:
        flux_per_wave = ref_model.flux(time=2.0, wave=band_wave)

    # Get SED flux
    if SED_model is not None and phase is not None:
        # For very low i.e. zero registered flux, sncosmo sometimes returns
        # negative values so use absolute value to work around this issue.
        flux_per_wave = abs(SED_model.flux(phase, band_wave))

    # Now integrate the combination of the SED flux and the bandpass
    response_flux = flux_per_wave*band_tp
    bandflux = simps(response_flux, band_wave)
    return np.asscalar(bandflux)


def flux_to_mag(bandflux, bandflux_ref):
    # Definte the flux reference based on the magnitude system reference to
    # compute the associated maggi
    maggi = bandflux/bandflux_ref
    magnitude = -2.5*np.log10(maggi)
    return np.asscalar(magnitude)


def flux_noise(bandflux, bandflux_error):
    # Add gaussian noise to the true bandflux
    new_bandflux = np.random.normal(loc=bandflux, scale=bandflux_error)
    if new_bandflux < 0.0:
        new_bandflux = 1.0e-30
    return new_bandflux


def magnitude_error(bandflux, bandflux_error, bandflux_ref):
    # Compute the per-band magnitude errors
    magnitude_error = abs(-2.5/(bandflux*np.log(10)))*bandflux_error
    return magnitude_error


def bandflux_error(fiveSigmaDepth, bandflux_ref):
    # Compute the integrated bandflux error
    # Note this is trivial since the five sigma depth incorporates the
    # integrated time of the exposures.
    Flux_five_sigma = bandflux_ref*pow(10, -0.4*fiveSigmaDepth)
    bandflux_error = Flux_five_sigma/5
    return bandflux_error


def observe(table_columns, transient, survey):
    pd_df = pd.DataFrame(columns=table_columns)
    # Begin matching of survey points to event positional overlaps
    positional_overlaps = deepcopy(survey.cadence.query('({3} - {0} <= {1} <= {3} + {0}) & ({4} - {0} <= {2} <= {4} + {0})'.format(survey.FOV_radius,
                                                              transient.ra,
                                                              transient.dec,
                                                              survey.col_ra,
                                                              survey.col_dec)))
    t_overlaps = deepcopy(positional_overlaps.query('{0} <= expMJD <= {1}'.format(transient.t0, transient.tmax)))

    overlap_indices = []
    for index, row in t_overlaps.iterrows():
        pointing_ra = row[survey.col_ra]
        pointing_dec = row[survey.col_dec]
        angdist = np.arccos(np.sin(pointing_dec)*np.sin(transient.dec) +
                            np.cos(pointing_dec) *
                            np.cos(transient.dec)*np.cos(transient.ra - pointing_ra))
        if angdist < survey.FOV_radius:
            overlap_indices.append(index)
    if not overlap_indices:
        return pd_df
    else:
        survey_overlap = t_overlaps.loc[overlap_indices]

        for index, row in survey_overlap.iterrows():
            band = 'lsst{}'.format(row['filter'])
            obs_phase = row['expMJD'] - transient.t0
            A_x = dust(ra=transient.ra, dec=transient.dec, band=band)
            source_bandflux = bandflux(survey.throughputs[band], SED_model=transient.model, phase=obs_phase)
            source_mag = flux_to_mag(source_bandflux, survey.reference_flux_response[band])
            extinct_mag = source_mag + A_x
            extinct_bandflux = mag_to_flux(extinct_mag, survey.reference_flux_response[band])
            fivesigma = row['fiveSigmaDepth']
            flux_error = bandflux_error(fivesigma, survey.reference_flux_response[band])
            source_mag_error = magnitude_error(source_bandflux, flux_error, survey.reference_flux_response[band])
            extinct_mag_error = magnitude_error(extinct_bandflux, flux_error, survey.reference_flux_response[band])
            inst_flux = flux_noise(extinct_bandflux, flux_error)
            inst_mag = flux_to_mag(inst_flux, survey.reference_flux_response[band])
            inst_mag_error = magnitude_error(inst_flux, flux_error,
                                             survey.reference_flux_response[band])
            pd_df.at[index, 'transient id'] = transient.id
            pd_df.at[index, 'mjd'] = row['expMJD']
            pd_df.at[index, 'bandfilter'] = row['filter']
            pd_df.at[index, 'instrument magnitude'] = inst_mag
            pd_df.at[index, 'instrument mag one sigma'] = inst_mag_error
            pd_df.at[index, 'instrument flux'] = inst_flux
            pd_df.at[index, 'instrument flux one sigma'] = flux_error
            pd_df.at[index, 'extincted magnitude'] = extinct_mag
            pd_df.at[index, 'extincted mag one sigma'] = extinct_mag_error
            pd_df.at[index, 'extincted flux'] = extinct_bandflux
            pd_df.at[index, 'extincted flux one sigma'] = flux_error
            pd_df.at[index, 'A_x'] = A_x
            pd_df.at[index, 'source magnitude'] = source_mag
            pd_df.at[index, 'source mag one sigma'] = source_mag_error
            pd_df.at[index, 'source flux'] = source_bandflux
            pd_df.at[index, 'source flux one sigma'] = flux_error
            pd_df.at[index, 'airmass'] = row['airmass']
            pd_df.at[index, 'five sigma depth'] = row['fiveSigmaDepth']
            pd_df.at[index, 'lightcurve phase'] = obs_phase
            pd_df.at[index, 'signal to noise'] = inst_flux/flux_error

        if (positional_overlaps.query('expMJD < {0} - 1.0'.format(transient.t0)).dropna().empty):
            pd_df['field previously observed'] = False
        else:
            pd_df['field previously observed'] = True
        if (positional_overlaps.query('expMJD > {0} + 1.0'.format(transient.tmax)).dropna().empty):
            pd_df['field observed after'] = False
        else:
            pd_df['field observed after'] = True

    return pd_df


def write_params(table_columns, transient):
    pandas_df = pd.DataFrame(columns=table_columns)
    pandas_df.at[0, 'transient id'] = transient.id
    pandas_df.at[0, 'm_ej'] = transient.m_ej
    pandas_df.at[0, 'v_ej'] = transient.v_ej
    pandas_df.at[0, 'kappa'] = transient.kappa
    pandas_df.at[0, 'true redshift'] = transient.z
    pandas_df.at[0, 'explosion time'] = transient.t0
    pandas_df.at[0, 'max time'] = transient.tmax
    pandas_df.at[0, 'ra'] = transient.ra
    pandas_df.at[0, 'dec'] = transient.dec
    pandas_df.at[0, 'peculiar velocity'] = transient.peculiar_vel
    return pandas_df


def mag_to_flux(mag, ref_flux):
    maggi = pow(10.0, mag/(-2.5))
    flux = maggi*ref_flux
    return flux


def dust(ra, dec, band, Rv=3.1):

    uncorr_ebv = sfdmap.ebv(ra, dec, frame='icrs', unit='radian')

    if band == 'lsstu':
            factor = 4.145
    elif band == 'lsstg':
            factor = 3.237
    elif band == 'lsstr':
            factor = 2.273
    elif band == 'lssti':
            factor = 1.684
    elif band == 'lsstz':
            factor = 1.323
    elif band == 'lssty':
            factor = 1.088

    A_x = factor*Rv*uncorr_ebv
    return A_x


def detect(pandas_obs_df, param_df, filter_dict):
    """
    The filter dict must be in the form:

    {filter1: {type: (value, count, both) *required*,
             num_count: Int or None,
             name: column name or None,
             value: Float or None,
             gt_lt_eq: Str ('gt','lt','eq') or None,
             absolute: Bool,
            }
     filter2: ....
    }

    """
    detected_inter = deepcopy(pandas_obs_df)

    for filter, properties in filter_dict.items():
        type = properties['type']
        count = properties['num_count']
        name = properties['name']
        value = properties['value']
        gt_lt_eq = properties['gt_lt_eq']
        absolute = properties['absolute']

        if type == 'value':
            detected_inter = filter_on_value(detected_inter, )
        elif type == 'count':
            detected_inter = filter_on_count(detected_inter, count, name,
                                             value, gt_lt_eq, absolute)
        elif type == 'both':
            detected_inter = filter_on_count(detected_inter, count, name,
                                             value, gt_lt_eq, absolute)
        else:
            print('The filter, {}, has incorrect synatx.'.format(filter))

    detected_transients = detected_inter
    return detected_transients


def param_observe_detect(param_df, obs_df=None, detect_df=None):
    if obs_df:
        obs_col = pd.DataFrame(columns=['observed'])
        for iter, row in param_df.iterrows():
            if obs_df.query('{} == transient id'.format(row['transient id'])).empty is False:
                obs_col.at[iter, 'observed'] = True
            else:
                obs_col.at[iter, 'observed'] = False
        param_df.join(obs_col)
    if detect_df:
        detect_col = pd.DataFrame(columns=['detected'])
        for iter, row in param_df.iterrows():
            if detect_df.query('{} == transient id'.format(row['transient id'])).empty is False:
                detect_col.at[iter, 'detected'] = True
            else:
                detect_col.at[iter, 'detected'] = False
        param_df.join(detect_col)
    return param_df


def filter_on_value(pandas_df, filter_column, value,  gt_lt_eq, absolute=False):
    """
    This is a function which applies a filter to the general pandas DataFrame
    of observation points to select subsamples based on filter criteria.
    """
    filtered_obs = pd.DataFrame(columns=pandas_df.columns)
    iters_passed = []
    passed_transients = []
    transient_id = None
    for iter, row in pandas_df.iterrows():
        if gt_lt_eq == 'gt':
            if row[filter_column] >= value:
                iters_passed.append(iter)
                if transient_id == row['transient id'] and skip is False:
                    passed_transients.append(transient_id)
                    skip = True
                else:
                    transient_id = row['transient id']
                    skip = False
        elif gt_lt_eq == 'lt':
            if row[filter_column] <= value:
                iters_passed.append(iter)
                if transient_id == row['transient id'] and skip is False:
                    passed_transients.append(transient_id)
                    skip = True
                else:
                    transient_id = row['transient id']
                    skip = False
        elif gt_lt_eq == 'eq':
            if row[filter_column] == value:
                iters_passed.append(iter)
                if transient_id == row['transient id'] and skip is False:
                    passed_transients.append(transient_id)
                    skip = True
                else:
                    transient_id = row['transient id']
                    skip = False

    if absolute is True:
        filtered_obs = pandas_df.loc[iters_passed]
    else:
        transient_iters = []
        for iter, row in pandas_df.iterrows():
            for id in passed_transients:
                if id == row['transient id']:
                    transient_iters.append(iter)
        filtered_obs = pandas_df.loc[transient_iters]

    return filtered_obs


def filter_on_count(pandas_df, num_count, filter_column=None, value=None,
                    gt_lt_eq=None, absolute=None):
    """
    This is a function which filters a Pandas DataFrame based on the number of
    counts that appear in a specified column based on the values in that column
    or simply counts of transients.
    """
    filtered_obs_on_count = pd.DataFrame(columns=pandas_df.columns)
    passed_transients = []
    if value and gt_lt_eq:
        value_filtered_obs = filter_on_value(pandas_df, filter_column, value,
                                             gt_lt_eq, absolute)
        transient_id = None
        for iter, row in value_filtered_obs.iterrows():
            if transient_id == row['transient id']:
                count += 1
                if count >= num_count and skip is False:
                    passed_transients.append(transient_id)
                    skip = True
            else:
                transient_id = row['transient id']
                skip = False
                count = 1
    else:
        transient_id = None
        count = 0
        for iter, row in pandas_df.iterrows():
            if transient_id == row['transient id']:
                count += 1
                if count >= num_count and skip is False:
                    passed_transients.append(transient_id)
                    skip = True
            else:
                transient_id = row['transient id']
                skip = False
                count = 1

    transient_iters = []
    for iter, row in pandas_df.iterrows():
        for id in passed_transients:
            if id == row['transient id']:
                transient_iters.append(iter)
    filtered_obs_on_count = pandas_df.loc[transient_iters]

    return filtered_obs_on_count


def other_observations(survey, param_df, t_before, t_after, other_obs_columns):
    pd_df = pd.DataFrame(columns=other_obs_columns)
    ra = param_df.at[0, 'ra']
    dec = param_df.at[0, 'dec']
    t0 = param_df.at[0, 'explosion time']
    tmax = param_df.at[0, 'max time']
    positional_overlaps = deepcopy(survey.cadence.query('({3} - {0} <= {1} <= {3} + {0}) & ({4} - {0} <= {2} <= {4} + {0})'.format(survey.FOV_radius, ra, dec, survey.col_ra, survey.col_dec)))
    t_overlaps = deepcopy(positional_overlaps.query('{0} - {1} <= expMJD <= {0} | {2} <= expMJD <= {2} + {3}'.format(t0, t_before, tmax, t_after)))
    overlap_indices = []
    for index, row in t_overlaps.iterrows():
        pointing_ra = row[survey.col_ra]
        pointing_dec = row[survey.col_dec]
        angdist = np.arccos(np.sin(pointing_dec)*np.sin(dec) +
                            np.cos(pointing_dec) *
                            np.cos(dec)*np.cos(ra - pointing_ra))
        if angdist < survey.FOV_radius:
            overlap_indices.append(index)
    if not overlap_indices:
        return None
    else:
        colocated_survey = t_overlaps.loc[overlap_indices]

        for index, row in colocated_survey.iterrows():
            band = 'lsst{}'.format(row['filter'])
            fivesigma = row['fiveSigmaDepth']
            flux_error = bandflux_error(fivesigma, survey.reference_flux_response[band])
            inst_flux = flux_noise(0.0, flux_error)
            inst_mag = flux_to_mag(inst_flux, survey.reference_flux_response[band])
            inst_mag_error = magnitude_error(inst_flux, flux_error,
                                             survey.reference_flux_response[band])
            pd_df.at[index, 'transient id'] = param_df.at[0, 'transient id']
            pd_df.at[index, 'mjd'] = row['expMJD']
            pd_df.at[index, 'bandfilter'] = row['filter']
            pd_df.at[index, 'instrument magnitude'] = inst_mag
            pd_df.at[index, 'instrument mag one sigma'] = inst_mag_error
            pd_df.at[index, 'instrument flux'] = inst_flux
            pd_df.at[index, 'instrument flux one sigma'] = flux_error
            pd_df.at[index, 'airmass'] = row['airmass']
            pd_df.at[index, 'five sigma depth'] = row['fiveSigmaDepth']
            pd_df.at[index, 'signal to noise'] = inst_flux/flux_error
            if row['expMJD'] <= t0:
                pd_df.at[index, 'when'] = 'before'
            else:
                pd_df.at[index, 'when'] = 'after'
    return pd_df


def class_method_in_pool(class_instance, method_str, method_args):
    return getattr(class_instance, method_str)(*method_args)


def extend_args_list(list1, list2):
    list1.extend(list2)
    return list1

# def Get_N_z(All_Sources, Detections, param_priors, fig_num):
#     param_key = 'parameters'
#     z_min = param_priors['zmin']
#     z_max = param_priors['zmax']
#     bin_size = param_priors['z_bin_size']
#     n_bins = int(round((z_max-z_min)/bin_size))
#     all_zs, detect_zs = [], []
#     mock_all_keys = All_Sources.keys()
#     mock_detect_keys = Detections.keys()
#     for key in mock_all_keys:
#         all_zs.append(All_Sources[key][param_key]['z'])
#     for key in mock_detect_keys:
#         detect_zs.append(Detections[key][param_key]['z'])
#     print('The redshift range of all sources is {0:.4f} to {1:.4f}.'.format(min(all_zs),max(all_zs)))
#     print('The redshift range of the detected sources is {0:.4f} to {1:.4f}.'.format(min(detect_zs),max(detect_zs)))
#     # Create the histogram
#     N_z_dist_fig = plt.figure(fig_num)
#     plt.hist(x=all_zs, bins=n_bins, range=(z_min, z_max), histtype='step', color='red', label='All Sources', linewidth=3.0)
#     plt.hist(x=detect_zs, bins=n_bins, range=(z_min, z_max), histtype='stepfilled', edgecolor='blue', color='blue', alpha=0.3, label='Detected Sources', )
#     # plt.tick_params(which='both', length=10, width=1.5)
#     plt.yscale('log')
#     plt.legend(loc=2)
#     plt.xlabel('z')
#     plt.ylabel(r'$N(z)$')
#     plt.title('Number of sources per {0:.3f} redshift bin'.format(bin_size))
#     fig_num += 1
#     return N_z_dist_fig, fig_num


# def Plot_Observations(Observations, fig_num):
#     # Function to take the specific observation data structure and plot the
#     # mock observations.
#     obs_key = 'observations'
#     source_list = Observations.keys()
#
#     # Plot max lc for talk
#     max_lc_p = 0
#     for key in source_list:
#         band_keys = Observations[key][obs_key].keys()
#         for i, band in enumerate(band_keys):
#             num_p_lc = len(Observations[key][obs_key][band]['times'])
#             if num_p_lc > max_lc_p:
#                 max_lc_p = num_p_lc
#                 max_band = band
#                 max_source_key = key
#
#     f = plt.figure(fig_num)
#     axes = f.add_subplot(1, 1, 1)
#     t_0 = Observations[max_source_key]['parameters']['min_MJD']
#     times = deepcopy(Observations[max_source_key][obs_key][max_band]['times'])
#     times = times - t_0
#     mags = deepcopy(Observations[max_source_key][obs_key][max_band]['magnitudes'])
#     errs = deepcopy(Observations[max_source_key][obs_key][max_band]['mag_errors'])
#     axes.errorbar(x=times, y=mags, yerr=errs, fmt='ro')
#     axes.legend(['{0}'.format(max_band)])
#     axes.set(xlabel=r'$t - t_{0}$ MJD', ylabel=r'$m_{ab}$')
#     axes.set_ylim(bottom=np.ceil(max(mags)/10.0)*10.0, top=np.floor(min(mags)/10.0)*10.0)
#     axes.set_title('Simulated Source: {}'.format(max_source_key))
#     #
#     # for key in source_list:
#     #     band_keys = Observations[key][obs_key].keys()
#     #     n_plots = len(band_keys)
#     #     # Max 6-color lightcurves
#     #     f = plt.figure(fig_num)
#     #     for i, band in enumerate(band_keys):
#     #         axes = f.add_subplot(n_plots, 1, i+1)
#     #         times = deepcopy(Observations[key][obs_key][band]['times'])
#     #         mags = deepcopy(Observations[key][obs_key][band]['magnitudes'])
#     #         errs = deepcopy(Observations[key][obs_key][band]['mag_errors'])
#     #         axes.errorbar(x=times, y=mags, yerr=errs, fmt='kx')
#     #         axes.legend(['{}'.format(band)])
#     #         axes.set(xlabel='MJD', ylabel=r'$m_{ab}$')
#     #         axes.set_ylim(bottom=np.ceil(max(mags)/10.0)*10.0, top=np.floor(min(mags)/10.0)*10.0)
#     #
#     #     axes.set_title('{}'.format(key))
#     #     # Break to only do one plot at the moment
#     #     fig_num += 1
#     #     break
#     return f, fig_num
