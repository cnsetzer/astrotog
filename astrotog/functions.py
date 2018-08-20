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
                            np.cos(transient.dec)*np.cos(transient.ra -
                                                         pointing_ra))
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
            source_bandflux = bandflux(survey.throughputs[band],
                                       SED_model=transient.model,
                                       phase=obs_phase)
            source_mag = flux_to_mag(source_bandflux,
                                     survey.reference_flux_response[band])
            extinct_mag = source_mag + A_x
            extinct_bandflux = mag_to_flux(extinct_mag,
                                           survey.reference_flux_response[band])
            fivesigma = row['fiveSigmaDepth']
            flux_error = bandflux_error(fivesigma,
                                        survey.reference_flux_response[band])
            source_mag_error = magnitude_error(source_bandflux, flux_error,
                                               survey.reference_flux_response[band])
            extinct_mag_error = magnitude_error(extinct_bandflux, flux_error,
                                                survey.reference_flux_response[band])
            inst_flux = flux_noise(extinct_bandflux, flux_error)
            inst_mag = flux_to_mag(inst_flux,
                                   survey.reference_flux_response[band])
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


def detect(pandas_obs_df, filter_dict):
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
            detected_inter = filter_on_value(detected_inter, name, value,
                                             gt_lt_eq, absolute)
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
        alert_col = pd.DataFrame(columns=['alerted'])
        for iter, row in param_df.iterrows():
            if obs_df.query('{} == transient id'.format(row['transient id'])).empty is False:
                obs_col.at[iter, 'observed'] = True
            else:
                obs_col.at[iter, 'observed'] = False
            if obs_df.query('{} == transient id & alert == True'.format(row['transient id'])).empty is False:
                alert_col.at[iter, 'alerted'] = True
            else:
                alert_col.at[iter, 'alerted'] = False
        param_df.join(obs_col)
        param_df.join(alert_col)

    if detect_df:
        detect_col = pd.DataFrame(columns=['detected'])
        for iter, row in param_df.iterrows():
            if detect_df.query('{} == transient id'.format(row['transient id'])).empty is False:
                detect_col.at[iter, 'detected'] = True
            else:
                detect_col.at[iter, 'detected'] = False
        param_df.join(detect_col)
    return param_df


# def transient_dependent_filter(param_df, obs_df, filter_column, value1, value2, time_interval,
#                        gt_lt_eq, absolute):
#     """
#     This is a function which applies a filter to the general pandas DataFrame
#     of observation points to select subsamples based on filter criteria.
#     """
#
#     for iter, row in param_df.iterrows():
#         t_grouped_df = obs_df.query('transient id == {}'.format(row['transient id']))
#         if t_grouped_df.empty is False:
#             t_grouped_df.query('')
#
#     return pandas_df


def filter_on_value(pandas_df, filter_column, value,  gt_lt_eq,
                    absolute=False):
    """
    This is a function which applies a filter to the general pandas DataFrame
    of observation points to select subsamples based on filter criteria.
    """
    filtered_obs = pd.DataFrame(columns=pandas_df.columns)
    iters_passed = []
    passed_transients = []
    for iter, row in pandas_df.iterrows():
        if gt_lt_eq == 'gt':
            if row[filter_column] >= value:
                iters_passed.append(iter)
                if not row['transient id'] in passed_transients:
                    passed_transients.append(row['transient id'])
        elif gt_lt_eq == 'lt':
            if row[filter_column] <= value:
                iters_passed.append(iter)
                if not row['transient id'] in passed_transients:
                    passed_transients.append(row['transient id'])
        elif gt_lt_eq == 'eq':
            if row[filter_column] == value:
                iters_passed.append(iter)
                if not row['transient id'] in passed_transients:
                    passed_transients.append(row['transient id'])
    if absolute is True:
        filtered_obs = pandas_df.loc[iters_passed]
    else:
        filtered_obs = pandas_df[pandas_df['transient id'].isin(passed_transients)]

    return filtered_obs


def filter_on_count(pandas_df, num_count, filter_column=None, value=None,
                    gt_lt_eq=None, absolute=None):
    """
    This is a function which filters a Pandas DataFrame based on the number of
    counts that appear in a specified column based on the values in that column
    or simply counts of transients.
    """
    filtered_obs_on_count = pd.DataFrame(columns=pandas_df.columns)
    count_df = pd.Dataframe(columns=['counts'], index=list(pandas_df['transient id'].unique()))
    passed_transients = []
    if value and gt_lt_eq:
        value_filtered_obs = filter_on_value(pandas_df, filter_column, value,
                                             gt_lt_eq, absolute)
        for iter, row in value_filtered_obs.iterrows():
            count_df.loc['{}'.format(row['transient id'])] += 1
            if count_df.loc['{}'.format(row['transient id'])] >= num_count and not row['transient id'] in passed_transients:
                passed_transients.append(transient_id)
    else:
        for iter, row in pandas_df.iterrows():
            count_df.loc['{}'.format(row['transient id'])] += 1
            if count_df.loc['{}'.format(row['transient id'])] >= num_count and not row['transient id'] in passed_transients:
                passed_transients.append(transient_id)

    filtered_obs_on_count = pandas_df[pandas_df['transient id'].isin(passed_transients)]

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


def efficiency_process(survey, obs_df):
    """
    Wrapper function to determine if an observation generates an alert or
    "detection" in the language of Scolnic et. al 2017.
    """
    eff_col = pd.DataFrame(columns=['alert'])
    for iter, row in obs_df.iterrows():
        snr = row['signal to noise']
        band = row['bandfilter'].replace('lsst', '')
        prob = np.asscalar(survey.detect_table.effSNR(band, snr))
        if np.random.binomial(1, prob)[0] == 1:
            eff_col.at[iter, 'alert'] = True
        else:
            eff_col.at[iter, 'alert'] = False

    obs_df.join(eff_col)
    return obs_df


def scolnic_detections(param_df, obs_df, other_obs_df, survey):
    detected_transients = []
    # LSST_eff_table = eft.fromDES_EfficiencyFile(sim_inst.efficiency_table_path)
    for iter, row in param_df.iterrows():
        t_obs_df = obs_df.query('transient id == {}'.format(row['transient id']))
        t_other_obs_df = other_obs_df.query('transient id == {}'.format(row['transient id']))

        # Bool flags to pass for transient to be a Scolnic detection
        step_one = False
        step_two_cr1 = False
        step_two_cr2 = False
        step_two_cr3 = False
        step_two_cr4 = False

        # Step one, trigger simulation
        alert_df = t_obs_df.query('alert == True')
        for iter3, row3 in alert_df.iterrows():
            for iter4, row4 in alert_df.iterrows():
                if abs(row3['mjd'] - row4['mjd']) >= 0.020833333333333 and step_one is False:
                    step_one = True

        # Step two, reject SN backgrounds

        # Criteria One
        if len(list(snr5_df['bandfilter'].unique())) >= 2:
            step_two_cr1 = True
        # Criteria Two
        snr5_df = t_obs_df.query('signal to noise >= 5.0')
        for iter3, row3 in snr5_df.iterrows():
            for iter4, row4 in snr5_df.iterrows():
                if abs(row3['mjd'] - row4['mjd']) < 25.0 and step_two_cr2 is False:
                    step_two_cr2 = True
        # Criteria Three and Four
        for iter3, row3 in t_other_obs_df.iterrows():
            if (row['explosion time'] - row3['mjd']) > 0 and (row['explosion time'] - row3['mjd']) <= 20.0 and step_two_cr3 is False:
                step_two_cr3 = True
            if (row3['mjd'] - row['max time']) > 0 and (row3['mjd'] - row['max time']) <= 20.0 and step_two_cr4 is False:
                step_two_cr4 = True

        # record the transients which have been detected
        if step_one is True and step_two_cr1 is True and step_two_cr2 is True and step_two_cr3 is True and step_two_cr4 is True and not row['transient id'] in detected_transients:
            detected_transients.append(row['transient id'])

    scolnic_detections = obs_df[obs_df['transient id'].isin(detected_transients)]

    return scolnic_detections


def process_nightly_coadds(obs_df, survey):
    """
    Function which coadds all observations within 12 hours of each other but
    avoids reprocessing observations more than once.

    """
    coadded_df = pd.DataFrame(columns=obs_df.columns.append('coadded night'))

    for transient in list(obs_df['transient id'].unique()):
        t_obs_df = obs_df.query('transient id == {}'.format(transient))
        for band in list(t_obs_df['bandfilter'].unique()):
            band_df = t_obs_df.query('bandfilter == {}'.format(band))
            coadded_indicies = []
            for iter, row in band_df.iterrows():
                if iter in coadded_indicies:
                    continue
                else:
                    same_night_df = band_df.query('-0.5 < mjd - {} & mjd - {} < 0.5'.format(row['mjd']))
                    coadding_series = deepcopy(row)
                    if len(same_night_df.index) > 1:
                        coadding_series['mjd'] = same_night_df['mjd'].mean()
                        coadding_series['extincted flux'] = same_night_df['extincted flux'].mean()
                        coadding_series['source flux'] = same_night_df['source flux'].mean()
                        coadding_series['intrument flux'] = same_night_df['intrument flux'].mean()
                        coadding_series['instrument flux one sigma'] = np.sqrt(np.sum(np.square(same_night_df['instrument flux one sigma'])))/len(same_night_df.index)
                        coadding_series['source flux one sigma'] = np.sqrt(np.sum(np.square(same_night_df['source flux one sigma'])))/len(same_night_df.index)
                        coadding_series['extincted flux one sigma'] = np.sqrt(np.sum(np.square(same_night_df['extincted flux one sigma'])))/len(same_night_df.index)
                        coadding_series['airmass'] = same_night_df['airmass'].mean()
                        coadding_series['five sigma depth'] = same_night_df['five sigma depth'].mean()
                        coadding_series['lightcurve phase'] = same_night_df['lightcurve phase'].mean()
                        coadding_series['extincted magnitude'] = flux_to_mag(coadding_series['extincted flux'], survey.reference_flux_response[band])
                        coadding_series['source magnitude'] = flux_to_mag(coadding_series['source flux'], survey.reference_flux_response[band])
                        coadding_series['instrument magnitude'] = flux_to_mag(coadding_series['instrument flux'], survey.reference_flux_response[band])
                        coadding_series['instrument mag one sigma'] = magnitude_error(coadding_series['instrument flux'], coadding_series['instrument flux one sigma'], survey.reference_flux_response[band])
                        coadding_series['extincted mag one sigma'] = magnitude_error(coadding_series['extincted flux'], coadding_series['extincted flux one sigma'], survey.reference_flux_response[band])
                        coadding_series['source mag one sigma'] = magnitude_error(coadding_series['source flux'], coadding_series['source flux one sigma'], survey.reference_flux_response[band])
                        coadding_series['signal to noise'] = coadding_series['instrument flux']/coadding_series['instrument flux one sigma']
                        coadding_series['coadded night'] = True
                    else:
                        coadding_series['coadded night'] = False
                    coadded_indicies.extend(list(same_night_df.index))
                    coadded_df = coadded_df.append(coadding_series, ignore_index=True)

    return coadded_df


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
