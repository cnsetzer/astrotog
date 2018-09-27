import numpy as np
import pandas as pd
from scipy.integrate import simps
from copy import deepcopy
from sfdmap import SFDMap as sfd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# I personally like this style.
sns.set_style('whitegrid')
# Easy to change context from `talk`, `notebook`, `poster`, `paper`.
sns.set_context('talk')
# Initialze sfdmap for dust corrections
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
    # Note this is trivial since the five_sigma_depth incorporates the
    # integrated time of the exposures.
    Flux_five_sigma = bandflux_ref*pow(10.0, -0.4*fiveSigmaDepth)
    bandflux_error = Flux_five_sigma/5.0
    return bandflux_error


def observe(table_columns, transient, survey):
    pd_df = pd.DataFrame(columns=table_columns)
    # Begin matching of survey points to event positional overlaps
    positional_overlaps = survey.cadence.query('({3} - {0} <= {1} <= {3} + {0}) & ({4} - {0} <= {2} <= {4} + {0})'.format(survey.FOV_radius,
                                                              transient.ra,
                                                              transient.dec,
                                                              survey.col_ra,
                                                              survey.col_dec))
    t_overlaps = positional_overlaps.query('{0} <= expMJD <= {1}'.format(transient.t0, transient.tmax))

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
            band = row['filter']
            obs_phase = row['expMJD'] - transient.t0
            A_x = dust(ra=transient.ra, dec=transient.dec,
                       dust_correction=survey.dust_corrections[band])
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
            pd_df.at[index, 'transient_id'] = transient.id
            pd_df.at[index, 'mjd'] = row['expMJD']
            pd_df.at[index, 'bandfilter'] = row['filter']
            pd_df.at[index, 'instrument_magnitude'] = inst_mag
            pd_df.at[index, 'instrument_mag_one_sigma'] = inst_mag_error
            pd_df.at[index, 'instrument_flux'] = inst_flux
            pd_df.at[index, 'instrument_flux_one_sigma'] = flux_error
            pd_df.at[index, 'extincted_magnitude'] = extinct_mag
            pd_df.at[index, 'extincted_mag_one_sigma'] = extinct_mag_error
            pd_df.at[index, 'extincted_flux'] = extinct_bandflux
            pd_df.at[index, 'extincted_flux_one_sigma'] = flux_error
            pd_df.at[index, 'A_x'] = A_x
            pd_df.at[index, 'source_magnitude'] = source_mag
            pd_df.at[index, 'source_mag_one_sigma'] = source_mag_error
            pd_df.at[index, 'source_flux'] = source_bandflux
            pd_df.at[index, 'source_flux_one_sigma'] = flux_error
            pd_df.at[index, 'airmass'] = row['airmass']
            pd_df.at[index, 'five_sigma_depth'] = row['fiveSigmaDepth']
            pd_df.at[index, 'lightcurve_phase'] = obs_phase
            pd_df.at[index, 'signal_to_noise'] = inst_flux/flux_error

        if (positional_overlaps.query('expMJD < {0} - 1.0'.format(transient.t0)).dropna().empty):
            pd_df['field_previously_observed'] = False
        else:
            pd_df['field_previously_observed'] = True
        if (positional_overlaps.query('expMJD > {0} + 1.0'.format(transient.tmax)).dropna().empty):
            pd_df['field_observed_after'] = False
        else:
            pd_df['field_observed_after'] = True

    return pd_df


def write_params(table_columns, transient):
    pandas_df = pd.DataFrame(columns=table_columns)
    pandas_df.at[0, 'transient_id'] = transient.id
    if transient.num_params > 0:
        for i in range(transient.num_params):
            pandas_df.at[0, getattr(transient, 'param{0}_name'.format(i+1))] = getattr(transient,'param{0}'.format(i+1))

    pandas_df.at[0, 'true_redshift'] = transient.z
    pandas_df.at[0, 'obs_redshift'] = transient.obs_z
    pandas_df.at[0, 'explosion_time'] = transient.t0
    pandas_df.at[0, 'max_time'] = transient.tmax
    pandas_df.at[0, 'ra'] = transient.ra
    pandas_df.at[0, 'dec'] = transient.dec
    pandas_df.at[0, 'peculiar_velocity'] = transient.peculiar_vel
    return pandas_df


def mag_to_flux(mag, ref_flux):
    maggi = pow(10.0, mag/(-2.5))
    flux = maggi*ref_flux
    return flux


def dust(ra, dec, dust_correction, Rv=3.1):

    uncorr_ebv = sfdmap.ebv(ra, dec, frame='icrs', unit='radian')
    factor = dust_correction
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
    if obs_df is not None:
        obs_col = pd.DataFrame(columns=['observed'])
        alert_col = pd.DataFrame(columns=['alerted'])
        for iter, row in param_df.iterrows():
            if obs_df.query('{} == transient_id'.format(row['transient_id'])).empty is False:
                obs_col.at[iter, 'observed'] = True
            else:
                obs_col.at[iter, 'observed'] = False
            if obs_df.query('{} == transient_id & alert == True'.format(row['transient_id'])).empty is False:
                alert_col.at[iter, 'alerted'] = True
            else:
                alert_col.at[iter, 'alerted'] = False
        param_df = param_df.join(obs_col)
        param_df = param_df.join(alert_col)

    if detect_df is not None:
        detect_col = pd.DataFrame(columns=['detected'])
        for iter, row in param_df.iterrows():
            if detect_df.query('{} == transient_id'.format(row['transient_id'])).empty is False:
                detect_col.at[iter, 'detected'] = True
            else:
                detect_col.at[iter, 'detected'] = False
        param_df = param_df.join(detect_col)
    return param_df


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
                if not row['transient_id'] in passed_transients:
                    passed_transients.append(row['transient_id'])
        elif gt_lt_eq == 'lt':
            if row[filter_column] <= value:
                iters_passed.append(iter)
                if not row['transient_id'] in passed_transients:
                    passed_transients.append(row['transient_id'])
        elif gt_lt_eq == 'eq':
            if row[filter_column] == value:
                iters_passed.append(iter)
                if not row['transient_id'] in passed_transients:
                    passed_transients.append(row['transient_id'])
    if absolute is True:
        filtered_obs = pandas_df.loc[iters_passed]
    else:
        filtered_obs = pandas_df[pandas_df['transient_id'].isin(passed_transients)]

    return filtered_obs


def filter_on_count(pandas_df, num_count, filter_column=None, value=None,
                    gt_lt_eq=None, absolute=None):
    """
    This is a function which filters a Pandas DataFrame based on the number of
    counts that appear in a specified column based on the values in that column
    or simply counts of transients.
    """
    filtered_obs_on_count = pd.DataFrame(columns=pandas_df.columns)
    count_df = pd.Dataframe(columns=['counts'],
                            index=list(pandas_df['transient_id'].unique()))
    passed_transients = []
    if value and gt_lt_eq:
        value_filtered_obs = filter_on_value(pandas_df, filter_column, value,
                                             gt_lt_eq, absolute)
        for iter, row in value_filtered_obs.iterrows():
            count_df.loc['{}'.format(row['transient_id'])] += 1
            if count_df.loc['{}'.format(row['transient_id'])] >= num_count and not row['transient_id'] in passed_transients:
                passed_transients.append(transient_id)
    else:
        for iter, row in pandas_df.iterrows():
            count_df.loc['{}'.format(row['transient_id'])] += 1
            if count_df.loc['{}'.format(row['transient_id'])] >= num_count and not row['transient_id'] in passed_transients:
                passed_transients.append(transient_id)

    filtered_obs_on_count = pandas_df[pandas_df['transient_id'].isin(passed_transients)]

    return filtered_obs_on_count


def other_observations(survey, param_df, t_before, t_after, other_obs_columns):
    pd_df = pd.DataFrame(columns=other_obs_columns)
    ra = param_df.at[0, 'ra']
    dec = param_df.at[0, 'dec']
    t0 = param_df.at[0, 'explosion_time']
    tmax = param_df.at[0, 'max_time']
    positional_overlaps = survey.cadence.query('({3} - {0} <= {1} <= {3} + {0}) & ({4} - {0} <= {2} <= {4} + {0})'.format(survey.FOV_radius, ra, dec, survey.col_ra, survey.col_dec))
    # Split the computaiton of time overlaps into two queries
    t_before_overlaps = positional_overlaps.query('{0} - {1} <= expMJD & expMJD <= {0}'.format(t0, t_before))
    t_after_overlaps = positional_overlaps.query('{0} <= expMJD & expMJD <= {0} + {1}'.format(tmax, t_after))
    t_overlaps = t_before_overlaps.append(t_after_overlaps, sort=False)
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
        return pd_df
    else:
        colocated_survey = t_overlaps.loc[overlap_indices]

        for index, row in colocated_survey.iterrows():
            band = row['filter']
            fivesigma = row['fiveSigmaDepth']
            flux_error = bandflux_error(fivesigma,
                                        survey.reference_flux_response[band])
            inst_flux = flux_noise(0.0, flux_error)
            inst_mag = flux_to_mag(inst_flux,
                                   survey.reference_flux_response[band])
            inst_mag_error = magnitude_error(inst_flux, flux_error,
                                             survey.reference_flux_response[band])
            pd_df.at[index, 'transient_id'] = param_df.at[0, 'transient_id']
            pd_df.at[index, 'mjd'] = row['expMJD']
            pd_df.at[index, 'bandfilter'] = row['filter']
            pd_df.at[index, 'instrument_magnitude'] = inst_mag
            pd_df.at[index, 'instrument_mag_one_sigma'] = inst_mag_error
            pd_df.at[index, 'instrument_flux'] = inst_flux
            pd_df.at[index, 'instrument_flux_one_sigma'] = flux_error
            pd_df.at[index, 'airmass'] = row['airmass']
            pd_df.at[index, 'five_sigma_depth'] = row['fiveSigmaDepth']
            pd_df.at[index, 'signal_to_noise'] = inst_flux/flux_error
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
        snr = row['signal_to_noise']
        if snr >= 50.0:
            eff_col.at[iter, 'alert'] = True
        else:
            band = row['bandfilter']
            prob = np.asscalar(survey.detect_table.effSNR(band, snr))
            if np.random.binomial(1, prob) == 1:
                eff_col.at[iter, 'alert'] = True
            else:
                eff_col.at[iter, 'alert'] = False
    if 'alert' not in obs_df.columns:
        obs_df = obs_df.join(eff_col)
    else:
        for iter, row in obs_df.iterrows():
            if eff_col.at[iter, 'alert'] is True:
                obs_df.at[iter, 'alert'] = True
            else:
                obs_df.at[iter, 'alert'] = False
    return obs_df


def scolnic_detections(param_df, obs_df, other_obs_df):
    detected_transients = []
    for iter, row in param_df.iterrows():
        t_obs_df = obs_df.query('transient_id == {}'.format(row['transient_id']))
        t_other_obs_df = other_obs_df.query('transient_id == {}'.format(row['transient_id']))

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
        snr5_df = t_obs_df.query('signal_to_noise >= 5.0')
        # Criteria One
        if len(list(snr5_df['bandfilter'].unique())) >= 2:
            step_two_cr1 = True

        min_snr5_df = snr5_df.min()
        max_snr5_df = snr5_df.max()

        first_snr5 = min_snr5_df['mjd']
        last_snr5 = max_snr5_df['mjd']

        # Criteria Two
        if (last_snr5 - first_snr5) < 25.0:
            step_two_cr2 = True

        # Criteria Three and Four
        for iter3, row3, in t_obs_df.iterrows():
            if (first_snr5 - row3['mjd']) > 0 and (first_snr5 - row3['mjd']) <= 20.0 and step_two_cr3 is False:
                step_two_cr3 = True
            if (row3['mjd'] - last_snr5) > 0 and (row3['mjd'] - last_snr5) <= 20.0 and step_two_cr4 is False:
                step_two_cr4 = True

        for iter3, row3 in t_other_obs_df.iterrows():
            if (first_snr5 - row3['mjd']) > 0 and (first_snr5 - row3['mjd']) <= 20.0 and step_two_cr3 is False:
                step_two_cr3 = True
            if (row3['mjd'] - last_snr5) > 0 and (row3['mjd'] - last_snr5) <= 20.0 and step_two_cr4 is False:
                step_two_cr4 = True

        # record the transients which have been detected
        if step_one is True and step_two_cr1 is True and step_two_cr2 is True and step_two_cr3 is True and step_two_cr4 is True:
            detected_transients.append(row['transient_id'])

    scolnic_detections = obs_df[obs_df['transient_id'].isin(detected_transients)]

    return scolnic_detections


def scolnic_like_detections(param_df, obs_df, other_obs_df):
    detected_transients = []
    for iter, row in param_df.iterrows():
        t_obs_df = obs_df.query('transient_id == {}'.format(row['transient_id']))
        t_other_obs_df = other_obs_df.query('transient_id == {}'.format(row['transient_id']))

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
        snr5_df = alert_df
        # Criteria One
        if len(list(snr5_df['bandfilter'].unique())) >= 2:
            step_two_cr1 = True

        min_snr5_df = snr5_df.min()
        max_snr5_df = snr5_df.max()

        first_snr5 = min_snr5_df['mjd']
        last_snr5 = max_snr5_df['mjd']

        # Criteria Two
        if (last_snr5 - first_snr5) < 25.0:
            step_two_cr2 = True

        # Criteria Three and Four
        for iter3, row3, in t_obs_df.iterrows():
            if (first_snr5 - row3['mjd']) > 0 and (first_snr5 - row3['mjd']) <= 20.0 and step_two_cr3 is False:
                step_two_cr3 = True
            if (row3['mjd'] - last_snr5) > 0 and (row3['mjd'] - last_snr5) <= 20.0 and step_two_cr4 is False:
                step_two_cr4 = True

        for iter3, row3 in t_other_obs_df.iterrows():
            if (first_snr5 - row3['mjd']) > 0 and (first_snr5 - row3['mjd']) <= 20.0 and step_two_cr3 is False:
                step_two_cr3 = True
            if (row3['mjd'] - last_snr5) > 0 and (row3['mjd'] - last_snr5) <= 20.0 and step_two_cr4 is False:
                step_two_cr4 = True

        # record the transients which have been detected
        if step_one is True and step_two_cr1 is True and step_two_cr2 is True and step_two_cr3 is True and step_two_cr4 is True:
            detected_transients.append(row['transient_id'])

    scolnic_detections = obs_df[obs_df['transient_id'].isin(detected_transients)]

    return scolnic_detections


def process_nightly_coadds(obs_df, survey):
    """
    Function which coadds all observations within 12 hours of each other but
    avoids reprocessing observations more than once.

    """
    coadded_df = pd.DataFrame(columns=obs_df.columns)
    new_coadded = pd.DataFrame(columns=['coadded_night'])
    coadded_df = coadded_df.join(new_coadded)

    for transient in list(obs_df['transient_id'].unique()):
        t_obs_df = obs_df.query('transient_id == {}'.format(transient))
        for band in list(t_obs_df['bandfilter'].unique()):
            band_df = t_obs_df.query('bandfilter == \'{}\''.format(band))
            coadded_indicies = []
            for iter, row in band_df.iterrows():
                if iter in coadded_indicies:
                    continue
                else:
                    # Midnight MJD UTC + UTC Offset to midnight +/- 12 hours
                    night_start = int(row['mjd']) - survey.utc_offset/24.0 - 0.5
                    night_end = int(row['mjd']) - survey.utc_offset/24 + 0.5
                    same_night_df = band_df.query('(mjd >= {0}) & (mjd <= {1})'.format(night_start, night_end))

                    coadding_series = deepcopy(row)
                    if len(same_night_df.index) > 1:
                        coadding_series['mjd'] = same_night_df['mjd'].mean()
                        coadding_series['extincted_flux'] = same_night_df['extincted_flux'].mean()
                        coadding_series['source_flux'] = same_night_df['source_flux'].mean()
                        coadding_series['instrument_flux'] = same_night_df['instrument_flux'].mean()
                        coadding_series['instrument_flux_one_sigma'] = np.sqrt(np.sum(np.square(same_night_df['instrument_flux_one_sigma'])))/len(same_night_df.index)
                        coadding_series['source_flux_one_sigma'] = np.sqrt(np.sum(np.square(same_night_df['source_flux_one_sigma'])))/len(same_night_df.index)
                        coadding_series['extincted_flux_one_sigma'] = np.sqrt(np.sum(np.square(same_night_df['extincted_flux_one_sigma'])))/len(same_night_df.index)
                        coadding_series['airmass'] = same_night_df['airmass'].mean()
                        coadding_series['five_sigma_depth'] = same_night_df['five_sigma_depth'].mean()
                        coadding_series['lightcurve_phase'] = same_night_df['lightcurve_phase'].mean()
                        coadding_series['extincted_magnitude'] = flux_to_mag(coadding_series['extincted_flux'], survey.reference_flux_response[band])
                        coadding_series['source_magnitude'] = flux_to_mag(coadding_series['source_flux'], survey.reference_flux_response[band])
                        coadding_series['instrument_magnitude'] = flux_to_mag(coadding_series['instrument_flux'], survey.reference_flux_response[band])
                        coadding_series['instrument_mag_one_sigma'] = magnitude_error(coadding_series['instrument_flux'], coadding_series['instrument_flux_one_sigma'], survey.reference_flux_response[band])
                        coadding_series['extincted_mag_one_sigma'] = magnitude_error(coadding_series['extincted_flux'], coadding_series['extincted_flux_one_sigma'], survey.reference_flux_response[band])
                        coadding_series['source_mag_one_sigma'] = magnitude_error(coadding_series['source_flux'], coadding_series['source_flux_one_sigma'], survey.reference_flux_response[band])
                        coadding_series['signal_to_noise'] = coadding_series['instrument_flux']/coadding_series['instrument_flux_one_sigma']
                        coadding_series['coadded_night'] = True
                    else:
                        coadding_series['coadded_night'] = False
                    coadded_indicies.extend(list(same_night_df.index))
                    coadded_df = coadded_df.append(coadding_series,
                                                   ignore_index=True,
                                                   sort=False)

    return coadded_df


def redshift_distribution(param_df, simulation):
    z_min = simulation.z_min
    z_max = simulation.z_max
    bin_size = simulation.z_bin_size
    n_bins = int(round((z_max-z_min)/bin_size))
    all_zs = list(param_df['true_redshift'])
    is_detected = not param_df[param_df['detected']].empty
    if is_detected is False:
        detect_zs = []
        max_depth_detect = []
    else:
        detect_zs = list(param_df[param_df['detected']]['true_redshift'])
        max_depth_detect = list(param_df[param_df['true_redshift'] <= max(detect_zs)]['true_redshift'])

    total_eff = (len(detect_zs)/len(all_zs))*100
    max_depth_eff = (len(detect_zs)/len(max_depth_detect))*100

    print('The redshift range of all sources is {0:.4f} to {1:.4f}.'.format(min(all_zs), max(all_zs)))
    print('The redshift range of the detected sources is {0:.4f} to {1:.4f}.'.format(min(detect_zs), max(detect_zs)))
    print('There are {0} detected transients out of {1}, which is an efficiency of {2:2.2f}%  of the total simulated number.'.format(len(detect_zs), len(all_zs), total_eff))
    print('However, this is an efficiency of {0:2.2f}%  of the total that occur within the range that was detected by {1}.'.format(max_depth_eff, simulation.instrument))
    # Create the histogram
    N_z_dist_fig = plt.figure()
    plt.hist(x=all_zs, bins=n_bins, range=(z_min, z_max), histtype='step', color='red', label='All Sources', linewidth=3.0)
    plt.hist(x=detect_zs, bins=n_bins, range=(z_min, z_max), histtype='stepfilled', edgecolor='blue', color='blue', alpha=0.3, label='Detected Sources', )
    # plt.tick_params(which='both', length=10, width=1.5)
    plt.yscale('log')
    plt.legend(loc=2)
    plt.xlabel('z')
    plt.ylabel(r'$N(z)$')
    plt.title('Redshift Distribution ({0:.2f} bins, {1:2.1f}%  detected depth efficiency.)'.format(bin_size, max_depth_eff))
    return N_z_dist_fig


def determine_ddf_transients(simulation, params):
    field_rad = np.deg2rad(3.5/2.0)

    path = simulation.cadence_path
    flag = 'ddf'
    vers = simulation.version
    cadence = oss.OpSimOutput.fromOpSimDB(path, subset=flag,
                                               opsimversion=vers).summary

    if re.search('minion', survey) is None:
        field_key = 'fieldId'
    elif re.search('kraken_2042|kraken_2044|nexus_2097|mothra_2049', survey) is not None:
        field_key = 'fieldRA'
    else:
        field_key = 'fieldID'

    ddf_ra = []
    ddf_dec = []
    for field in list(cadence[field_key].unique()):
        if re.search('minion',survey_name) is None:
            ddf_ra.append(np.deg2rad(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldRA'].unique())))
            ddf_dec.append(np.deg2rad(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldDec'].unique())))
        else:
            ddf_ra.append(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldRA'].unique()))
            ddf_dec.append(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldDec'].unique()))

    ids_in_ddf = []
    num_ddf_fields = len(list(cadence[field_key].unique()))
    for i in range(num_ddf_fields):
        field_ra = eval(ddf_ra[i])
        field_dec = eval(ddf_dec[i])
        inter1 = params.query('ra - {0} <= {1} & {0} - ra <= {1}'.format(field_ra,field_rad))
        inter2 = inter1.query('dec - {0} <= {1} & {0} - dec <= {1}'.format(field_dec,field_rad))
        for index, row in inter2.iterrows():
            ra = row['ra']
            dec = row['dec']
            angdist = np.arccos(np.sin(field_dec)*np.sin(dec) +
                        np.cos(field_dec) *
                        np.cos(dec)*np.cos(ra - field_ra))
            if angdist < field_rad:
                ids_in_ddf.append(row['transient_id'])

    subset_df = pd.DataFrame(index=params.index, columns=['subset'])
    in_ddf = ids_in_ddf
    for id in list(params['transient_id']):
        if id in in_ddf:
            subset_df.at[id,'subset'] = 'ddf'
        else:
            subset_df.at[id,'subset'] = 'wfd'

    params = params.join(subset_df)

    return params
