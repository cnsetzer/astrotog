import numpy as np
import pandas as pd
from scipy.integrate import simps
from copy import deepcopy
from sfdmap import SFDMap as sfd

import matplotlib.pyplot as plt
import seaborn as sns
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
    positional_overlaps = deepcopy(survey.cadence.query('(ditheredRA - {0} <= {1} <= ditheredRA + {0}) & (ditheredDec - {0} <= {2} <= ditheredDec + {0})'.format(survey.FOV_radius,
                                                              transient.ra,
                                                              transient.dec)))
    t_overlaps = deepcopy(survey.cadence.query('{0} <= expMJD <= {1}'.format(transient.t0, transient.tmax)))

    overlap_indices = []
    for index, row in t_overlaps.iterrows():
        pointing_ra = row['ditheredRA']
        pointing_dec = row['ditheredDec']
        angdist = np.arccos(np.sin(pointing_dec)*np.sin(transient.dec) +
                            np.cos(pointing_dec) *
                            np.cos(transient.dec)*np.cos(transient.ra - pointing_ra))
        if angdist < survey.FOV_radius:
            overlap_indices.append(index)
    if not overlap_indices:
        pd_df.at[0, 'transient id'] = transient.id
        pd_df.at[0, 'm_ej'] = transient.m_ej
        pd_df.at[0, 'v_ej'] = transient.v_ej
        pd_df.at[0, 'kappa'] = transient.kappa
        pd_df.at[0, 'true redshift'] = transient.z
        pd_df.at[0, 'explosion time'] = transient.t0
        pd_df.at[0, 'max time'] = transient.tmax
        pd_df.at[0, 'ra'] = transient.ra
        pd_df.at[0, 'dec'] = transient.dec
        pd_df.at[index, 'peculiar velocity'] = transient.peculiar_vel
        pd_df.at[0, 'observed'] = False
        pd_df.fillna(value='-')
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
            inst_mag_error = magnitude_error(inst_flux, flux_error, survey.reference_flux_response[band])

            pd_df.at[index, 'transient id'] = transient.id
            pd_df.at[index, 'm_ej'] = transient.m_ej
            pd_df.at[index, 'v_ej'] = transient.v_ej
            pd_df.at[index, 'kappa'] = transient.kappa
            pd_df.at[index, 'true redshift'] = transient.z
            pd_df.at[index, 'explosion time'] = transient.t0
            pd_df.at[index, 'max time'] = transient.tmax
            pd_df.at[index, 'ra'] = transient.ra
            pd_df.at[index, 'dec'] = transient.dec
            pd_df.at[index, 'peculiar velocity'] = transient.peculiar_vel
            pd_df.at[index, 'observed'] = True
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
            pd_df.at[index, 'seeing'] = row['rawSeeing']
            pd_df.at[index, 'airmass'] = row['airmass']
            pd_df.at[index, 'five sigma depth'] = row['fiveSigmaDepth']
            pd_df.at[index, 'lightcurve phase'] = obs_phase
            pd_df.at[index, 'signal to noise'] = inst_flux/flux_error

        if (survey.cadence.query('expMJD < {0} - 1.0'.format(transient.t0)).dropna().empty):
            pd_df['field previously observed'] = False
        else:
            pd_df['field previously observed'] = True
        if (survey.cadence.query('expMJD > {0} + 1.0'.format(transient.tmax)).dropna().empty):
            pd_df['field observed after'] = False
        else:
            pd_df['field observed after'] = True

    return pd_df


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


def detect(pandas_df):
    """

    """
    return None


def class_method_in_pool(class_instance, method, method_args):
    return getattr(class_instance, method)(*method_args)


def extend_args_list(list1, list2):
    list1.extend(list2)
    return list1


# def Output_Observations(Detections):
#     run_dir = 'run_' + datetime.datetime.now().strftime('%d%m%y_%H%M%S')
#     save_path = '/Users/cnsetzer/Documents/LSST/astrotog_output/' + run_dir + '/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     for i, name in enumerate(Detections.keys()):
#         for j, band in enumerate(Detections[name]['observations'].keys()):
#             df = pd.DataFrame.from_dict(data=Detections[name]['observations'][band])
#             file_str = save_path + name + '_' + band + '.dat'
#             df.to_csv(file_str)
#     return


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

def compile_transients():
    """
    Wrapper function to put together a set of transient instances
    """

    return transient_dict


def run_simulation():
    """
    Wrapper function to run the full simulation set of functions like a script.
    """

    return


def run_parallel(simulation, survey, transient_distribution):
    """
    This is the wrapper function needed to run in parallel with the python
    multiprocess framework.
    """

    return
