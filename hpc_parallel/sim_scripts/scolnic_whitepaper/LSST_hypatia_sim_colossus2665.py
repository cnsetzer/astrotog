import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from astropy.cosmology import Planck15 as cosmo
from astrotog import functions as afunc
from astrotog import classes as aclasses
from astrotog import top_level_classes as atopclass
from mpi4py import MPI
import multiprocessing as mp
from itertools import repeat
import datetime
import time
from copy import copy
import pandas as pd

# Set seed for reproduceability
np.random.seed(12345)

# Execute parallel script only if used as the main script
if __name__ == "__main__":
    # MPI.Init_thread(int required=THREAD_MULTIPLE)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        t_start = time.time()
        os.system('clear')

        # ----------------------------------------------------------------------
        # Section that user can edit to tailor simulation
        # ----------------------------------------------------------------------
        batch_mp_workers = 2
        verbose = True
        batch_size = 50  # can also be set to 'all'
        dithers = False
        add_dithers = False
        t_before = 40.0
        t_after = 40.0
        z_max = 0.5  # Maximum redshift depth for simulation
        z_bin_size = 0.04  # Binning for redshift distribution histogram
        z_min = 0.0  # Given if you want to simulate shells
        rate = 1000  # Rate in events per GPC^3 per restframe time
        instrument_class_name = 'LSST'
        survey_version = 'lsstv4'
        cadence_flags = 'combined'  # Currently use default in class
        transient_model_name = 'scolnic_kilonova'
        detect_type = ['scolnic_detections']  # ['detect'], ['scolnic_detections'], or multiple
        seds_path = '/share/data1/csetzer/kilonova_seds/scolnic_decam/DECAMGemini_SED.txt'
        cadence_path = '/share/data1/csetzer/lsst_cadences/colossus_2665.db'
        throughputs_path = '/share/data1/csetzer/lsst/throughputs/lsst'
        reference_flux_path = '/share/data1/csetzer/lsst/throughputs/references'
        efficiency_table_path = '/home/csetzer/software/Cadence/LSSTmetrics/example_data/SEARCHEFF_PIPELINE_DES.DAT'
        run_dir = 'lsst_scolnic_colossus2665_' + datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        output_path = '/share/data1/csetzer/lsst_kne_sims_outputs/' + run_dir + '/'

        # Define filters for detections
        filters = {'snr': {'type': 'value',
                           'num_count': None,
                           'name': 'signal_to_noise',
                           'value': 0.001,
                           'gt_lt_eq': 'gt',
                           'absolute': True}
                   # 'snr': {'type': 'value',
                   #         'num_count': None,
                   #         'name': 'signal_to_noise',
                   #         'value': 5.0,
                   #         'gt_lt_eq': 'gt',
                   #         'absolute': False}
                   }
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        sim_inst = atopclass.simulation(cadence_path=cadence_path,
                                        throughputs_path=throughputs_path,
                                        reference_path=reference_flux_path,
                                        z_max=z_max, output_path=output_path,
                                        cadence_flags=cadence_flags,
                                        z_min=z_min, z_bin_size=z_bin_size,
                                        batch_size=batch_size, cosmology=cosmo,
                                        rate_gpc=rate, dithers=dithers,
                                        simversion=survey_version,
                                        add_dithers=add_dithers,
                                        t_before=t_before, t_after=t_after,
                                        response_path=efficiency_table_path,
                                        instrument=instrument_class_name)
        survey = getattr(atopclass, instrument_class_name)(sim_inst)
        transient_dist = aclasses.transient_distribution(survey, sim_inst)
        tran_param_dist = getattr(atopclass, transient_model_name)(parameter_dist=True,
                                                      num_samples=transient_dist.number_simulated)
        num_params_pprocess = int(np.ceil(transient_dist.number_simulated/size))
        num_transient_params = tran_param_dist.num_params
        pre_dist_params = tran_param_dist.pre_dist_params
        if verbose:
            print('\nWe are using {0} MPI workers with {1} multiprocess threads per process.'.format(size, batch_mp_workers))
            print('The number of transients is: {}'.format(transient_dist.number_simulated))
            print('The number of parameters per transient is: {}'.format(num_transient_params))
    else:
        tran_param_dist = None
        transient_model_name = None
        num_transient_params = None
        num_params_pprocess = None
        survey = None
        sim_inst = None
        batch_size = None
        batch_mp_workers = None
        verbose = None
        seds_path = None
        pre_dist_params = None

    if size > 1:
        comm.barrier()
        num_transient_params = comm.bcast(num_transient_params, root=0)
        num_params_pprocess = comm.bcast(num_params_pprocess, root=0)
        tran_param_dist = comm.bcast(tran_param_dist, root=0)
        transient_model_name = comm.bcast(transient_model_name, root=0)
        batch_size = comm.bcast(batch_size, root=0)
        batch_mp_workers = comm.bcast(batch_mp_workers, root=0)
        verbose = comm.bcast(verbose, root=0)
        seds_path = comm.bcast(seds_path, root=0)
        pre_dist_params = comm.bcast(pre_dist_params, root=0)

    total_array = None
    if rank == 0:
        sky_loc_array = np.hstack((transient_dist.ids,
                                   transient_dist.time_dist,
                                   transient_dist.ra_dist,
                                   transient_dist.dec_dist,
                                   transient_dist.redshift_dist))
        stack_list = []
        if num_transient_params > 0 and pre_dist_params is True:
            for i in range(num_transient_params):
                stack_list.append(getattr(tran_param_dist, 'param{0}'.format(i+1)))
            param_array = np.hstack(stack_list)
        else:
            param_array = None

    if size > 1:
        if rank == 0:
            if param_array is not None:
                total_array = np.hstack((sky_loc_array, param_array))
            else:
                total_array = sky_loc_array

        sky_loc_array = None
        param_array = None

        if pre_dist_params is True:
            param_buffer_size = num_transient_params
        else:
            param_buffer_size = 0

        receive_array = np.empty((num_params_pprocess,
                                 5 + param_buffer_size))
        comm.barrier()
        comm.Scatter([total_array, (5+param_buffer_size)*num_params_pprocess, MPI.DOUBLE],
                     [receive_array, (5+param_buffer_size)*num_params_pprocess, MPI.DOUBLE], root=0)
        sky_loc_array = receive_array[:, 0:5]
        if num_transient_params > 0 and pre_dist_params is True:
            param_array = receive_array[:, 5:]

    # Trim the nonsense from the process arrays
    sky_del = []
    param_del = []
    for i in range(num_params_pprocess):
        if any(abs(sky_loc_array[i]) < 1e-250):
            sky_del.append(i)
        if param_array is not None:
            if any(abs(param_array[i]) < 1e-250):
                param_del.append(i)

    sky_loc_array = np.delete(sky_loc_array, sky_del, 0)
    if param_array is not None:
        param_array = np.delete(param_array, param_del, 0)
        assert len(param_array[:]) == len(sky_loc_array[:])
    num_params_pprocess = len(sky_loc_array[:])

    if rank == 0 and verbose:
        print('The split of parameters between processes is:')
    if verbose:
        print('Process rank = {0} has {1} transients assigned to it.'.format(rank, num_params_pprocess))

    if size > 1:
        # There is definitely a problem here. Might need a virtual server...
        # ----------------------------------------------
        comm.barrier()
        survey = comm.bcast(survey, root=0)
        sim_inst = comm.bcast(sim_inst, root=0)
        # ----------------------------------------------`

    if batch_size == 'all':
        batch_size = num_params_pprocess
        num_batches = 1
    else:
        num_batches = int(np.ceil(num_params_pprocess/batch_size))

    # Create pandas table
    param_columns = ['transient_id', 'true_redshift', 'obs_redshift',
                     'explosion_time', 'max_time', 'ra', 'dec',
                     'peculiar_velocity']

    # Add model specific transient parameters to array
    if num_transient_params > 0:
        for i in range(num_transient_params):
            param_columns.append(getattr(tran_param_dist, 'param{0}_name'.format(i+1)))

    stored_param_data = pd.DataFrame(columns=param_columns)

    obs_columns = ['transient_id', 'mjd', 'bandfilter', 'instrument_magnitude',
                   'instrument_mag_one_sigma', 'instrument_flux',
                   'instrument_flux_one_sigma', 'A_x',
                   'signal_to_noise', 'source_magnitude',
                   'source_mag_one_sigma', 'source_flux',
                   'source_flux_one_sigma', 'extincted_magnitude',
                   'extincted_mag_one_sigma', 'extincted_flux',
                   'extincted_flux_one_sigma', 'airmass',
                   'five_sigma_depth', 'lightcurve_phase',
                   'field_previously_observed', 'field_observed_after']
    stored_obs_data = pd.DataFrame(columns=obs_columns)

    other_obs_columns = ['transient_id', 'mjd', 'bandfilter',
                         'instrument_magnitude', 'instrument_mag_one_sigma',
                         'instrument_flux', 'instrument_flux_one_sigma',
                         'signal_to_noise', 'airmass',
                         'five_sigma_depth', 'when']
    stored_other_obs_data = pd.DataFrame(columns=other_obs_columns)

    if rank == 0 and verbose:
        print('\nLaunching multiprocess pool of {} workers per MPI core.'.format(batch_mp_workers))
        print('The batch processing will now begin.')
        t0 = time.time()

    if size > 1:
        comm.barrier()
    # Launch x threads per MPI worker
    p = mp.Pool(batch_mp_workers)
    for i in range(num_batches):
        # Handle uneven batch sizes
        if i == (num_batches-1):
            current_batch_size = num_params_pprocess - i*batch_size
        else:
            current_batch_size = batch_size

        transient_batch = []

        if 'path' in getattr(atopclass, transient_model_name).__init__.__code__.co_varnames:
            batch_params = list(zip(repeat(seds_path, current_batch_size)))
        elif param_array is not None:
            batch_params = param_array[i*batch_size:i*batch_size +
                                       current_batch_size, :]
        batch_sky_loc = sky_loc_array[i*batch_size:i*batch_size +
                                      current_batch_size, :]

        transient_batch = p.starmap(getattr(atopclass, transient_model_name), batch_params)
        args_for_method = list(zip(batch_sky_loc.tolist(), repeat([cosmo])))

        extended_args = p.starmap(afunc.extend_args_list, args_for_method)

        batch_method_iter_for_pool = list(zip(transient_batch,
                                              repeat('put_in_universe'),
                                              extended_args))

        transient_batch = p.starmap(afunc.class_method_in_pool,
                                    batch_method_iter_for_pool)

        parameter_batch_iter = list(zip(repeat(param_columns),
                                        transient_batch))

        parameter_df = p.starmap(afunc.write_params, parameter_batch_iter)
        stored_param_data = stored_param_data.append(parameter_df,
                                                     ignore_index=True, sort=False)

        observation_batch_iter = list(zip(repeat(obs_columns),
                                          transient_batch,
                                          repeat(survey)))

        observation_df = p.starmap(afunc.observe, observation_batch_iter)
        stored_obs_data = stored_obs_data.append(observation_df,
                                                 ignore_index=True, sort=False)

        other_obs_iter = list(zip(repeat(survey), parameter_df,
                                  repeat(sim_inst.t_before), repeat(sim_inst.t_after),
                                  repeat(other_obs_columns)))

        # get other observations
        other_obs_df = p.starmap(afunc.other_observations, other_obs_iter)
        stored_other_obs_data = stored_other_obs_data.append(other_obs_df,
                                                            ignore_index=True,
                                                            sort=False)
        if rank == 0 and verbose:
            # Write computaiton time estimates
            print('Batch {0} complete of {1} batches.'.format(i+1, num_batches))
            t1 = time.time()
            delta_t = int(((t1-t0)/(i+1))*((num_params_pprocess-(i+1)*current_batch_size)/(batch_size)) + 0.1045*transient_dist.number_simulated)
            print('Estimated time remaining is: {}'.format(datetime.timedelta(seconds=delta_t)))

    # Now process observations for detections and other information
    transient_batch = None
    batch_method_iter_for_pool = None
    parameter_batch_iter = None
    observation_batch_iter = None
    other_obs_iter = None

    if rank == 0 and verbose:
        print('Processing transients for alert triggers.')
    stored_obs_data = afunc.efficiency_process(survey, stored_obs_data)

    # Process the pandas dataframes for output and shared usage
    if size > 1:
        comm.barrier()

    stored_obs_data.dropna(inplace=True)
    stored_param_data.dropna(inplace=True)
    stored_other_obs_data.dropna(inplace=True)

    # Join all batches and mpi workers and write the dataFrame to file
    if size > 1:
        obs_receive = comm.allgather(stored_obs_data)
        params_receive = comm.allgather(stored_param_data)
        other_obs_receive = comm.allgather(stored_other_obs_data)

        output_params = pd.concat(params_receive, sort=False, ignore_index=True)
        output_observations = pd.concat(obs_receive, sort=False, ignore_index=True)
        output_other_observations = pd.concat(other_obs_receive, sort=False, ignore_index=True)

    else:
        output_observations = stored_obs_data
        output_params = stored_param_data
        output_other_observations = stored_other_obs_data

    if rank == 0:
        if verbose:
            print('\nWriting out parameters and observations to {}'.format(output_path))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_params.to_csv(output_path + 'parameters.csv')
        output_observations.to_csv(output_path + 'observations.csv')
        output_other_observations.to_csv(output_path + 'other_observations.csv')
        if verbose:
            print('Finished writing observation results.')

    stored_obs_data = output_observations
    stored_param_data = output_params
    stored_other_obs_data = output_other_observations
    output_params = None
    output_observations = None
    output_other_observations = None
    obs_receive = None
    params_receive = None
    other_obs_receive = None

    if size > 1:
        comm.barrier()

    # Detections
    if rank == 0:
        if verbose:
            print('\nNow processing for detections.')

        # Do coadds first
        if verbose:
            print('Doing nightly coadds...')
        coadded_observations = afunc.process_nightly_coadds(stored_obs_data, survey)

        if verbose:
            print('Processing coadded nights for transients alert triggers.')
        coadded_observations.drop(columns=['alert'], inplace=True)
        coadded_observations = afunc.efficiency_process(survey, coadded_observations)

        intermediate_filter = coadded_observations
        for type in detect_type:
            if verbose and type == 'scolnic_detections':
                print('Processing coadded observations for detections in line with Scolnic et. al 2018.')
                intermediate_filter = getattr(afunc, type)(stored_param_data, intermediate_filter, stored_other_obs_data, survey)
            else:
                if verbose:
                    print('Processing coadded observations with the given filter dictionary.')
                intermediate_filter = getattr(afunc, type)(intermediate_filter, filters)
        detected_observations = intermediate_filter

        stored_param_data = afunc.param_observe_detect(stored_param_data, stored_obs_data, detected_observations)

        # Get efficiencies and create redshift histogram
        redshift_histogram = afunc.redshift_distribution(stored_param_data, sim_inst)

        if verbose:
            print('Outputting coadded observations, scolnic detections, parameters modified with observed, alerted, and detected flags, and the redshift distribution.')
        coadded_observations.to_csv(output_path + 'coadded_observations.csv')
        detected_observations.to_csv(output_path + 'scolnic_detections.csv')
        stored_param_data.to_csv(output_path + 'modified_parameters.csv')
        redshift_histogram.savefig(output_path + 'redshift_distribution.pdf', bbox_inches='tight')
        if verbose:
            print('Done writing the detection results.')

    if size > 1:
        comm.barrier()

    if rank == 0 and verbose:
        t_end = time.time()
        #print('\n Estimated time for the last bit is: {}'.format((t_end-t1)/transient_dist.number_simulated))
        print('\nSimulation completed successfully with elapsed time: {}.'.format(datetime.timedelta(seconds=int(t_end-t_start))))