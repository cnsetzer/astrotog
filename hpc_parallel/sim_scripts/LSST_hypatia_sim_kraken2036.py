import os
import numpy as np
from math import ceil
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
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# Set seed for reproduceability
np.random.seed(12345)
# Execute parallel script only if used as the main script
if __name__ == "__main__":
    # MPI.Init_thread(int required=THREAD_MULTIPLE)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # ---------------------------------------------------------------
    # Section that user can edit to tailor simulation on global level
    # ---------------------------------------------------------------
    batch_size = 200
    batch_mp_workers = 2
    verbose = False
    # batch_size = 'all'

    if rank == 0:
        t_start = time.time()
        os.system('clear')
        if verbose:
            print('\nWe are using {0} MPI workers with {1} multiprocess threads per process.'.format(size, batch_mp_workers))
        # -------------------------------------------------------------------
        # Section that user can edit to tailor simulation on primary process
        # -------------------------------------------------------------------
        dithers = False
        survey_version = 'lsstv4'
        add_dithers = False

        seds_path = '/Users/cnsetzer/Documents/LSST/sedb/rosswog/NSNS/winds'
        cadence_path = \
            '/home/csetzer/LSST/OpSim_outputs/kraken_2036.db'
        throughputs_path = '/home/csetzer/LSST/throughputs/lsst'
        reference_flux_path = '/home/csetzer/LSST/throughputs/references'
        run_dir = 'LSST_sim_run_kraken2036_' + datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        output_path = '/home/csetzer/LSST/astrotog_output/' + run_dir + '/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        z_max = 0.5
        num_processes = size
        z_bin_size = 0.02
        if size >= 1:
            multiprocess = True
        else:
            multiprocess = False
        transient_rate_gpc = 1000.0  # Currently this is the default
        z_min = 0.0  # Currently this is the default in the simulation class
        cadence_flags = 'combined'  # Currently use default in class
        rate = 1000
        # -----------------------------------------------
        # -----------------------------------------------

        sim_inst = atopclass.simulation(cadence_path=cadence_path,
                              throughputs_path=throughputs_path,
                              reference_path=reference_flux_path, z_max=z_max,
                              output_path=output_path,
                              cadence_flags=cadence_flags, z_min=z_min,
                              z_bin_size=z_bin_size, multiproc=multiprocess,
                              num_processes=num_processes,
                              batch_size=batch_size, cosmology=cosmo,
                              rate_gpc=rate,
                              dithers=dithers, simversion=survey_version,
                              add_dithers=add_dithers)
        LSST_survey = atopclass.LSST(sim_inst)
        transient_dist = aclasses.transient_distribution(LSST_survey, sim_inst)
        tran_param_dist = atopclass.rosswog_kilonovae(parameter_dist=True,
                                            num_samples=transient_dist.number_simulated)
        num_params_pprocess = int(ceil(transient_dist.number_simulated/size))
        num_transient_params = tran_param_dist.num_params
        if verbose:
            print('The number of transients is: {}'.format(transient_dist.number_simulated))
            print('The number of parameters per transient is: {}'.format(num_transient_params))
    else:
        num_transient_params = None
        num_params_pprocess = None
        LSST_survey = None

    if size > 1:
        comm.barrier()
        num_transient_params = comm.bcast(num_transient_params, root=0)
        num_params_pprocess = comm.bcast(num_params_pprocess, root=0)

    total_array = None
    if rank == 0:
        sky_loc_array = np.hstack((transient_dist.ids,
                                   transient_dist.time_dist,
                                   transient_dist.ra_dist,
                                   transient_dist.dec_dist,
                                   transient_dist.redshift_dist))
        param_array = np.hstack((tran_param_dist.m_ej, tran_param_dist.v_ej,
                                tran_param_dist.kappa))

    if size > 1:
        if rank == 0:
            total_array = np.hstack((sky_loc_array, param_array))

        sky_loc_array = None
        param_array = None
        receive_array = np.empty((num_params_pprocess, 5 + num_transient_params))
        comm.barrier()
        comm.Scatter([total_array, (5+num_transient_params)*num_params_pprocess, MPI.DOUBLE],
                 [receive_array, (5+num_transient_params)*num_params_pprocess, MPI.DOUBLE], root=0)
        sky_loc_array = receive_array[:, 0:5]
        param_array = receive_array[:, 5:]

    # Trim the nonsense from the process arrays
    sky_del = []
    param_del = []
    for i in range(num_params_pprocess):
        if any(abs(sky_loc_array[i]) < 1e-250):
            sky_del.append(i)
        if any(abs(param_array[i]) < 1e-250):
            param_del.append(i)

    sky_loc_array = np.delete(sky_loc_array, sky_del, 0)
    param_array = np.delete(param_array, param_del, 0)

    assert len(param_array[:]) == len(sky_loc_array[:])
    num_params_pprocess = len(param_array[:])

    if rank == 0 and verbose:
        print('The split of parameters between processes is:')
    if verbose:
        print('Process rank = {0} has {1} transients assigned to it.'.format(rank, num_params_pprocess))

    if size > 1:
        # There is definitely a problem here. Might need a virtual server...
        # ----------------------------------------------
        comm.barrier()
        LSST_survey = comm.bcast(copy(LSST_survey), root=0)
        # ----------------------------------------------`

    if batch_size == 'all':
        batch_size = num_params_pprocess
        num_batches = 1
    else:
        num_batches = int(ceil(num_params_pprocess/batch_size))

    # Create pandas table
    param_columns = ['transient id', 'm_ej', 'v_ej', 'kappa', 'true redshift',
                     'explosion time', 'max time', 'ra', 'dec',
                     'peculiar velocity']

    stored_param_data = pd.DataFrame(columns=param_columns)

    obs_columns = ['transient id', 'mjd', 'bandfilter', 'instrument magnitude',
                      'instrument mag one sigma', 'instrument flux',
                      'instrument flux one sigma', 'A_x',
                      'signal to noise', 'source magnitude',
                      'source mag one sigma', 'source flux',
                      'source flux one sigma', 'extincted magnitude',
                      'extincted mag one sigma', 'extincted flux',
                      'extincted flux one sigma', 'airmass',
                      'five sigma depth', 'lightcurve phase',
                      'field previously observed', 'field observed after']
    stored_obs_data = pd.DataFrame(columns=obs_columns)

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
        batch_params = param_array[i*batch_size:i*batch_size +
                                   current_batch_size, :]
        batch_sky_loc = sky_loc_array[i*batch_size:i*batch_size +
                                      current_batch_size, :]
        transient_batch = p.starmap(atopclass.rosswog_kilonovae, batch_params)
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
                                          repeat(LSST_survey)))

        observation_df = p.starmap(afunc.observe, observation_batch_iter)
        stored_obs_data = stored_obs_data.append(observation_df,
                                                 ignore_index=True, sort=False)

        if rank == 0 and verbose:
            # if i == 0:
            #     print('Debug Check of Pandas Dataframe:')
            #     print(stored_obs_data)

            print('Batch {0} complete of {1} batches.'.format(i+1, num_batches))
            t1 = time.time()
            delta_t = int(((t1-t0)/(i+1))*(num_batches-i-1) + 15.0)  # estimated write time
            print('Estimated time remaining is: {}'.format(datetime.timedelta(seconds=delta_t)))
    # Now process observations for detections and other information
    transient_batch = None
    observation_batch_iter = None

    # Process the pandas dataframes for output and shared usage
    if size > 1:
        comm.barrier()

    stored_obs_data.dropna(inplace=True)
    stored_param_data.dropna(inplace=True)

    # Join all batches and mpi workers and write the dataFrame to file
    if size > 1:
        obs_receive = comm.allgather(stored_obs_data)
        params_receive = comm.allgather(stored_param_data)

        output_params = pd.DataFrame(columns=param_columns)
        output_observations = pd.DataFrame(columns=obs_columns)

        # Create a single pandas dataframe
        for i in range(size):
            output_params = output_params.append(params_receive[i], sort=False)
            output_observations = output_observations.append(obs_receive[i], sort=False)

    else:
        output_observations = stored_obs_data
        output_params = stored_param_data

    if rank == 0:
        if verbose:
            print('\nWriting out parameters and observations to {}'.format(output_path))
        output_params.to_csv(output_path + 'parameters.csv')
        output_observations.to_csv(output_path + 'observations.csv')
        if verbose:
            print('Finished writing observation results.')

    stored_obs_data = output_observations
    stored_param_data = stored_param_data
    output_params = None
    output_observations = None
    obs_receive = None
    params_receive = None

    comm.barrier()
    # detections = p.starmap()

    if rank == 0 and verbose:
        t_end = time.time()
        print('\nSimulation completed successfully with elapsed time: {}.'.format(datetime.timedelta(seconds=int(t_end-t_start))))
