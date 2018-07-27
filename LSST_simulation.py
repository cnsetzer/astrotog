import os
import re
import numpy as np
from math import *
from astropy.cosmology import Planck15 as cosmo
from astrotog import*
from mpi4py import MPI
import multiprocessing as mp
from itertools import repeat
from time import time
import datetime
from copy import copy, deepcopy
import pandas as pd
np.random.seed(12345)


if __name__ == "__main__":
    # MPI.Init_thread(int required=THREAD_MULTIPLE)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # ---------------------------------------------------------------
    # Section that user can edit to tailor simulation on global level
    # ---------------------------------------------------------------
    batch_size = 'all'

    if rank == 0:
        # -------------------------------------------------------------------
        # Section that user can edit to tailor simulation on primary process
        # -------------------------------------------------------------------
        seds_path = '/Users/cnsetzer/Documents/LSST/sedb/rosswog/NSNS/winds'
        cadence_path = \
            '/Users/cnsetzer/Documents/LSST/surveydbs/minion_1016_sqlite.db'
        throughputs_path = '/Users/cnsetzer/Documents/LSST/throughputs'
        reference_flux_path = '/Users/cnsetzer/Documents/LSST/throughputs/references'
        run_dir = 'LSST_sim_run_' + datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        output_path = '/Users/cnsetzer/Documents/LSST/astrotog_output/' + run_dir + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        z_max = 0.3
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

        sim_inst = simulation(cadence_path=cadence_path,
                              throughputs_path=throughputs_path,
                              reference_path=reference_flux_path, z_max=z_max,
                              output_path=output_path,
                              cadence_flags=cadence_flags, z_min=z_min,
                              z_bin_size=z_bin_size, multiproc=multiprocess,
                              num_processes=num_processes,
                              batch_size=batch_size, cosmology=cosmo,
                              rate_gpc=rate)
        LSST_survey = LSST(sim_inst)
        transient_dist = transient_distribution(LSST_survey, sim_inst)
        tran_param_dist = rosswog_kilonovae(parameter_dist=True,
                                            num_samples=transient_dist.number_simulated)
        num_params_pprocess = int(ceil(transient_dist.number_simulated/size))
    else:
        num_params_pprocess = None
        LSST_survey = None

    if size > 1:
        num_transient_params = comm.bcast(tran_param_dist.num_params, root=0)
        num_params_pprocess = comm.bcast(num_params_pprocess, root=0)

    if rank == 0:
        sky_loc_array = np.vstack((transient_dist.time_dist,
                                   transient_dist.ra_dist,
                                   transient_dist.de_dist,
                                   transient_dist.redshift_dist))
        param_array = np.vstack(tran_param_dist.m_ej, tran_param_dist.v_ej,
                                tran_param_dist.kappa)
    else:
        sky_loc_array = np.empty((num_params_pprocess, 4))
        param_array = np.empty((num_params_pprocess, num_transient_params))

    if size > 1:
        comm.Scatter([sky_loc_array, 4*num_params_pprocess, MPI.DOUBLE],
                 [sky_loc_array, 4*num_params_pprocess, MPI.DOUBLE], root=0)
        comm.Scatter([param_array, num_transient_params*num_params_pprocess,
                     MPI.DOUBLE], [param_array,
                     num_transient_params*num_params_pprocess, MPI.DOUBLE],
                     root=0)

    # Trim the nonsense from the process arrays
    for i in range(num_params_pprocess):
        if any(sky_loc_array[i] < 1e-250):
            sky_loc_array = np.delete(sky_loc_array, i, 0)
        if any(param_array[i] < 1e-250):
            param_array = np.delete(param_array, i, 0)

    if size > 1:
        # There is definitely a problem here. Might need a virtual server...
        # ----------------------------------------------
        LSST_survey = comm.bcast(LSST_survey.copy(), root=0)
        # ----------------------------------------------`



    assert len(param_array[:]) == len(sky_loc_array[:])
    num_params_pprocess = len(param_array[:])

    if batch_size == 'all':
        batch_size = num_params_pprocess
        num_batches = 1
    else:
        num_batches = int(ceil(num_params_pprocess/batch_size))

    param_index = 0
    # Launch 2 threads per MPI worker
    p = mp.Pool(2)
    for i in range(num_batches):
        # Handle uneven batch sizes
        if i == num_batches-1:
            current_batch_size = num_params_pprocess - i*batch_size
        else:
            current_batch_size = batch_size

        transient_batch = []
        batch_params = param_array[i*batch_size:i*batch_size +
                                   current_batch_size-1, :]
        batch_sky_loc = sky_loc_array[i*batch_size:i*batch_size +
                                      current_batch_size-1, :]

        transient_batch = p.starmap(rosswog_kilonovae, batch_params)

        batch_method_iter_for_pool = list(zip(transient_batch,
                                              repeat('put_in_universe'),
                                              batch_sky_loc, repeat(cosmo)))
        transient_batch = p.starmap(class_method_in_pool,
                                    batch_method_iter_for_pool)
        observation_batch_iter = list(zip(transient_batch, repeat(LSST_survey)))
        observations = p.starmap(observe, observation_batch_iter)
