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
    batch_size = 50
    batch_mp_workers = 2
    verbose = True
    # batch_size = 'all'

    if rank == 0:
        t_start = time.time()
        os.system('clear')
        if verbose:
            print('\nWe are using {0} MPI workers with {1} multiprocess threads per process.'.format(size, batch_mp_workers))
        # -------------------------------------------------------------------
        # Section that user can edit to tailor simulation on primary process
        # -------------------------------------------------------------------
        run_dir = 'LSST_sim_run_kraken2036_' + datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        output_path = '/Users/cnsetzer/Documents/LSST/astrotog_output/' + run_dir + '/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        num_processes = size
        z_bin_size = 0.02
        if size >= 1:
            multiprocess = True
        else:
            multiprocess = False
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


    other_obs_columns = ['transient id', 'mjd', 'bandfilter', 'instrument magnitude',
                      'instrument mag one sigma', 'instrument flux',
                      'instrument flux one sigma',
                      'signal to noise', 'airmass',
                      'five sigma depth', 'when']
    other_obs_data = pd.DataFrame(columns=ohter_obs_columns)

    comm.barrier()
    # detections = p.starmap()

    if rank == 0 and verbose:
        t_end = time.time()
        print('\nSimulation completed successfully with elapsed time: {}.'.format(datetime.timedelta(seconds=int(t_end-t_start))))
