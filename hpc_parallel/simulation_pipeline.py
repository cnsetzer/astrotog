import os
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
seed = np.int(sys.argv[2])
np.random.seed(seed)
pd.options.mode.chained_assignment = None  # None|'warn'|'raise'

# Execute parallel script only if used as the main script
if __name__ == "__main__":
    # MPI.Init_thread(int required=THREAD_MULTIPLE)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("\nThe random seed for this sample is: {}".format(seed))

        t_start = time.time()
        if size == 1:
            os.system("clear")
        else:
            print("\n ")

        exec("from {} import *".format(sys.argv[1]))
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("\n")
                f.write("This is the debug log for {}".format(output_path))
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Creating the simulation instance")
        sim_inst = atopclass.simulation(
            cadence_path=cadence_path,
            throughputs_path=throughputs_path,
            reference_path=reference_flux_path,
            z_max=z_max,
            output_path=output_path,
            cadence_flags=cadence_flags,
            ra_col=cadence_ra_col,
            dec_col=cadence_dec_col,
            z_min=z_min,
            z_bin_size=z_bin_size,
            batch_size=batch_size,
            cosmology=cosmo,
            rate_gpc=rate,
            dithers=dithers,
            simversion=survey_version,
            add_dithers=add_dithers,
            t_before=t_before,
            t_after=t_after,
            response_path=efficiency_table_path,
            instrument=instrument_class_name,
            filter_null=cadence_has_nulls,
            desc_dithers=desc_dithers,
            dither_path=dither_path,
            same_dist=same_dist,
            min_dec=min_dec,
            max_dec=max_dec,
            trans_duration=transient_duration,
        )
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Getting the survey library")
        survey = getattr(atopclass, instrument_class_name)(sim_inst)
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Creating the transient distribution")
        transient_dist = aclasses.transient_distribution(survey, sim_inst)
        tran_param_dist = getattr(atopclass, transient_model_name)(
            parameter_dist=True, num_samples=transient_dist.number_simulated
        )
        num_params_pprocess = int(np.ceil(transient_dist.number_simulated / size))
        num_transient_params = tran_param_dist.num_params
        pre_dist_params = tran_param_dist.pre_dist_params
        if verbose:
            print(
                "\nWe are using {0} MPI workers with {1} multiprocess threads per process.".format(
                    size, batch_mp_workers
                )
            )
            print(
                "The number of transients is: {}".format(
                    transient_dist.number_simulated
                )
            )
            print(
                "The number of parameters per transient is: {}".format(
                    num_transient_params
                )
            )
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write(
                    "\nWe are using {0} MPI workers with {1} multiprocess threads per process.".format(
                        size, batch_mp_workers
                    )
                )
                f.write(
                    "The number of transients is: {}".format(
                        transient_dist.number_simulated
                    )
                )
                f.write(
                    "The number of parameters per transient is: {}".format(
                        num_transient_params
                    )
                )
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
        detect_type = None

    if rank == 0 and debug is True:
        with open(debug_file, mode="a") as f:
            f.write("\n")
            f.write("-------------Debug:-------------")
            f.write(
                "Broadcasting simulation and transient information to all processes"
            )
    if size > 1:
        comm.barrier()
        detect_type = comm.bcast(detect_type, root=0)
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
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Constructing arrays of transient distribution parameters")
        sky_loc_array = np.hstack(
            (
                transient_dist.ids,
                transient_dist.time_dist,
                transient_dist.ra_dist,
                transient_dist.dec_dist,
                transient_dist.redshift_dist,
            )
        )
        stack_list = []
        if num_transient_params > 0 and pre_dist_params is True:
            for i in range(num_transient_params):
                stack_list.append(getattr(tran_param_dist, "param{0}".format(i + 1)))
            param_array = np.hstack(stack_list)
        else:
            param_array = None

    if size > 1:
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write(
                    "Scattering the parameters of the distribution between mpi processes"
                )
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

        receive_array = np.empty((num_params_pprocess, 5 + param_buffer_size))
        comm.barrier()
        comm.Scatter(
            [total_array, (5 + param_buffer_size) * num_params_pprocess, MPI.DOUBLE],
            [receive_array, (5 + param_buffer_size) * num_params_pprocess, MPI.DOUBLE],
            root=0,
        )
        sky_loc_array = receive_array[:, 0:5]
        if num_transient_params > 0 and pre_dist_params is True:
            param_array = receive_array[:, 5:]

    # Empty buffers
    total_array = None
    receive_array = None

    if rank == 0 and debug is True:
        with open(debug_file, mode="a") as f:
            f.write("\n")
            f.write("-------------Debug:-------------")
            f.write(
                "Delete the empty array rows for the last mpi process due to uneven division of parameters per processes"
            )
    if rank == size - 1:
        # Trim the nonsense from the process arrays
        sky_del = []
        param_del = []
        for i in range(num_params_pprocess):
            if any(elem == np.nan for elem in sky_loc_array[i]):
                sky_del.append(i)
                print("Nan encountered.")
            elif any(abs(sky_loc_array[i]) < 1e-250):
                sky_del.append(i)
            if param_array is not None:
                if any(abs(param_array[i]) < 1e-250):
                    param_del.append(i)

        sky_loc_array = np.delete(sky_loc_array, sky_del, 0)
        if param_array is not None:
            param_array = np.delete(param_array, param_del, 0)
            assert len(param_array[:]) == len(sky_loc_array[:])
        num_params_pprocess = len(sky_loc_array[:])

        # Empty del lists
        sky_del = None
        param_del = None

    if rank == 0 and verbose:
        print("The split of parameters between processes is:")
    if verbose:
        print(
            "Process rank = {0} has {1} transients assigned to it.".format(
                rank, num_params_pprocess
            )
        )

    if size > 1:
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Distribute survey and simulation instances to all processes.")
        # There is definitely an inefficiency here. Might need a virtual server...
        # ----------------------------------------------
        comm.barrier()
        survey = comm.bcast(survey, root=0)
        sim_inst = comm.bcast(sim_inst, root=0)
        # ----------------------------------------------`

    if batch_size == "all":
        batch_size = num_params_pprocess
        num_batches = 1
    else:
        num_batches = int(np.ceil(num_params_pprocess / batch_size))

    # Create pandas table
    param_columns = [
        "transient_id",
        "true_redshift",
        "obs_redshift",
        "explosion_time",
        "max_time",
        "ra",
        "dec",
        "peculiar_velocity",
    ]

    # Add model specific transient parameters to array
    if num_transient_params > 0:
        for i in range(num_transient_params):
            param_columns.append(
                getattr(tran_param_dist, "param{0}_name".format(i + 1))
            )

    stored_param_data = pd.DataFrame(columns=param_columns)

    # Empty variables to control memory usage
    tran_param_dist = None

    obs_columns = [
        "transient_id",
        "mjd",
        "bandfilter",
        "instrument_magnitude",
        "instrument_mag_one_sigma",
        "instrument_flux",
        "instrument_flux_one_sigma",
        "A_x",
        "signal_to_noise",
        "source_magnitude",
        "source_mag_one_sigma",
        "source_flux",
        "source_flux_one_sigma",
        "extincted_magnitude",
        "extincted_mag_one_sigma",
        "extincted_flux",
        "extincted_flux_one_sigma",
        "airmass",
        "five_sigma_depth",
        "lightcurve_phase",
        "field_previously_observed",
        "field_observed_after",
    ]
    stored_obs_data = pd.DataFrame(columns=obs_columns)

    other_obs_columns = [
        "transient_id",
        "mjd",
        "bandfilter",
        "instrument_magnitude",
        "instrument_mag_one_sigma",
        "instrument_flux",
        "instrument_flux_one_sigma",
        "signal_to_noise",
        "airmass",
        "five_sigma_depth",
        "when",
    ]
    stored_other_obs_data = pd.DataFrame(columns=other_obs_columns)

    if rank == 0 and verbose:
        print(
            "\nLaunching multiprocess pool of {} workers per MPI core.".format(
                batch_mp_workers
            )
        )
        print("The batch processing will now begin.")
        t0 = time.time()
        t_mod1 = t0

    if rank == 0 and debug is True:
        with open(debug_file, mode="a") as f:
            f.write("\n")
            f.write(
                "\nLaunching multiprocess pool of {} workers per MPI core.".format(
                    batch_mp_workers
                )
            )
            f.write("\nThe batch processing will now begin.")

    if size > 1:
        comm.barrier()
    # Launch x threads per MPI worker
    p = mp.Pool(batch_mp_workers)
    for i in range(num_batches):
        # Handle uneven batch sizes
        if i == (num_batches - 1):
            current_batch_size = num_params_pprocess - i * batch_size
        else:
            current_batch_size = batch_size

        transient_batch = []

        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write(
                    "For batch {}, create parameter sub arrays for generation of transients in this batch.".format(
                        i
                    )
                )

        if (
            "path"
            in getattr(atopclass, transient_model_name).__init__.__code__.co_varnames
        ):
            batch_params = list(zip(repeat(seds_path, current_batch_size)))
        elif param_array is not None:
            batch_params = param_array[
                i * batch_size : i * batch_size + current_batch_size, :
            ]
        batch_sky_loc = sky_loc_array[
            i * batch_size : i * batch_size + current_batch_size, :
        ]
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("For batch {}, create transient seds.".format(i))

        transient_batch = p.starmap(
            getattr(atopclass, transient_model_name), batch_params
        )
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write(
                    "For batch {}, extend arguments list for proper format to use class method in parallel.".format(
                        i
                    )
                )
        args_for_method = list(zip(batch_sky_loc.tolist(), repeat([cosmo])))
        extended_args = p.starmap(afunc.extend_args_list, args_for_method)

        # Try to reorganize for better memory management
        batch_sky_loc = None
        batch_params = None
        args_for_method = None

        batch_method_iter_for_pool = list(
            zip(transient_batch, repeat("put_in_universe"), extended_args)
        )
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write(
                    "For batch {}, execute class method 'put in universe' in parallel.".format(
                        i
                    )
                )

        transient_batch = p.starmap(
            afunc.class_method_in_pool, batch_method_iter_for_pool
        )

        # Try to reorganize for better memory management
        batch_method_iter_for_pool = None

        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write(
                    "For batch {}, write transient parameters to dataframe.".format(i)
                )

        parameter_batch_iter = list(zip(repeat(param_columns), transient_batch))

        parameter_df = p.starmap(afunc.write_params, parameter_batch_iter)
        stored_param_data = stored_param_data.append(
            parameter_df, ignore_index=True, sort=False
        )
        # Try to reorganize for better memory management
        parameter_batch_iter = None

        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write(
                    "For batch {}, execute observations of current batch of transients.".format(
                        i
                    )
                )

        observation_batch_iter = list(
            zip(repeat(obs_columns), transient_batch, repeat(survey))
        )

        observation_df = p.starmap(afunc.observe, observation_batch_iter)

        # Try to reorganize for better memory management
        transient_batch = None
        observation_batch_iter = None

        stored_obs_data = stored_obs_data.append(
            observation_df, ignore_index=True, sort=False
        )

        # Try to reorganize for better memory management
        observation_df = None
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write(
                    "For batch {}, find observation non-detections on either side of each transient given the specfied time windows.".format(
                        i
                    )
                )
        other_obs_iter = list(
            zip(
                repeat(survey),
                parameter_df,
                repeat(sim_inst.t_before),
                repeat(sim_inst.t_after),
                repeat(other_obs_columns),
            )
        )

        # get other observations
        other_obs_df = p.starmap(afunc.other_observations, other_obs_iter)
        stored_other_obs_data = stored_other_obs_data.append(
            other_obs_df, ignore_index=True, sort=False
        )
        # Try to reorganize for better memory management
        other_obs_iter = None
        parameter_df = None
        other_obs_df = None

        if rank == 0 and verbose:
            # Write computaiton time estimates
            print("Batch {0} complete of {1} batches.".format(i + 1, num_batches))
            if debug is True:
                with open(debug_file, mode="a") as f:
                    f.write(
                        "Batch {0} complete of {1} batches.".format(i + 1, num_batches)
                    )
            t1 = time.time()
            delta_t = int(
                ((t1 - t0) / (i + 1))
                * (
                    (num_params_pprocess - (i * batch_size + current_batch_size))
                    / (batch_size)
                )
                + 0.015 * transient_dist.number_simulated
            )
            print(
                "Estimated time remaining is: {}".format(
                    datetime.timedelta(seconds=delta_t)
                )
            )
            if debug is True:
                with open(debug_file, mode="a") as f:
                    f.write(
                        "Estimated time remaining is: {}".format(
                            datetime.timedelta(seconds=delta_t)
                        )
                    )

    if rank == 0 and verbose:
        t_mod2 = time.time()
        print(
            "\nEstimated time for generating and observing the transients per transient is: {}".format(
                (t_mod2 - t_mod1) / transient_dist.number_simulated
            )
        )
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write(
                    "\nEstimated time for generating and observing the transients per transient is: {}".format(
                        (t_mod2 - t_mod1) / transient_dist.number_simulated
                    )
                )

    # Empty arrays as no longer needed
    sky_loc_array = None
    param_array = None
    # Now process observations for detections and other information
    if rank == 0 and verbose:
        print("\nProcessing transients for alert triggers.")
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\nProcessing transients for alert triggers.")
    stored_obs_data = afunc.efficiency_process(survey, stored_obs_data)

    # Process the pandas dataframes for output and shared usage
    if size > 1:
        comm.barrier()

    stored_obs_data.dropna(inplace=True)
    stored_param_data.dropna(inplace=True)
    stored_other_obs_data.dropna(inplace=True)

    # Join all batches and mpi workers and write the dataFrame to file
    if size > 1:
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Allgather the parameter and observation data.")
        obs_receive = comm.allgather(stored_obs_data)
        params_receive = comm.allgather(stored_param_data)
        other_obs_receive = comm.allgather(stored_other_obs_data)

        output_params = pd.concat(params_receive, sort=False, ignore_index=True)
        output_observations = pd.concat(obs_receive, sort=False, ignore_index=True)
        output_other_observations = pd.concat(
            other_obs_receive, sort=False, ignore_index=True
        )

    else:
        output_observations = stored_obs_data
        output_params = stored_param_data
        output_other_observations = stored_other_obs_data

    if rank == 0:
        if verbose and save_all_output:
            print("\nWriting out parameters and observations to {}".format(output_path))
            if debug is True:
                with open(debug_file, mode="a") as f:
                    f.write(
                        "\nWriting out parameters and observations to {}".format(
                            output_path
                        )
                    )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if save_all_output is True:
            # output_params.to_csv(output_path + 'parameters.csv')
            output_observations.to_csv(output_path + "observations.csv")
            output_other_observations.to_csv(output_path + "other_observations.csv")
            if verbose:
                print("Finished writing observation results.")
            if debug is True:
                with open(debug_file, mode="a") as f:
                    f.write("\nFinished writing observation results.")

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
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Separate the dataframes per process.")

        comm.barrier()
        # Split back into processes
        if rank == 0:
            num_params_last = (
                transient_dist.number_simulated - (size - 1) * num_params_pprocess
            )
            comm.send(num_params_last, dest=size - 1, tag=11)
            ids_per_process = list(
                np.arange(
                    start=num_params_pprocess * rank + 1,
                    stop=num_params_pprocess * (rank + 1) + 1,
                )
            )
        elif rank == size - 1:
            num_params_last = comm.recv(source=0, tag=11)
            ids_per_process = list(
                np.arange(
                    start=num_params_pprocess * rank + 1,
                    stop=num_params_pprocess * rank + 1 + num_params_last,
                )
            )
        else:
            ids_per_process = list(
                np.arange(
                    start=num_params_pprocess * rank + 1,
                    stop=num_params_pprocess * (rank + 1) + 1,
                )
            )
    else:
        ids_per_process = list(
            np.arange(start=1, stop=transient_dist.number_simulated + 1)
        )

    # Split stored_obs_data, stored_param_data, stored_other_obs_data
    process_obs_data = stored_obs_data[
        stored_obs_data["transient_id"].isin(ids_per_process)
    ]
    process_other_obs_data = stored_other_obs_data[
        stored_other_obs_data["transient_id"].isin(ids_per_process)
    ]
    process_param_data = stored_param_data[
        stored_param_data["transient_id"].isin(ids_per_process)
    ]

    stored_obs_data = None
    stored_param_data = None
    stored_other_obs_data = None

    # Detections
    if verbose and rank == 0:
        print("\nNow processing for detections.")
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\nNow processing for detections.")

    # Do coadds first
    if verbose and rank == 0:
        print("Doing nightly coadds...")
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\nDoing nightly coadds...")
    coadded_observations = afunc.process_nightly_coadds(process_obs_data, survey)

    if verbose and rank == 0:
        print("Processing coadded nights for transients alert triggers.")
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\nProcessing coadded nights for transients alert triggers.")
    coadded_observations.drop(columns=["alert"], inplace=True)
    coadded_observations = afunc.efficiency_process(survey, coadded_observations)

    for type in detect_type:
        if type == "scolnic_detections":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections in line with Scolnic et. al 2018."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections in line with Scolnic et. al 2018."
                        )
            detected_observations = getattr(afunc, type)(
                coadded_observations, process_other_obs_data
            )
        elif type == "scolnic_detections_no_coadd":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections in line with Scolnic et. al 2018, but no coadds."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections in line with Scolnic et. al 2018, but no coadds."
                        )
            detected_observations2 = getattr(afunc, "scolnic_detections")(
                process_obs_data, process_other_obs_data
            )
        elif type == "scolnic_like_detections":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections like Scolnic et. al 2018, but with alerts instead of SNR>5."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections like Scolnic et. al 2018, but with alerts instead of SNR>5."
                        )
            detected_observations3 = getattr(afunc, "scolnic_detections")(
                coadded_observations, process_other_obs_data, alerts=True
            )
        elif type == "scolnic_like_detections_no_coadd":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections like Scolnic et. al 2018, but with alerts instead of SNR>5 and no coadds."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections like Scolnic et. al 2018, but with alerts instead of SNR>5 and no coadds."
                        )
            detected_observations4 = getattr(afunc, "scolnic_detections")(
                process_obs_data, process_other_obs_data, alerts=True
            )
        elif type == "cowperthwaite_detections":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections in line with Cowperthwaite et. al 2018."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections in line with Cowperthwaite et. al 2018."
                        )
            detected_observations5 = getattr(afunc, "cowperthwaite_detections")(
                coadded_observations
            )
        elif type == "cowperthwaite_detections_no_coadd":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections in line with Cowperthwaite et. al 2018, but no coadds."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections in line with Cowperthwaite et. al 2018, but no coadds."
                        )
            detected_observations6 = getattr(afunc, "cowperthwaite_detections")(
                process_obs_data
            )
        elif type == "cowperthwaite_like_detections":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections like Cowperthwaite et. al 2018, but with alerts instead of SNR>5."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections like Cowperthwaite et. al 2018, but with alerts instead of SNR>5."
                        )
            detected_observations7 = getattr(afunc, "cowperthwaite_detections")(
                coadded_observations, alerts=True
            )
        elif type == "cowperthwaite_like_detections_no_coadd":
            if rank == 0 and verbose:
                print(
                    "Processing coadded observations for detections like Cowperthwaite et. al 2018, but with alerts instead of SNR>5 and no coadds."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations for detections like Cowperthwaite et. al 2018, but with alerts instead of SNR>5 and no coadds."
                        )
            detected_observations8 = getattr(afunc, "cowperthwaite_detections")(
                process_obs_data, alerts=True
            )
        else:
            if verbose and rank == 0:
                print(
                    "Processing coadded observations with the given filter dictionary."
                )
                if debug is True:
                    with open(debug_file, mode="a") as f:
                        f.write(
                            "\nProcessing coadded observations with the given filter dictionary."
                        )
            intermediate_filter = getattr(afunc, type)(intermediate_filter, filters)

    if rank == 0 and debug is True:
        with open(debug_file, mode="a") as f:
            f.write("\n")
            f.write("-------------Debug:-------------")
            f.write("Finished with detections and begin gather for output.")
    process_param_data = afunc.param_observe_detect(
        process_param_data, process_obs_data, detected_observations
    )
    # process_param_data = afunc.determine_ddf_transients(sim_inst, process_param_data)

    # Gather up all data to root
    coadded_observations.dropna(inplace=True)
    detected_observations.dropna(inplace=True)
    detected_observations2.dropna(inplace=True)
    detected_observations3.dropna(inplace=True)
    detected_observations4.dropna(inplace=True)
    detected_observations5.dropna(inplace=True)
    detected_observations6.dropna(inplace=True)
    detected_observations7.dropna(inplace=True)
    detected_observations8.dropna(inplace=True)
    process_param_data.dropna(inplace=True)

    coadd_receive = None
    params_receive = None
    detected_receive = None
    detected_receive2 = None
    detected_receive3 = None
    detected_receive4 = None
    detected_receive5 = None
    detected_receive6 = None
    detected_receive7 = None
    detected_receive8 = None
    # Join all batches and mpi workers and write the dataFrame to file
    if size > 1:
        if rank == 0 and debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\n")
                f.write("-------------Debug:-------------")
                f.write("Gather all the data to the root process.")
        coadd_receive = comm.gather(coadded_observations, root=0)
        params_receive = comm.gather(process_param_data, root=0)
        detected_receive = comm.gather(detected_observations, root=0)
        detected_receive2 = comm.gather(detected_observations2, root=0)
        detected_receive3 = comm.gather(detected_observations3, root=0)
        detected_receive4 = comm.gather(detected_observations4, root=0)
        detected_receive5 = comm.gather(detected_observations5, root=0)
        detected_receive6 = comm.gather(detected_observations6, root=0)
        detected_receive7 = comm.gather(detected_observations7, root=0)
        detected_receive8 = comm.gather(detected_observations8, root=0)

        if rank == 0:
            if debug is True:
                with open(debug_file, mode="a") as f:
                    f.write("\n")
                    f.write("-------------Debug:-------------")
                    f.write("Concatenating the dataframes for output.")
            output_params = pd.concat(params_receive, sort=False, ignore_index=True)
            output_coadd = pd.concat(coadd_receive, sort=False, ignore_index=True)
            output_detections = pd.concat(
                detected_receive, sort=False, ignore_index=True
            )
            output_detections2 = pd.concat(
                detected_receive2, sort=False, ignore_index=True
            )
            output_detections3 = pd.concat(
                detected_receive3, sort=False, ignore_index=True
            )
            output_detections4 = pd.concat(
                detected_receive4, sort=False, ignore_index=True
            )
            output_detections5 = pd.concat(
                detected_receive5, sort=False, ignore_index=True
            )
            output_detections6 = pd.concat(
                detected_receive6, sort=False, ignore_index=True
            )
            output_detections7 = pd.concat(
                detected_receive7, sort=False, ignore_index=True
            )
            output_detections8 = pd.concat(
                detected_receive8, sort=False, ignore_index=True
            )

    else:
        output_coadd = coadded_observations
        output_params = process_param_data
        output_detections = detected_observations
        output_detections2 = detected_observations2
        output_detections3 = detected_observations3
        output_detections4 = detected_observations4
        output_detections5 = detected_observations5
        output_detections6 = detected_observations6
        output_detections7 = detected_observations7
        output_detections8 = detected_observations8

    coadd_receive = None
    params_receive = None
    detected_receive = None
    detected_receive2 = None
    detected_receive3 = None
    detected_receive4 = None
    detected_receive5 = None
    detected_receive6 = None
    detected_receive7 = None
    detected_receive8 = None
    coadded_observations = None
    process_obs_data = None
    process_param_data = None
    process_other_obs_data = None
    detected_observations = None
    detected_observations2 = None
    detected_observations3 = None
    detected_observations4 = None
    detected_observations5 = None
    detected_observations6 = None
    detected_observations7 = None
    detected_observations8 = None

    if rank == 0:
        if verbose:
            print(
                "Outputting coadded observations, scolnic detections, parameters modified with observed, alerted, and detected flags, and the redshift distribution."
            )
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write(
                    "Outputting coadded observations, scolnic detections, parameters modified with observed, alerted, and detected flags, and the redshift distribution."
                )
        if save_all_output is True:
            output_coadd.to_csv(output_path + "coadded_observations.csv")
        output_detections.to_csv(output_path + "scolnic_detections.csv")
        output_detections2.to_csv(output_path + "scolnic_detections_no_coadd.csv")
        output_detections3.to_csv(output_path + "scolnic_like_detections.csv")
        output_detections4.to_csv(output_path + "scolnic_like_detections_no_coadd.csv")
        output_detections5.to_csv(output_path + "cowperthwaite_detections.csv")
        output_detections6.to_csv(output_path + "cowperthwaite_detections_no_coadd.csv")
        output_detections7.to_csv(output_path + "cowperthwaite_like_detections.csv")
        output_detections8.to_csv(
            output_path + "cowperthwaite_like_detections_no_coadd.csv"
        )
        output_params.to_csv(output_path + "modified_parameters.csv")
        if verbose:
            print("Done writing the detection results.")
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write("\nDone writing the detection results.")

    if size > 1:
        comm.barrier()

    if rank == 0 and verbose:
        t_end = time.time()
        print(
            "\n Estimated time for the last bit is: {}".format(
                (t_end - t1) / transient_dist.number_simulated
            )
        )
        print(
            "\nSimulation completed successfully with elapsed time: {}.".format(
                datetime.timedelta(seconds=int(t_end - t_start))
            )
        )
        if debug is True:
            with open(debug_file, mode="a") as f:
                f.write(
                    "\n Estimated time for the last bit is: {}".format(
                        (t_end - t1) / transient_dist.number_simulated
                    )
                )
                f.write(
                    "\nSimulation completed successfully with elapsed time: {}.".format(
                        datetime.timedelta(seconds=int(t_end - t_start))
                    )
                )
