import os
import re
import pandas as pd
import astrotog.functions as afunc

reproc_path = '/share/data1/csetzer/lsst_kne_sims_outputs/'
detect_files = os.listdir(reproc_path)

for file in detect_files:
    type = re.sub(".csv", "", file)

    if type == "wfd_scolnic_detections":
        print(
            "Processing coadded observations for detections in line with Scolnic et. al 2018."
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
