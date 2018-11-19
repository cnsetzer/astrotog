import os
import re
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LSSTmetrics.efficiencyTable import EfficiencyTable as eft

detect_function = eft.fromDES_EfficiencyFile(
    "/Users/cnsetzer/software/Cadence/LSSTmetrics/example_data/SEARCHEFF_PIPELINE_DES.DAT"
)

sim_results_folder = (
    "/Users/cnsetzer/Documents/LSST/astrotog_output/paper_results/seed0/"
)
plot_output = "/Users/cnsetzer/Documents/LSST/astrotog/paper/detectability/figures/"

info_columns = [
    "model",
    "cadence",
    "N_scolnic",
    "N_scolnic_nocd",
    "N_scolnic_like",
    "N_scolnic_like_nocd",
    "N_cowperthwaite",
    "N_cowperthwaite_nocd",
    "N_cowperthwaite_like",
    "N_cowperthwaite_like_nocd",
]

sim_info = pd.DataFrame(columns=info_columns)

for i, results_folder in enumerate(os.listdir(sim_results_folder)):
    obs_df = pd.read_csv(
        sim_results_folder + results_folder + "/observations.csv", index_col=0
    )
    param_df = pd.read_csv(
        sim_results_folder + results_folder + "/modified_parameters.csv", index_col=0
    )
    coadd_obs_df = pd.read_csv(
        sim_results_folder + results_folder + "/coadded_observations.csv", index_col=0
    )
    sd = pd.read_csv(
        sim_results_folder + results_folder + "/scolnic_detections.csv", index_col=0
    )
    sdnc = pd.read_csv(
        sim_results_folder + results_folder + "/scolnic_detections_no_coadd.csv",
        index_col=0,
    )
    sld = pd.read_csv(
        sim_results_folder + results_folder + "/scolnic_like_detections.csv",
        index_col=0,
    )
    sldnc = pd.read_csv(
        sim_results_folder + results_folder + "/scolnic_like_detections_no_coadd.csv",
        index_col=0,
    )
    cd = pd.read_csv(
        sim_results_folder + results_folder + "/cowperthwaite_detections.csv",
        index_col=0,
    )
    cdnc = pd.read_csv(
        sim_results_folder + results_folder + "/cowperthwaite_detections_no_coadd.csv",
        index_col=0,
    )
    cld = pd.read_csv(
        sim_results_folder + results_folder + "/cowperthwaite_like_detections.csv",
        index_col=0,
    )
    cldnc = pd.read_csv(
        sim_results_folder
        + results_folder
        + "/cowperthwaite_like_detections_no_coadd.csv",
        index_col=0,
    )

    folder_splits = results_folder.split("_")
    for split in folder_splits:
        if re.search("desgw", split):
            model = "DES-GW"
        elif re.search("nsbh", split):
            model = "SAEE-NSBH"
        elif re.search("nsns", split):
            model = "SAEE-NSNS"
        elif re.search("kraken2026", split):
            cadence = "opsim_baseline"
        elif re.search("kraken2042", split):
            cadence = "opsim_single_exp"
        elif re.search("nexus2097", split):
            cadence = "opsim_large_rolling_3yr"
        elif re.search("pontus2002", split):
            cadence = "opsim_large"
        elif re.search("pontus2489", split):
            cadence = "opsim_20s_exp"
        elif re.search("pontus2573", split):
            cadence = "slair_mixed_filter_pairs"
        elif re.search("alt", split):
            cadence = "alt_sched_rolling"

    plot_model_name = model.replace("-", "_")

    # Plot One
    if re.search("SAEE", model) and re.search("baseline", cadence):
        mej = param_df["m_ej"].values
        vej = param_df["v_ej"].values
        fig, ax = plt.subplots()
        plt.rc("text", usetex=True)
        # plt.rc('font', family='serif')
        ax.hexbin(mej, vej, gridsize=40, xscale="log", yscale="log")
        ax.xlabel(r"Median Ejecta Mass ($\mathrm{M}_{\odot}$)")
        ax.ylabel(r"Median Ejecta Velocity ($c$)")
        fig.savefig(
            plot_output + "{0}_{1}_hexbin_params.pdf".format(plot_model_name, cadence),
            bbox_inches="tight",
        )
        plt.close(fig)

    # Plot Two
    if re.search("baseline", cadence):
        prob_df = pd.DataFrame(columns=["Alert_Probability"])
        obs_df.join(prob_df)
        prob_mask = obs_df["signal_to_noise"] >= 50.0
        inv_mask = obs_df["signal_to_noise"] < 50.0
        obs_df.loc[prob_mask, "Alert_Probability"] = 1.0
        obs_df.loc[inv_mask, "Alert_Probability"] = detect_function.effSNR(
            obs_df.loc[inv_mask, "bandfilter"].values,
            obs_df.loc[inv_mask, "signal_to_noise"].values,
        )
        obs_df["max_snr"] = obs_df.groupby(["transient_id"])[
            "signal_to_noise"
        ].transform(max)
        obs_df["tot_alert_prob"] = obs_df.grouby(["transient_id"])[
            "Alert_Probability"
        ].transform(sum)
        max_snr = obs_df.grouby(["transient_id"])["max_snr"].values[0]
        tot_alert_prob = obs_df.grouby(["transient_id"])["tot_alert_prob"].values[0]
        fig, ax = plt.subplots()
        ax.hist(
            max_snr,
            bins=20,
            weights=tot_alert_prob,
            histtype="bar",
            log=True,
            color="k",
        )
        ax.ylabel("Alert Probability Weighted Counts")
        ax.xlabel("Maximum Signal to Noise")
        fig.savefig(
            plot_output
            + "{0}_{1}_alert_prob_hist.pdf".format(plot_model_name, cadence),
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot Three
        sd["Num_lc_all"] = sd.grouby(["transient_id"]).transform(len)
        sd['Num"lc_band'] = sd.grouby(["transient_id", "bandfilter"]).transform(len)
        sdnc["Num_lc_all"] = sdnc.grouby(["transient_id"]).transform(len)
        sdnc['Num"lc_band'] = sdnc.grouby(["transient_id", "bandfilter"]).transform(len)
        sld["Num_lc_all"] = sld.grouby(["transient_id"]).transform(len)
        sld['Num"lc_band'] = sld.grouby(["transient_id", "bandfilter"]).transform(len)
        sldnc["Num_lc_all"] = sldnc.grouby(["transient_id"]).transform(len)
        sldnc['Num"lc_band'] = sldnc.grouby(["transient_id", "bandfilter"]).transform(
            len
        )
        cd["Num_lc_all"] = cd.grouby(["transient_id"]).transform(len)
        cd['Num"lc_band'] = cd.grouby(["transient_id", "bandfilter"]).transform(len)
        cdnc["Num_lc_all"] = cdnc.grouby(["transient_id"]).transform(len)
        cdnc['Num"lc_band'] = cdnc.grouby(["transient_id", "bandfilter"]).transform(len)
        cld["Num_lc_all"] = cld.grouby(["transient_id"]).transform(len)
        cld['Num"lc_band'] = cld.grouby(["transient_id", "bandfilter"]).transform(len)
        cldnc["Num_lc_all"] = cldnc.grouby(["transient_id"]).transform(len)
        cldnc['Num"lc_band'] = cldnc.grouby(["transient_id", "bandfilter"]).transform(
            len
        )

        fig, ax = plt.subplots()

    # Save data for plots four, five, six, and seven
    sim_info.at[i, "model"] = model
    sim_info.at[i, "cadence"] = cadence
    sim_info.at[i, "N_scolnic"] = len(sd["transient_id"].unique())
    sim_info.at[i, "N_scolnic_nocd"] = len(sdnc["transient_id"].unique())
    sim_info.at[i, "N_scolnic_like"] = len(sld["transient_id"].unique())
    sim_info.at[i, "N_scolnic_like_nocd"] = len(sldnc["transient_id"].unique())
    sim_info.at[i, "N_cowperthwaite"] = len(cd["transient_id"].unique())
    sim_info.at[i, "N_cowperthwaite_nocd"] = len(cdnc["transient_id"].unique())
    sim_info.at[i, "N_cowperthwaite_like"] = len(cld["transient_id"].unique())
    sim_info.at[i, "N_cowperthwaite_like_nocd"] = len(cldnc["transient_id"].unique())

    if re.search("baseline", cadence) and re.search("DES-GW", model):
        # Plot eight redshift distribution
        z_min = 0.0
        z_max = 0.75
        bin_size = 0.025
        n_bins = int(round((z_max - z_min) / bin_size))
        all_zs = list(param_df["true_redshift"])
        detect_zs = list(
            param_df[param_df["transient_id"].isin(list(sd["transient_id"].unique()))][
                "true_redshift"
            ]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output + "{0}_{1}_redshift_dist_scolnic.pdf".format(plot_model_name)
        )
        plt.close(fig)

        detect_zs = list(
            param_df[
                param_df["transient_id"].isin(list(sdnc["transient_id"].unique()))
            ]["true_redshift"]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output
            + "{0}_{1}_redshift_dist_scolnic_no_coadd.pdf".format(plot_model_name)
        )
        plt.close(fig)

        detect_zs = list(
            param_df[param_df["transient_id"].isin(list(sld["transient_id"].unique()))][
                "true_redshift"
            ]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output
            + "{0}_{1}_redshift_dist_scolnic_like.pdf".format(plot_model_name)
        )
        plt.close(fig)

        detect_zs = list(
            param_df[
                param_df["transient_id"].isin(list(sldnc["transient_id"].unique()))
            ]["true_redshift"]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output
            + "{0}_{1}_redshift_dist_scolnic_like_no_coadd.pdf".format(plot_model_name)
        )
        plt.close(fig)

        detect_zs = list(
            param_df[param_df["transient_id"].isin(list(cd["transient_id"].unique()))][
                "true_redshift"
            ]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output
            + "{0}_{1}_redshift_dist_cowperthwaite.pdf".format(plot_model_name)
        )
        plt.close(fig)

        detect_zs = list(
            param_df[
                param_df["transient_id"].isin(list(cdnc["transient_id"].unique()))
            ]["true_redshift"]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output
            + "{0}_{1}_redshift_dist_cowperthwaite_no_coadd.pdf".format(plot_model_name)
        )
        plt.close(fig)

        detect_zs = list(
            param_df[param_df["transient_id"].isin(list(cld["transient_id"].unique()))][
                "true_redshift"
            ]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output
            + "{0}_{1}_redshift_dist_cowperthwaite_like.pdf".format(plot_model_name)
        )
        plt.close(fig)

        detect_zs = list(
            param_df[
                param_df["transient_id"].isin(list(cldnc["transient_id"].unique()))
            ]["true_redshift"]
        )
        # Create the histogram
        fig = plt.subplots()
        n, bins, patches = ax.hist(
            x=all_zs,
            bins=n_bins,
            range=(z_min, z_max),
            histtype="step",
            color="red",
            label="All Sources",
            linewidth=3.0,
        )
        ndetect, bins, patches = ax.hist(
            x=detect_zs,
            bins=bins,
            histtype="stepfilled",
            edgecolor="blue",
            color="blue",
            alpha=0.3,
            label="Detected Sources",
        )
        # plt.tick_params(which='both', length=10, width=1.5)
        ax.yscale("log")
        ax.legend(loc=2)
        ax.xlabel("z")
        ax.ylabel(r"$N(z)$")
        fig.savefig(
            plot_output
            + "{0}_{1}_redshift_dist_cowperthwaite_like_no_coadd.pdf".format(
                plot_model_name
            )
        )
        plt.close(fig)
