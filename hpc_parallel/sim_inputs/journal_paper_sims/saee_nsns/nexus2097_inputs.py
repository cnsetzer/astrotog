import datetime
import numpy as np

# ----------------------------------------------------------------------
# Section that user can edit to tailor simulation
# ----------------------------------------------------------------------
save_all_output = False
batch_mp_workers = 2
verbose = True
batch_size = 100  # can also be set to 'all'
dithers = True
desc_dithers = True
add_dithers = False
cadence_has_nulls = False
same_dist = True
min_dec = np.deg2rad(-85.0)
max_dec = np.deg2rad(25.0)
transient_duration = (
    50.0
)  # in days used to select time before survey to begin injecting transients
t_before = 21.0
t_after = 21.0
z_max = 0.75  # Maximum redshift depth for simulation
z_bin_size = 0.02  # Binning for redshift distribution histogram
z_min = 0.0  # Given if you want to simulate shells
rate = 1000  # Rate in events per GPC^3 per restframe time
instrument_class_name = "lsst"
survey_version = "lsstv4"
cadence_flags = "combined"  # Currently use default in class
transient_model_name = "saee_nsns"
detect_type = [
    "scolnic_detections",
    "scolnic_like_detections",
    "scolnic_detections_no_coadd",
    "scolnic_like_detections_no_coadd",
]  # ['detect'], ['scolnic_detections'], or multiple
seds_path = None
cadence_path = "/share/data1/csetzer/lsst_cadences/nexus_2097.db"
dither_path = "/share/data1/csetzer/lsst_cadences/descDithers_nexus_2097.csv"
cadence_ra_col = "_ra"
cadence_dec_col = "_dec"
throughputs_path = "/share/data1/csetzer/lsst/throughputs/lsst"
reference_flux_path = "/share/data1/csetzer/lsst/throughputs/references"
efficiency_table_path = (
    "/home/csetzer/software/Cadence/LSSTmetrics/example_data/SEARCHEFF_PIPELINE_DES.DAT"
)
run_dir = "lsst_saee_nsns_nexus2097_" + datetime.datetime.now().strftime(
    "%d%m%y_%H%M%S"
)
output_path = "/share/data1/csetzer/lsst_kne_sims_outputs/" + run_dir + "/"

# Define filters for detections
filters = {
    "snr": {
        "type": "value",
        "num_count": None,
        "name": "signal_to_noise",
        "value": 0.001,
        "gt_lt_eq": "gt",
        "absolute": True,
    }
    # 'snr': {'type': 'value',
    #         'num_count': None,
    #         'name': 'signal_to_noise',
    #         'value': 5.0,
    #         'gt_lt_eq': 'gt',
    #         'absolute': False}
}
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
