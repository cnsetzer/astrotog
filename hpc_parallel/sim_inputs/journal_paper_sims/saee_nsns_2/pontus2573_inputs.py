import datetime
import numpy as np

# ----------------------------------------------------------------------
# Section that user can edit to tailor simulation
# ----------------------------------------------------------------------
debug = False
debug_file = "/home/csetzer/LSST/astrotog_output/debug_log_saeensns_pontus2573_3.txt"
save_all_output = True
batch_mp_workers = 2
verbose = True
batch_size = 200  # can also be set to 'all'
dithers = False  # external dithers
desc_dithers = False  # are these external dithers DESC format dithers
add_dithers = False
cadence_has_nulls = False
same_dist = (
    True
)  # Flag to set largest distribution so that this is equal over all cadences
min_dec = np.deg2rad(-90.0)
max_dec = np.deg2rad(35.0)
transient_duration = (
    30.0
)  # in days used to select time before survey to begin injecting transients
t_before = 21.0
t_after = 21.0
z_max = 0.5  # Maximum redshift depth for simulation
z_bin_size = 0.02  # Binning for redshift distribution histogram
z_min = 0.0  # Given if you want to simulate shells
rate = 1000  # Rate in events per GPC^3 per restframe time
instrument_class_name = "lsst"
survey_version = "lsstv4"
cadence_flags = "wfd"  # Currently use default in class
transient_model_name = "saee_nsns"
detect_type = [
    "scolnic_detections",
    "scolnic_like_detections",
    "scolnic_detections_no_coadd",
    "scolnic_like_detections_no_coadd",
    "cowperthwaite_detections",
    "cowperthwaite_like_detections",
    "cowperthwaite_detections_no_coadd",
    "cowperthwaite_like_detections_no_coadd",
]  # ['detect'], ['scolnic_detections'], or multiple
seds_path = None
cadence_path = "/share/data1/csetzer/lsst_cadences/pontus_2573.db"
dither_path = None
cadence_ra_col = "_ra"
cadence_dec_col = "_dec"
throughputs_path = "/share/data1/csetzer/lsst/throughputs/lsst"
reference_flux_path = "/share/data1/csetzer/lsst/throughputs/references"
efficiency_table_path = (
    "/home/csetzer/software/Cadence/LSSTmetrics/example_data/SEARCHEFF_PIPELINE_DES.DAT"
)
run_dir = "lsst_saee_nsns_pontus2573_" + datetime.datetime.now().strftime(
    "%d%m%y_%H%M%S"
)
output_path = "/share/data1/csetzer/lsst_kne_sims_outputs/" + run_dir + "/"

# Define filters for detections
# filters = {
#     "snr": {
#         "type": "value",
#         "num_count": None,
#         "name": "signal_to_noise",
#         "value": 0.001,
#         "gt_lt_eq": "gt",
#         "absolute": True,
#     }
    # 'snr': {'type': 'value',
    #         'num_count': None,
    #         'name': 'signal_to_noise',
    #         'value': 5.0,
    #         'gt_lt_eq': 'gt',
    #         'absolute': False}
# }
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------