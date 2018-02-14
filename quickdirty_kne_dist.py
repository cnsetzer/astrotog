import os
import re
import sncosmo
import numpy as np
from scipy.interpolate import neare
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

seds_data = {}
key_list = []
# Define directory for location of SEDS
seds_path = "./sedb/rosswog/"
# Get the list of SED files
fl = os.listdir(seds_path)
# Read in all  SEDS
for filei in fl:
    filename = seds_path + filei
    fileio = open(filename, 'r')
    # Initialize dicts for sedsdb
    seds_key = filei.strip(".dat")
    key_list.append(seds_key)
    seds_data[seds_key] = {}
    # Read header for parameter data for model (Specific for Rosswog)
    for headline in fileio:
        if headline.strip().startswith("#"):
            if re.search("kappa =", headline):
                kappa = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
            elif re.search("m_ej = |m_w =", headline):
                m_ej = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
            elif re.search("v_ej = |v_w =", headline):
                v_ej = float(re.search(r"\s\d+\.*\d*\s", headline).group(0))
            else:
                continue
        else:
            fileio.close()
            break
# Debug Print of seds_key to find problematic sed
    # print(seds_key)

    # Read in SEDS data with sncosmo tools
    phase, wave, flux = sncosmo.read_griddata_ascii(filename)
    source = sncosmo.TimeSeriesSource(phase, wave, flux, zero_before=True)
    model = sncosmo.Model(source=source)
    # Construct the full sed db
    seds_data[seds_key]['model'] = model
    seds_data[seds_key]['kappa'] = kappa
    seds_data[seds_key]['m_ej'] = m_ej
    seds_data[seds_key]['v_ej'] = v_ej

# Test import with plots of lc
    # sncosmo.plot_lc(model=model, bands=['lsstr'])
    # plt.show()

# Test Grid with Scatter plot in the 3dimensions
pkappa, pm_ej, pv_ej = [], [], []
for key in key_list:
        pkappa.append(int(seds_data[key]['kappa']))
        pm_ej.append(seds_data[key]['m_ej'])
        pv_ej.append(seds_data[key]['v_ej'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pkappa, pm_ej, pv_ej)
ax.set_xlabel('Kappa')
ax.set_ylabel('M_ej')
ax.set_zlabel('V_ej')
plt.show()

# Slice the parameters space based on Kappa
pm_ej_k1, pm_ej_k10, pm_ej_k100 = [], [], []
pv_ej_k1, pv_ej_k10, pv_ej_k100 = [], [], []
for i in np.arange(len(key_list)):
    if pkappa[i] == 1:
        pv_ej_k1[i] = pv_ej[i]
        pm_ej_k1[i] = pm_ej[i]
    elif pkappa[i] == 10:
        pv_ej_k10[i] = pv_ej[i]
        pm_ej_k10[i] = pm_ej[i]
    else:
        pv_ej_k100[i] = pv_ej[i]
        pm_ej_k100[i] = pm_ej[i]

# Build parameter grid that hosts the read-in SEDs


# Define Priors from which to draw the KNe distribution
kappa = [min(pkappa), 10]
v_ej = [min(pv_ej), max(pv_ej)]
m_ej = [min(pm_ej), max(pm_ej)]

# Set number of KNe to generate
ndist = 100

for i in np.arange(ndsit):
    # First we need to draw from the prior parameters
    # Sample Binomial distribution for Kappa
    KNe_tmp_kappa = kappa[int(np.random.binomial(1, 0.5, size=None))]
    # Sample ejecta/wind velocity and ejecta/wind mass from flat prior
    KNe_tmp_v_ej = np.random.uniform(low=v_ej[0], high=v_ej[1])
    KNe_tmp_m_ej = np.random.uniform(low=m_ej[0], high=m_ej[1])

    # Interpolation on 2D (v_ej,m_ej) space separate for each Kappa
    if KNe_tmp_kappa == kappa[0]:
        # Find nearest neighbors in this kappa-slice grid

    else:
        # Find nearest neighbors in this kappa-slice grid

wave
