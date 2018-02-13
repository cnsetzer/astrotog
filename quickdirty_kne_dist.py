import os
import re
import sncosmo
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
        pkappa.append(seds_data[key]['kappa'])
        pm_ej.append(seds_data[key]['m_ej'])
        pv_ej.append(seds_data[key]['v_ej'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pkappa, pm_ej, pv_ej)
ax.set_xlabel('Kappa')
ax.set_ylabel('M_ej')
ax.set_zlabel('V_ej')
plt.show()

# Build parameter grid that hosts the read-in SEDs
