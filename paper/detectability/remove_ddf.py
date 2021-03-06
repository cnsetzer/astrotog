import os
import re
import pandas as pd
import opsimsummary as oss

results_folders = '/share/data1/csetzer/lsst_kne_sims_outputs/'
cad_path = '/share/data1/csetzer/lsst_cadences/'
sims = os.listdir(results_folders)

for sim in sims:
    sim_outs = os.listdir(results_folders+sim+'/')
    done = 0
    for prod in sim_outs:
        if re.search(r'wfd|ddf', prod):
            done += 1
    if done == 16:
        continue
    else:
        print('\nProcessing wfd/ddf separation for {}'.format(sim))
        if re.search('kraken2026',sim):
            ddf_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'kraken_2026.db',
                subset='ddf',
                opsimversion='lsstv4',
            ).summary
            wfd_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'kraken_2026.db',
                subset='wfd',
                opsimversion='lsstv4',
            ).summary
        if re.search('alt',sim):
            ddf_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'alt_sched_rolling.db',
                subset='ddf',
                opsimversion='sstf',
                filterNull=True,
            ).summary
            wfd_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'alt_sched_rolling.db',
                subset='wfd',
                opsimversion='sstf',
                filterNull=True,
            ).summary
        if re.search('kraken2042',sim):
            ddf_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'kraken_2042.db',
                subset='ddf',
                opsimversion='lsstv4',
            ).summary
            wfd_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'kraken_2042.db',
                subset='wfd',
                opsimversion='lsstv4',
            ).summary
        if re.search('nexus',sim):
            ddf_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'nexus_2097.db',
                subset='ddf',
                opsimversion='lsstv4',
            ).summary
            wfd_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'nexus_2097.db',
                subset='wfd',
                opsimversion='lsstv4',
            ).summary
        if re.search('pontus2002',sim):
            ddf_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'pontus_2002.db',
                subset='ddf',
                opsimversion='lsstv4',
            ).summary
            wfd_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'pontus_2002.db',
                subset='wfd',
                opsimversion='lsstv4',
            ).summary
        if re.search('pontus2489',sim):
            ddf_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'pontus_2489.db',
                subset='ddf',
                opsimversion='lsstv4',
            ).summary
            wfd_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'pontus_2489.db',
                subset='wfd',
                opsimversion='lsstv4',
            ).summary
        if re.search('pontus2573',sim):
            ddf_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'pontus_2573.db',
                subset='ddf',
                opsimversion='lsstv4',
            ).summary
            wfd_obs = oss.OpSimOutput.fromOpSimDB(
                cad_path + 'pontus_2573.db',
                subset='wfd',
                opsimversion='lsstv4',
            ).summary

        for prod in sim_outs:
            ddf_ind = []
            wfd_ind = []
            if re.search(r'wfd|ddf', prod):
                continue
            if ('wfd_'+prod in sim_outs) and ('ddf_'+prod in sim_outs):
                continue
            if re.search('detections', prod):
                print('\nDoing separation for {}'.format(prod))
                file = pd.read_csv(results_folders+sim+'/'+prod, index_col=0)
                if 'mjd' in file.columns:
                    for index, series in file.iterrows():
                        obsmjd = series['mjd']
                        qry = ddf_obs.query('{0} - 0.000115 <= expMJD <= {0} + 0.000115'.format(obsmjd))
                        qry2 = wfd_obs.query('{0} - 0.000115 <= expMJD <= {0} + 0.000115'.format(obsmjd))
                        if qry.empty:
                            pass
                        else:
                            ddf_ind.append(index)
                        if qry2.empty:
                            pass
                        else:
                            wfd_ind.append(index)
                else:
                    continue
            else:
                continue

            wfd_ver = file[file.index.isin(wfd_ind)]
            ddf_ver = file[file.index.isin(ddf_ind)]

            wfd_ver.to_csv(results_folders+sim+'/wfd_'+prod)
            ddf_ver.to_csv(results_folders+sim+'/ddf_'+prod)
