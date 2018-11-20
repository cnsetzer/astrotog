import os
import re
import pandas as pd

results_path = "/Users/cnsetzer/Documents/LSST/astrotog_output/paper_results/seed0/"

sims_folders = os.listdir(results_path)

for folder in sims_folders:
    path_to_files = results_path + folder + "/"
    current_files = os.listdir(path_to_files)
    # find number of ranks
    rank_nums = []
    files_to_join = []
    for file in current_files:
        m = re.search(r'rank\d+')
        if m:
            rank_nums.append(float(m.group(0).replace('rank', '')))
            base_name = re.sub(r'_rank\d+.csv', '', file)
            if base_name in files_to_join:
                pass
            else:
                files_to_join.append(base_name)
        else:
            continue

    max_rank = max(rank_nums)

    for file in files_to_join:
        print('\nJoining outputs for {}'.format(file))
        rank_list = []
        for i in range(max_rank+1):
            df = pd.read_csv(path_to_files + file + '_rank{0}.csv'.format(i), index_col=0)
            rank_list.append(df)
        joined_df = pd.concat(rank_list, sort=False, ignore_index=True)
        joined_df.to_csv(path_to_files + file + '.csv')
        joined_df = None
