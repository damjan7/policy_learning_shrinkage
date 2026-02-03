import pandas as pd
import numpy as np
import pickle
import os

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6])

from collections import defaultdict

from helpers import eval_funcs_multi_target
from helpers import eval_funcs

base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
# IMPORT SHRK DATASETS
shrk_data_path = None
pf_size = 500
permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\cov2Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

# get all the validation indices
len_train = 5040
end_date = fixed_shrk_data.shape[0]
# temp here
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]

rets = []
reb_date_1 = permnos.index[0]
add_idx = np.where(rets_full.index == reb_date_1)[0][0]

from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf

CVC_shrinkages = []

# define factors
all_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
               1.9, 2.0]

reb_date_indices = [21*i for i in range(10353//21)]

all_res = defaultdict(list)
all_rawres = defaultdict(list)

# the evs are before preserving the trace (see QIS code)
sample_eigenvalues = []
CVC_shrinkages = []

if 1 == 1:
    tmp_res = defaultdict(list)
    tmp_rawres = defaultdict(list)
    for idx in range(permnos.shape[0]):
        try:
            past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
            past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
            past_ret_mat = past_ret_mat.fillna(0)
            fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
        except:
            print("f2")

        shrinkage, sample, target = estimators.cov2Para(past_ret_mat)

        for factor in all_factors:
            if factor == 1:
                shrk = shrinkage
            else:
                shrk = shrinkage * factor
                shrk = min(shrk, 1.0)

            sigmahat = shrk*target+(1-shrk)*sample
            try:
                weights = hf.calc_global_min_variance_pf(sigmahat)
            except:
                print("f")
            # store results
            tmp_res[factor].append(np.std(fut_ret_mat @ weights) * np.sqrt(252) * 100)
            if idx % 21 == 0:
                tmp_rawres[factor] += list(fut_ret_mat @ weights)
    all_res = tmp_res
    all_rawres= tmp_rawres

print("done")
########## DONE

print("done and save")
path = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\CVC_shrinkages_grid_data"
df = pd.DataFrame(all_res)
df.to_csv(path + f"\\CVC_grid_allres_evs.csv", index=False)
df = pd.DataFrame(all_rawres)
df.to_csv(path + f"\\CVC_grid_all_rawres_evs.csv", index=False)


# for sample and qis eigenvalues
for i in range(permnos.shape[0]):
    idx = i
    past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
    past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
    past_ret_mat = past_ret_mat.fillna(0)
    delta, lambda1 = estimators.QIS_get_eigenvalues(past_ret_mat)
    shrinkage, sample, target = estimators.cov2Para(past_ret_mat)
    qis_deltas.append(delta)
    sample_eigenvalues.append(lambda1)
    CVC_shrinkages.append(shrinkage)

    # then calculate pf sd of each pf for next 21 days and store it for cvc shrinkage and
    # for factors in 0.5 and 2, capped at 2 if


