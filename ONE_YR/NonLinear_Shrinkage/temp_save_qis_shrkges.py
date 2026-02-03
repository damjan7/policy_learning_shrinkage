import pandas as pd
import numpy as np
import pickle
import os

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6])

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

QIS_shrkges = []

# the evs are before preserving the trace (see QIS code)
sample_eigenvalues = []
qis_deltas = []
QIS_shrkges = []

# for sample and qis eigenvalues
for i in range(permnos.shape[0]):
    idx = i
    past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
    past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
    past_ret_mat = past_ret_mat.fillna(0)
    delta, lambda1 = estimators.QIS_get_eigenvalues(past_ret_mat)
    tmp1, tmp3, deltaQIS = estimators.QIS_for_RL(past_ret_mat)
    qis_deltas.append(delta)
    sample_eigenvalues.append(lambda1)
    QIS_shrkges.append(deltaQIS)


for i in range(permnos.shape[0]):
    idx = i
    past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
    past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
    past_ret_mat = past_ret_mat.fillna(0)
    fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]

    tmp1, tmp3, deltaQIS = estimators.QIS_for_RL(past_ret_mat)
    estimators.cov2Para(past_ret_mat)
    QIS_shrkges.append(deltaQIS)





print("done and save")
path = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\QIS_shrinkages.csv"
QIS_shrkges_df = pd.DataFrame(QIS_shrkges)
QIS_shrkges_df.to_csv(path, index=False)

# save version 2
path1 = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\QIS_deltas.csv"
path2 = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\sample_eigenvalues.csv"
df1 = pd.DataFrame(qis_deltas)
df2 = pd.DataFrame(sample_eigenvalues)
df1.to_csv(path1, index=False)
df2.to_csv(path2, index=False)
