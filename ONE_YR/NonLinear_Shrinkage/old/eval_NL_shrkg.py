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
from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf


#define fct for eval
def eval_nl_shrkg(rets_full, permnos, val_indices):
    rets = []
    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]

    for i in range(len(val_indices)):
        idx = val_indices[i]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]

        tmp1, tmp3, deltaQIS = estimators.QIS_for_RL(past_ret_mat)
        N, p = past_ret_mat.shape
        sample = pd.DataFrame(np.matmul(past_ret_mat.T.to_numpy(),past_ret_mat.to_numpy())) / (N-1)

        #eig decomp. of sample mat
        lambda1, u = np.linalg.eigh(sample)
        lambda1 = sorted(lambda1) # sorted in ascending order
        #deltaQIS = deltaQIS + np.mean(lambda1)

        tmp2 = np.diag(deltaQIS)
        sigmahat = pd.DataFrame(np.matmul(np.matmul(tmp1,tmp2),tmp3))
        weights = hf.calc_global_min_variance_pf(sigmahat)

        rets += list(fut_ret_mat @ weights)

    res = results = {
        "estimator pf return" : round(np.mean(rets) * 252 * 100, 2) ,
        "estimator pf sd" : round(np.std(rets) * np.sqrt(252) *100, 2) ,
    }
    return res

base_folder_path = r'/'
# IMPORT SHRK DATASETS
shrk_data_path = None
pf_size = 500
permnos = pd.read_pickle(
    fr"{base_folder_path}\1YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\1YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\1YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{base_folder_path}\1YR\preprocessing\training_dfs\PF{pf_size}\cov2Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

# get all the validation indices
len_train = 5040
end_date = fixed_shrk_data.shape[0]
# temp here
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]

res = eval_nl_shrkg(rets_full, permnos, val_indices_results)


print("done")