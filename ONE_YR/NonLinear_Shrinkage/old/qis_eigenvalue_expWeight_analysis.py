import pandas as pd
import numpy as np
import pickle
import os

from collections import defaultdict

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6])
p.cpu_affinity([10,11,12,13,14])


from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs


base_folder_path = r'/'
# IMPORT SHRK DATASETS
pf_size = 100
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
reb_date_1 = permnos.index[0]
add_idx = np.where(rets_full.index == reb_date_1)[0][0]


## Actual QIS Eigenvalues used by model; apply factor to these
qis_shrkges_path = r"/ONE_YR/preprocessing/QIS_shrinkages.csv"
qis_shrkges_v2 = pd.read_csv(qis_shrkges_path)
qis_deltas_full = qis_shrkges_v2.copy()

# define factors
all_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
w_05 = [sigmoid(i/100) for i in range(500)][::-1]
w_2 = [1/sigmoid(i/100) for i in range(500)][::-1]

all_factors = [0.5, 1.0, 2.0]
factor_weights = defaultdict(list)
factor_weights[0.5] = w_05
factor_weights[2.0] = w_2

reb_date_indices = [21*i for i in range(10353//21)]

all_res = defaultdict(list)
all_rawres = defaultdict(list)

for num_ev in [1]:
    tmp_res = defaultdict(list)
    tmp_rawres = defaultdict(list)
    for idx in range(0, permnos.shape[0]):
        try:
            past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
            past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
            past_ret_mat = past_ret_mat.fillna(0)
            fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
        except:
            print("Some Error..")
        N, p = past_ret_mat.shape
        sample = pd.DataFrame(np.matmul(past_ret_mat.T.to_numpy(), past_ret_mat.to_numpy())) / (N - 1)
        lambda1, u = np.linalg.eigh(sample)
        lambda1 = lambda1.real.clip(min=0)
        dfu = pd.DataFrame(u,columns=lambda1)
        dfu.sort_index(axis=1,inplace = True)
        temp1 = dfu.to_numpy()
        temp3 = dfu.T.to_numpy().conjugate()

        qis_base = qis_deltas_full.iloc[idx, :].copy()
        for factor in all_factors:
            if factor == 1:
                qis = qis_base.copy()
            else:
                qis = qis_base.copy()
                qis = qis * factor_weights[factor]

            temp2 = np.diag(qis)
            sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
            try:
                weights = hf.calc_global_min_variance_pf(sigmahat)
            except:
                print("Some Other Error..")
            # store results
            tmp_res[factor].append(np.std(fut_ret_mat @ weights, ddof=1) * np.sqrt(252) * 100)
            if idx % 21 == 0:
                tmp_rawres[factor] += list(fut_ret_mat @ weights)
    all_res[num_ev] = tmp_res
    all_rawres[num_ev] = tmp_rawres

print("done")
########## DONE

path= rf"/ONE_YR/NonLinear_Shrinkage/transformed_qis_eigenvalues"
qis_exp_05.to_csv(path + f"\\qis_exp_05.csv", index=False)
qis_exp_2.to_csv(path + f"\\qis_exp_2.csv", index=False)


# path = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\qis_eigenvalue_grid_data"
for num_ev in num_eigenvalues:
    df = pd.DataFrame(all_res[num_ev])
    df.to_csv(path + f"\\qis_grid_allres_{num_ev}_evs.csv", index=False)
    df = pd.DataFrame(all_rawres[num_ev])
    df.to_csv(path + f"\\qis_grid_all_rawres_{num_ev}_evs.csv", index=False)



qis_results_for_eval = pd.DataFrame([raw_res1, raw_res2, raw_res3]).T
qis_results_for_eval.columns = ['base', 'upper', 'lower']
path = r"/ONE_YR/preprocessing/qis_results_for_eval.csv"
qis_results_for_eval.to_csv(path, index=False)

qis_results_full = pd.DataFrame([res1, res2, res3]).T
qis_results_full.columns = ['base', 'upper', 'lower']
path = r"/ONE_YR/preprocessing/qis_results_full.csv"
qis_results_full.to_csv(path, index=False)


###### NEW DATA (quotient of new and old eigenvalues instead of just qis eigenvalues)
qis_results_for_eval_v2 = pd.DataFrame([raw_res1, raw_res2, raw_res3, raw_res4, raw_res5, raw_res6, raw_res7]).T
qis_results_for_eval_v2.columns = ['base', '0.7', '0.8', '0.9', '1.1', '1.2', '1.3']
path = r"/ONE_YR/preprocessing/qis_results_for_eval_v3.csv"
qis_results_for_eval_v2.to_csv(path, index=False)


qis_results_full_v2 = pd.DataFrame([res1, res2, res3, res4, res5, res6, res7]).T
qis_results_full_v2.columns = ['base', '0.7', '0.8', '0.9', '1.1', '1.2', '1.3']
path = r"/ONE_YR/preprocessing/qis_results_full_v3.csv"
qis_results_full_v2.to_csv(path, index=False)

