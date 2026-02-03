import pandas as pd
import plotly.express as px
import numpy as np
import os
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import pickle


import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([1,2,3,4, 5,6,7,8,9,10,11,12])

os.chdir(r'H:\all\RL_Shrinkage_2024')
from helpers import helper_functions as hf
from ONE_YR.NonLinear_Shrinkage import regression_evaluation_funcs as re_hf
from helpers import eval_function_new


returns = {}
cov1para_shrinkages = {}
for tmp_pf_size in [30, 50, 100, 225, 500]:
    with open(fr"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\training_dfs\PF{tmp_pf_size}\cov1Para_factor-1.0_p{tmp_pf_size}.pickle", 'rb') as f:
        tmp = pickle.load(f)
    cov1para_shrinkages[tmp_pf_size] = tmp['shrk_factor']
    returns[tmp_pf_size] = tmp['pf_return']

dates = pd.to_datetime(tmp['date'], format="%Y%m%d")
returns  = pd.DataFrame(returns, dtype=np.float64)
returns.index = dates.values

cov1para_shrinkages = pd.DataFrame(cov1para_shrinkages, dtype=np.float64)
cov1para_shrinkages.index = dates.values
cov1para_shrinkages = cov1para_shrinkages.round(3)

# save them in results
tmp_res_path = r"H:\all\RL_Shrinkage_2024\ONE_YR\Linear_Shrinkage\results"
cov1para_shrinkages.to_csv(tmp_res_path + "\\cov1para_intensities.csv")

from helpers import rl_covmat_ests_for_dataset as estimators
def eval_funcs_get_retuns(val_preds, rets_full, permnos, reb_days, val_indices):
    weighted_rets_model = []
    weights = {}
    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]

    for i in range(len(val_indices)):
        idx = val_indices[i]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]

        shrk, sample, target = estimators.cov2Para(past_ret_mat)
        model_covmat_est = val_preds[i] * target + (1-val_preds[i]) * sample

        weights_model = hf.calc_global_min_variance_pf(model_covmat_est)
        weighted_rets_model += list(fut_ret_mat @ weights_model)

        weights[permnos.index[idx]] = weights_model

    return weights, weighted_rets_model


base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
# IMPORT SHRK DATASETS
pf_size = 500
permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")


fixed_shrk_name = 'cov1Para'
opt_shrk_name = 'cov1Para'
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_{fixed_shrk_name}_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\{opt_shrk_name}_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_rawres_{fixed_shrk_name}_p{pf_size}.pickle", 'rb') as f:
    rawres_fixed_shrk_data = pickle.load(f)

# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = fr"{base_folder_path}\helpers"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too
start_date = str(optimal_shrk_data['date'].iloc[0])
start_date = start_date[0:4] + '-' + start_date[4:6] + "-" + start_date[6:]
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]

cov1para_shrk = optimal_shrk_data['shrk_factor'].values[5040:]

len_train = 5040
end_date = fixed_shrk_data.shape[0]
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]


weights, daily_rets = eval_funcs_get_retuns(cov1para_shrinkages[pf_size].values[5040:], rets_full, permnos, 0, val_indices_results)

out_path = fr"H:\all\RL_Shrinkage_2024\ONE_YR\Linear_Shrinkage\results\p{pf_size}"

dates = pd.to_datetime(permnos.index, format="%Y%m%d")
weights = pd.DataFrame(pd.DataFrame(weights))
weights.to_csv(out_path + f"\c1p_p{pf_size}_weights.csv")

returns.index = dates
optimal_shrk_data.index = dates
returns.to_csv(out_path + f"\c1p_p{pf_size}_daily_returns.csv")
optimal_shrk_data['shrk_factor'].to_csv(out_path + f"\c1p_p{pf_size}_intensities.csv")


print("done")


# get OOS weights for the RL-Linear Model

for PF_SIZE in [30, 50, 100, 225, 500]:
    in_path = rf"H:\all\RL_Shrinkage_2024\ONE_YR\Linear_Shrinkage\results\p{PF_SIZE}"
    out_path = in_path
    # load intensities
    oos_intensities = pd.read_csv(in_path + f"\oos_linear_model_intensity_p{PF_SIZE}.csv",index_col=0)
    #intensities = [intensities.iloc[i][0] for i in range(0, intensities.shape[0], 21)]
    weights, daily_rets = eval_funcs_get_retuns(oos_intensities.iloc[:,0].values, rets_full, permnos, 0, val_indices_results)
    weights = pd.DataFrame(weights)
    weights.to_csv(out_path + rf"\oos_linear_model_weights_p{PF_SIZE}.csv")
