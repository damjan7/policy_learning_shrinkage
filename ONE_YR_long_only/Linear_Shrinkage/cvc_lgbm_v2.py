import pandas as pd
import numpy as np
import pickle
import os
from collections import  Counter
from collections import defaultdict
import matplotlib.pyplot as plt

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6,7,8,9])

from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs
from helpers import eval_function_new

import helpers_linear_shrinkage as hf_ls
from ONE_YR.NonLinear_Shrinkage import regression_evaluation_funcs as re_hf
from sklearn.preprocessing import StandardScaler, Normalizer


base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
# IMPORT SHRK DATASETS
pf_size = 500
permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")


fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
    # fixed shrk data contains the 21 day lead pf sds for portfolios built
    # using the covariance matrix obtained by shrinkage intensity x and CVC target
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\cov2Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

# load min signals and rawres
cvc_grid_data = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\cvc_shrinkages_grid_data"
all_res = pd.read_csv(cvc_grid_data + f"\\CVC_grid_allres.csv")
all_rawres = pd.read_csv(cvc_grid_data + f"\\CVC_grid_all_rawres.csv")

allres_min_idxes_full = all_res.idxmin(axis=1)[: -21].values
allres_min_idxes_full = np.insert(allres_min_idxes_full, 0, np.repeat(["1.0"], 21))

# for sanity check: BIASED version should generally be better than
# non biased version as it is literally the minimum over the future 21 days
# so using it as a signal should outperform
allres_min_idxes_BIASED = all_res.idxmin(axis=1).values
allres_min_idxes_BIASED = allres_min_idxes_BIASED


# simple argmin rule, with full allres_min, should be same results as above
allres_min_idxes_full_v2 = allres_min_idxes_full[list(range(0, allres_min_idxes_full.shape[0], 21))]
allres_min_idxes_full_v2 = np.repeat(allres_min_idxes_full_v2, 21)
res_simple_argmin_rule = np.diag(all_rawres.loc[:, allres_min_idxes_full_v2])[5040:]


# simple argmin rule, biased (as a sanity check)
allres_min_idxes_BIASED_v2 = allres_min_idxes_BIASED[list(range(0, allres_min_idxes_BIASED.shape[0], 21))]
allres_min_idxes_BIASED_v2 = np.repeat(allres_min_idxes_BIASED_v2, 21)
res_simple_argmin_rule_biased = np.diag(all_rawres.loc[:, allres_min_idxes_BIASED_v2])[5040:]

res_actual_argmin = []
for i in range(5313//21):
    tmp_data = all_rawres.iloc[5040 + 21*i: 5040 + 21*(i+1)]
    curmin_idx = tmp_data.std().idxmin()
    curmin = tmp_data.loc[:, curmin_idx]
    res_actual_argmin += curmin.tolist()
# np.std(res_actual_argmin) * np.sqrt(252) * 100  --> 10.375

res_actual_argmin_nonbiased = []
for i in range(5313//21):
    idx_min_data = all_rawres.iloc[5040 - 21 + 21*i: 5040 - 21 + 21*(i+1)]
    curmin_idx = idx_min_data.std().idxmin()
    tmp_data = all_rawres.iloc[5040 + 21*i: 5040 + 21*(i+1)]
    curmin = tmp_data.loc[:, curmin_idx]
    res_actual_argmin_nonbiased += curmin.tolist()
#np.std(res_actual_argmin_nonbiased) * np.sqrt(252) * 100  --> 10.65

#min_idxes = fixed_shrk_data.iloc[:, 3:].idxmin(axis=1)
#opt_vals = np.diag(fixed_shrk_data.iloc[:, 3:].loc[:, min_idxes])

# get all the validation indices
len_train = 5040
end_date = fixed_shrk_data.shape[0]
# temp here
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
reb_date_1 = permnos.index[0]
add_idx = np.where(rets_full.index == reb_date_1)[0][0]


lgbm_params = {'boosting_type': 'gbdt',
 'learning_rate': 0.05,
 'n_estimators': 200,
 'num_leaves': 31,
 'reg_lambda': 5}

all_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
               1.9, 2.0]
# num eigenvalues to modify
num_eigenvalues = [1, 5, 10, 25, 50]



model_preds_ALL = {}
Y = allres_min_idxes_BIASED.astype(float)
opt_values = allres_min_idxes_BIASED.astype(float)[:-21]
opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
Y = np.array(re_hf.map_factors_to_preds(Y.reshape(-1), all_factors))
opt_values = np.array(re_hf.map_factors_to_preds(opt_values, all_factors))
opt_v3 = np.diag(all_res.loc[:, allres_min_idxes_BIASED])[:-21]
opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))

# opt_v3 and opt_values is then overwritten within params during training of corresp. num_ev
params = {
    'pf_size' : [pf_size],
    'opt_values_factors' : [opt_values],
    'include_ew_month_vola' : (True, False),
    'include_ew_year_vola' : (True, False),
    'include_sample_covmat_trace' : (True, False),
    'include_allstocks_year_avgvola' : (True, False),
    'include_allstocks_month_avgvola' : (True, False),
    'include_factors' : (True, False),
    'include_ts_momentum_allstocks' : (True, False),
    'include_ts_momentum_var_allstocks' : (True, False),
    'include_ewma_year' : (True, False),
    'include_ewma_month' : (True, False),
    'include_mean_of_correls' : (True, False),
    'include_iqr' : (True, False),
    'additional_inputs' : [opt_v3]
}

from sklearn.model_selection import ParameterSampler
param_combs = list(ParameterSampler(params, 50))

p1 = {'boosting_type': 'gbdt',
      'learning_rate': 0.1,
      'n_estimators': 500,
      'num_leaves': 62,
      'reg_lambda': 5}

RES_ALL = defaultdict(list)
mapped_res_ALL = defaultdict(list)


SCALE = False
for lgbm_param in [p1]:
    curparam = param_combs[40]
    curparam['pf_size'] = 500

    # load correct data according to eigenvalue
    Y = allres_min_idxes_BIASED.astype(float)
    opt_values = allres_min_idxes_BIASED.astype(float)[:-21]
    opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
    Y = np.array(re_hf.map_factors_to_preds(Y.reshape(-1), all_factors))
    opt_values = np.array(re_hf.map_factors_to_preds(opt_values, all_factors))
    opt_v3 = np.diag(all_res.loc[:, allres_min_idxes_BIASED])[:-21]
    opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))
    # DONE
    curparam["additional_inputs"] = opt_v3
    curparam['opt_values_factors'] = opt_values
    X = re_hf.load_additional_train_data(**curparam)
    train_size= 252
    #### SCALE
    if SCALE:
        scaler = StandardScaler()
        X[len_train - train_size : len_train + 21*(i-1), :] = scaler.fit_transform(X[len_train - train_size : len_train + 21*(i-1), :])
        X[len_train:, :] = scaler.transform(X[len_train:, :])

    res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_param,
                                                            single_train=False, expanding=False, train_size=train_size)
    mapped_res = re_hf.map_preds_to_factors(res, all_factors)
    Y_eval = all_rawres
    res = re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train)
    print(res)
    #RES_ALL.append(res)
    #mapped_res_ALL.append(mapped_res)

    print(re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train))


x = 0

"""
'include_ts_momentum_var_allstocks': False,
'include_ts_momentum_allstocks': True,
'include_sample_covmat_trace': True,
'include_mean_of_correls': True,
'include_iqr': False,
'include_factors': True,
'include_ewma_year': False,
'include_ewma_month': True,
'include_ew_year_vola': False,
'include_ew_month_vola': False,
'include_allstocks_year_avgvola': True,
'include_allstocks_month_avgvola': False,
"""

bools_temp = [False, True, True, True, False, True, False, True, False, False, True, False]
bools_temp =   [True, False, False, True, False, False, False, True, True, True, False, False]

for i, k in enumerate(list(curparam.keys())[2:-1]):
    curparam[k] = bools_temp[i]