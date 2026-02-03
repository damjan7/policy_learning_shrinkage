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

# opt shrink = np.diag(all_res.loc[:, allres_min_idxes_full])
# opt shrink = allres_min_idxes_full.astype(float)

p1 = {'boosting_type': 'gbdt',
 'learning_rate': 0.1,
 'n_estimators': 500,
 'num_leaves': 62,
 'reg_lambda': 5}

opt_v3 = np.diag(all_res.loc[:, allres_min_idxes_BIASED])[:-21]
opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))

# OR
opt_values = allres_min_idxes_BIASED.astype(float)[:-21]
opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))


print("ok")
X = hf_ls.load_additional_train_data(
    pf_size=pf_size,
    opt_shrinkage_intensities=opt_values,
    include_ew_month_vola=True,
    include_ew_year_vola=True,
    include_sample_covmat_trace=True,
    include_allstocks_year_avgvola=True,
    include_allstocks_month_avgvola=True,
    include_factors=False,
    include_ts_momentum_allstocks=True,
    include_ts_momentum_var_allstocks=True,
    include_ewma_year=False,
    include_ewma_month=False,
    include_mean_of_correls=True,
    include_iqr=False
)

#Y = fixed_shrk_data.iloc[:, 3:].values

Y = all_res.to_numpy()

idx_subset = list(range(0,100,5))
#Y_subset = Y[:, idx_subset]
#subset_mapped_res = [idx_subset[i] for i in res]

res_full = hf_ls.basic_multi_output_LGBM_NonLagged(X, Y, len_train, p1)
# OR for quick testing
# res_full = hf_ls.basic_multi_output_elastic_net_NonLagged(X, Y, len_train)


res_full = hf_ls.basic_multi_output_LGBM_test(X, Y_subset, len_train)
res = hf_ls.basic_multi_output_LGBM_single_training(X, Y, len_train)

keys = [i for i in range(0, 100)]
values = fixed_shrk_data.iloc[:, 3:].columns
mydict = dict(zip(keys, values))
res2 = [mydict[k] for k in res]

print("done")

from helpers import eval_funcs, eval_function_new
r = eval_function_new.eval_fct_networkonly_1YR(np.array(res/100), rets_full, permnos, 0, val_indices_results)

# evaluate when we predict the shrk factor
sd = np.diag(all_rawres.iloc[5040:, :].iloc[:, res_full]).std() * np.sqrt(252) * 100
mean = np.diag(all_rawres.iloc[5040:, :].iloc[:, res_full]).mean() * 252 * 100
{'network pf return': round(mean, 3), 'network pf sd': round(sd, 3)}

# res_full = hf_ls.basic_multi_output_LGBM(X, Y, len_train)
path = r"H:\all\RL_Shrinkage_2024\ONE_YR\Linear_Shrinkage\results"

res_full_mapped = [all_res.columns[i] for i in res_full]

pd.DataFrame(np.diag((all_rawres.iloc[5040:, :].iloc[:, res_full]))).to_csv(
    path+"\\lgbm_cvc_model_returns.csv", index=False)

pd.DataFrame(res_full_mapped).to_csv(
    path+"\\lgbm_cvc_model_predictions.csv", index=False)

x=0
