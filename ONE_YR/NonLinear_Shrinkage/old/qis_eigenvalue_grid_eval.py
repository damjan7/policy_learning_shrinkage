import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
from collections import defaultdic

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3])

from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs

import regression_evaluation_funcs as re_hf
import helper_functions_NL_RL as NL_hf



# define factors
all_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
               1.9, 2.0]
# num eigenvalues to modify
num_eigenvalues = [1, 5, 10, 25, 50]

# again all_res stores the results of next 21 days so we can use it for training
# but only with a gap of 21 days to avoid forward looking bias of course
'''
Description:
all_res is the 21 day lead std dev of the returns of the minvar portfolio
all_rawres is the return of each day
'''
all_rawres = {}
all_res = {}
qis_grid_data = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\qis_eigenvalue_grid_data"
for num_ev in num_eigenvalues:
    df = pd.read_csv(qis_grid_data + f"\\qis_grid_allres_{num_ev}_evs.csv")
    all_res[num_ev] = df
    df = pd.read_csv(qis_grid_data + f"\\qis_grid_all_rawres_{num_ev}_evs.csv")
    all_rawres[num_ev] = df

print("loaded data")

# Calculates the OOS results for each factor and each number of eigenvalues
# while keeping the factor constant for the whole OOS period
factor_results = {}
for k, v in all_rawres.items():
    tmp = round(v.iloc[5040:,].std() * np.sqrt(252) * 100, 2)
    factor_results[k] = tmp
factor_results_df = pd.DataFrame(factor_results)


# CAN use allres_min_idxes as a signal, as it is shifted by 21 days
# so no forward looking bias
allres_min_idxes = {}
for k, v in all_res.items():
    minima = v.idxmin(axis=1)[5040-22: -22].values
    allres_min_idxes[k] = minima
allres_min_idxes_df = pd.DataFrame(allres_min_idxes)


allres_min_idxes_full = {}
for k, v in all_res.items():
    minima = v.idxmin(axis=1)[: -21].values
    minima = np.insert(minima, 0, np.repeat(["1.0"], 21))
    allres_min_idxes_full[k] = minima
allres_min_idxes_full_df = pd.DataFrame(allres_min_idxes_full)

# for sanity check: BIASED version should generally be better than
# non biased version as it is literally the minimum over the future 21 days
# so using it as a signal should outperform
allres_min_idxes_BIASED = {}
for k, v in all_res.items():
    minima = v.idxmin(axis=1).values
    allres_min_idxes_BIASED[k] = minima
allres_min_idxes_BIASED_df = pd.DataFrame(allres_min_idxes_BIASED)

## Test
tmp = []
for i in range(allres_min_idxes[10].shape[0]):
    t1 = all_rawres[10].iloc[5040:].loc[:, allres_min_idxes[10][i]].iloc[i]
    tmp.append(t1)
print( np.std(tmp) * np.sqrt(252) * 100 )
    # 10.254910152698425
# or yielding the same result:
np.diag(all_rawres[10].iloc[5040:, ].loc[:, allres_min_idxes[10]]).std() * np.sqrt(252) * 100


# simple argmin rule, can be used as a benchmark
res_simple_argmin_rule = {}
for num_ev in num_eigenvalues:
    tmp = np.diag(all_rawres[num_ev].iloc[5040:, ].loc[:, allres_min_idxes[num_ev]])
    res_simple_argmin_rule[num_ev] = tmp
res_simple_argmin_rule = pd.DataFrame(res_simple_argmin_rule)

# simple argmin rule, with full allres_min, should be same results as above
res_simple_argmin_rule_v2 = {}
for num_ev in num_eigenvalues:
    tmp = np.diag(all_rawres[num_ev].loc[:, allres_min_idxes_full[num_ev]])[5040:]
    res_simple_argmin_rule_v2[num_ev] = tmp
res_simple_argmin_rule_v2 = pd.DataFrame(res_simple_argmin_rule_v2)

# simple argmin rule, biased (as a sanity check)
res_simple_argmin_rule_biased = {}
for num_ev in num_eigenvalues:
    tmp = np.diag(all_rawres[num_ev].loc[:, allres_min_idxes_BIASED[num_ev]])[5040:]
    res_simple_argmin_rule_biased[num_ev] = tmp
res_simple_argmin_rule_biased = pd.DataFrame(res_simple_argmin_rule_biased)

# actual argmin of rawres (as a sanity check)
res_actual_argmin = {}
for num_ev in num_eigenvalues:
    res_actual_argmin[num_ev] = []
    for i in range(5313//21):
        tmp_data = all_rawres[num_ev].iloc[5040 + 21*i: 5040 + 21*(i+1)]
        curmin_idx = tmp_data.std().idxmin()
        curmin = tmp_data.loc[:, curmin_idx]
        res_actual_argmin[num_ev] += curmin.tolist()
res_actual_argmin = pd.DataFrame(res_actual_argmin)


res_simple_RAWRES_argmin_rule = {}
for num_ev in num_eigenvalues:
    rawres_min_signal = all_rawres[num_ev].rolling(window=21, min_periods=21).std().idxmin(axis=1, skipna=True).fillna('1.0')
    tmp = (all_rawres[num_ev].loc[:, rawres_min_signal])
    tmp = np.diag(tmp)[5040:]
    res_simple_RAWRES_argmin_rule[num_ev] = tmp
res_simple_RAWRES_argmin_rule = pd.DataFrame(res_simple_RAWRES_argmin_rule)

res_actual_argmin_nonbiased = {}
for num_ev in num_eigenvalues:
    res_actual_argmin_nonbiased[num_ev] = []
    for i in range(5313//21):
        idx_min_data = all_rawres[num_ev].iloc[5040 - 21 + 21*i: 5040 - 21 + 21*(i+1)]
        curmin_idx = idx_min_data.std().idxmin()
        tmp_data = all_rawres[num_ev].iloc[5040 + 21*i: 5040 + 21*(i+1)]
        curmin = tmp_data.loc[:, curmin_idx]
        res_actual_argmin_nonbiased[num_ev] += curmin.tolist()
res_actual_argmin_nonbiased = pd.DataFrame(res_actual_argmin_nonbiased)

curmin_idxes_rawreturns = {}
for num_ev in num_eigenvalues:
    curmin_idxes_rawreturns[num_ev] = []
    t=[]
    for i in range(all_rawres[num_ev].shape[0]//21):
        tmp_data = all_rawres[num_ev].iloc[21*i : 21*(i+1)]
        curmin_idx = tmp_data.std().idxmin()
        curmin_idxes_rawreturns[num_ev] += [float(curmin_idx)]
curmin_idxes_rawreturns = pd.DataFrame(curmin_idxes_rawreturns)




# now run a simple model, i.e., regression:
# i.e. as I did before; all factors --> run a regression for each factor
# to predict what factor to use at each time point
# run multioutputregressor again, but also need additional train data for that

# Load additional train data as before (see LR_rolling_eval.py)
pf_size = 500
len_train = 5040



new_opt_res = {}
for num_ev in num_eigenvalues:
    opt_factors = allres_min_idxes_full[num_ev]
    Y = all_res[num_ev]  # for first test
    opt_values = np.diag(Y.loc[:, opt_factors])
    all_res_input = np.concatenate([np.array([[0.04] * 16] * 22), all_res[10].values], axis=0)[:-22]
    X = re_hf.load_additional_train_data(
        pf_size=pf_size,
        opt_values_factors=opt_values,
        include_ew_month_vola=True,
        include_ew_year_vola=True,
        include_sample_covmat_trace=True,
        include_allstocks_year_avgvola=True,
        include_allstocks_month_avgvola=True,
        include_factors=True,
        include_ts_momentum_allstocks=True,
        include_ts_momentum_var_allstocks=True,
        include_ewma_year=True,
        include_ewma_month=True,
        include_mean_of_correls=True,
        include_iqr=True
    )
    X_new = np.concatenate([X, all_res_input], axis=1)
    model_preds = re_hf.basic_multi_output_elastic_net_v2(X, Y.to_numpy(), len_train)
    res = re_hf.map_preds_to_factors(model_preds, all_factors)
    Y_eval = all_rawres[num_ev]
    res_final = re_hf.evaluate_all_factor_preds(res, Y_eval, len_train)
    new_opt_res[num_ev] = res_final

"""
model_preds2 = basic_multi_output_elastic_net_new(X, Y.to_numpy(), len_train)
res2 = re_hf.map_preds_to_factors(model_preds2, all_factors)
Y_eval = all_rawres[num_ev]
res_final2 = re_hf.evaluate_all_factor_preds(res2, Y_eval, len_train)
print(res_final2)
"""

''' Results:
{1: 10.36558308180358,
 5: 10.38645283459991,
 10: 10.397461364038545,
 25: 10.40191734516073,
 50: 10.398247680622294}
 Benchmark:
 res_simple_argmin_rule.std() * np.sqrt(252) * 100
1     10.298189
5     10.289072
10    10.255875
25    10.262643
50    10.299646

{1: 10.375566005953543,
 5: 10.377473965617572,
 10: 10.38975308642397,
 25: 10.391200461420533,
 50: 10.381207540414353}
 
 {1: 10.375566005953543,
 5: 10.377473965617572,
 10: 10.38975308642397,
 25: 10.391200461420533,
 50: 10.381207540414353}
'''
print("done")


for n in num_eigenvalues:
    t = all_rawres[n].loc[:, allres_min_idxes_full[n]]
    t = np.diag(t)
    print(n, t.std() * np.sqrt(252) * 100)

for n in num_eigenvalues:
    print(n, np.diag(all_rawres[n].iloc[5040:].loc[:, allres_min_idxes_full[n][5040:]]).std() * np.sqrt(252) * 100)

# benchmark = 9.345
np.diag(all_rawres[10].loc[:, ['1.0' for i in range(all_rawres[10].shape[0])]]).std() * np.sqrt(252) * 100

# should be 10.38, i.e. the NL with factor 1
np.diag(all_rawres[10].iloc[5040:].loc[:, ['1.0' for i in range(all_rawres[10].shape[0]-5040)]]).std() * np.sqrt(252) * 100

