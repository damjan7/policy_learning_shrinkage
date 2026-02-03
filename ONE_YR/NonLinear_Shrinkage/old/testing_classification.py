import pandas as pd
import numpy as np
import pickle
import os

from collections import defaultdict

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([3,4, 5,6,7,8,9, 10, 11, 12, 13,14,15])

from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs
from helpers import eval_function_new

import regression_evaluation_funcs as re_hf
import helper_functions_NL_RL as NL_hf
from collections import Counter


from sklearn.preprocessing import StandardScaler, Normalizer


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


res_simple_RAWRES_argmin_rule = {}
for num_ev in num_eigenvalues:
    rawres_min_signal = all_rawres[num_ev].rolling(window=21, min_periods=1).mean().idxmin(axis=1)
    rawres_min_signal.pop(rawres_min_signal.index[-1])
    rawres_min_signal = pd.concat([ pd.Series(['1.0']), rawres_min_signal])
    tmp = (all_rawres[num_ev].loc[:, rawres_min_signal ])
    tmp = np.diag(tmp)[5040:]
    res_simple_RAWRES_argmin_rule[num_ev] = tmp
res_simple_RAWRES_argmin_rule = pd.DataFrame(res_simple_RAWRES_argmin_rule)
# now run a simple model, i.e., regression:
# i.e. as I did before; all factors --> run a regression for each factor
# to predict what factor to use at each time point
# run multioutputregressor again, but also need additional train data for that

# Load additional train data as before (see LR_rolling_eval.py)
pf_size = 500
len_train = 5040

lgbm_params = {'boosting_type': 'gbdt',
 'learning_rate': 0.05,
 'n_estimators': 200,
 'num_leaves': 31,
 'reg_lambda': 5}


num_ev=10


new_data = [[0.1 for i in range(21)] for i in range(16)]
new_data_df = pd.DataFrame(new_data).T
new_data_df.columns = all_res[num_ev].columns
normalized_lagged_allres = pd.concat((new_data_df, all_res[num_ev])).iloc[:-21,:]
scaler = StandardScaler()
normalized_lagged_allres = scaler.fit_transform(normalized_lagged_allres)


model_preds_ALL = {}
num_ev=10

Y = allres_min_idxes_BIASED[num_ev].astype(float)
opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))

Y = np.array(re_hf.map_factors_to_preds(Y.reshape(-1), all_factors))
opt_values = np.array(re_hf.map_factors_to_preds(opt_values, all_factors))

opt_v3 = np.diag(all_res[num_ev].loc[:, allres_min_idxes_BIASED[num_ev]])[:-21]
opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))

# evaluate
#mapped_res = re_hf.map_preds_to_factors(res, all_factors)
#Y_eval = all_rawres[num_ev]
#re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train)


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
    'additional_inputs' : (opt_v3)
}


from sklearn.model_selection import ParameterSampler
param_combs = list(ParameterSampler(params, 50))

res_dict = {}
for expanding in (True, False):
    for i, param in enumerate(param_combs):
        param["additional_inputs"] = opt_v3
        param['pf_size'] = 500
        X = re_hf.load_additional_train_data(**param)
        res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=True, expanding=expanding)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        res_dict[str(i)] = mapped_res

res_dict2   = {}
for expanding in (True, False):
    for i, param in enumerate(param_combs):
        param["additional_inputs"] = normalized_lagged_allres
        param['pf_size'] = 500
        X = re_hf.load_additional_train_data(**param)
        res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=False, expanding=expanding)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        res_dict2[str(i)] = mapped_res

        #(18, (9.367, 10.355)), (9.412, 10.322)
        # (40, (9.352, 10.356)), (9.399, 10.329)
        #[(0, (9.345, 10.364)),

res_dict_v2 = {}
for expanding in (True, False):
    for i, param in enumerate(param_combs):
        if i in [0, 18, 40]:
            param["additional_inputs"] = opt_v3
            param['pf_size'] = 500
            X = re_hf.load_additional_train_data(**param)
            res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=False, expanding=expanding)
            mapped_res = re_hf.map_preds_to_factors(res, all_factors)
            res_dict_v2[str(i)] = mapped_res

res_dict_v3 = {}
for expanding in (True):
    for i, param in enumerate(param_combs):
        if i in [0, 18, 40]:
            param["additional_inputs"] = opt_v3
            param['pf_size'] = 500
            X = re_hf.load_additional_train_data(**param)
            res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=False, expanding=expanding)
            mapped_res = re_hf.map_preds_to_factors(res, all_factors)
            res_dict_v3[str(i)] = mapped_res

res_dict_v4 = {}
for expanding in [False]:
    for i, param in enumerate(param_combs):
        if i in [0, 18, 40]:
            param["additional_inputs"] = opt_v3
            param['pf_size'] = 500
            X = re_hf.load_additional_train_data(**param)
            res = general_single_output_LGBMRegression_Lagged_SHORTER_TRAIN(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=False, expanding=expanding)
            mapped_res = re_hf.map_preds_to_factors(res, all_factors)
            res_dict_v4[str(i)] = mapped_res

res_dict_all_evs = {}
for num_ev in num_eigenvalues:
    curparam = param_combs[18]
    curparam['pf_size'] = 500

    # load correct data according to eigenvalue
    Y = allres_min_idxes_BIASED[num_ev].astype(float)
    opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
    opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
    Y = np.array(re_hf.map_factors_to_preds(Y.reshape(-1), all_factors))
    opt_values = np.array(re_hf.map_factors_to_preds(opt_values, all_factors))
    opt_v3 = np.diag(all_res[num_ev].loc[:, allres_min_idxes_BIASED[num_ev]])[:-21]
    opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))
    # DONE

    curparam["additional_inputs"] = opt_v3
    curparam['opt_values_factors'] = opt_values
    X = re_hf.load_additional_train_data(**curparam)
    res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=False, expanding=False)
    mapped_res = re_hf.map_preds_to_factors(res, all_factors)
    res_dict_all_evs[num_ev] = mapped_res
print("done")


### single trains
for num_ev in num_eigenvalues:
    curparam = param_combs[18]
    curparam['pf_size'] = 500

    # load correct data according to eigenvalue
    Y = allres_min_idxes_BIASED[num_ev].astype(float)
    opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
    opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
    Y = np.array(re_hf.map_factors_to_preds(Y.reshape(-1), all_factors))
    opt_values = np.array(re_hf.map_factors_to_preds(opt_values, all_factors))
    opt_v3 = np.diag(all_res[num_ev].loc[:, allres_min_idxes_BIASED[num_ev]])[:-21]
    opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))
    # DONE

    curparam["additional_inputs"] = opt_v3
    curparam['opt_values_factors'] = opt_values
    X = re_hf.load_additional_train_data(**curparam)
    res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=True, expanding=False, train_size=1000)
    mapped_res = re_hf.map_preds_to_factors(res, all_factors)
    Y_eval = all_rawres[num_ev]
    print(re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train))


p1 = {'boosting_type': 'gbdt',
 'learning_rate': 0.1,
 'n_estimators': 500,
 'num_leaves': 62,
 'reg_lambda': 5}

p2 = {'boosting_type': 'gbdt',
 'learning_rate': 0.05,
 'n_estimators': 200,
 'num_leaves': 31,
 'reg_lambda': 5}

p3 = {'boosting_type': 'gbdt',
 'learning_rate': 0.1,
 'n_estiFmators': 200,
 'num_leaves': 62,
 'reg_lambda': 1}

res_dict_all_evs_v2 = defaultdict(list)
for num_ev in num_eigenvalues:
    for lgbm_param in [p1, p2, p3]:

        curparam = param_combs[18]
        curparam['pf_size'] = 500

        # load correct data according to eigenvalue
        Y = allres_min_idxes_BIASED[num_ev].astype(float)
        opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
        Y = np.array(re_hf.map_factors_to_preds(Y.reshape(-1), all_factors))
        opt_values = np.array(re_hf.map_factors_to_preds(opt_values, all_factors))
        opt_v3 = np.diag(all_res[num_ev].loc[:, allres_min_idxes_BIASED[num_ev]])[:-21]
        opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))
        # DONE
        curparam["additional_inputs"] = opt_v3
        curparam['opt_values_factors'] = opt_values
        X = re_hf.load_additional_train_data(**curparam)
        res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_param, single_train=False, expanding=False, train_size=1000)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        Y_eval = all_rawres[num_ev]
        res = re_hf.evaluate_all_factor_preds(v, Y_eval, len_train)

        res_dict_all_evs_v2[num_ev].append(res)

        print(re_hf.evaluate_all_factor_preds(v, Y_eval, len_train))


params = {
    'pf_size' : pf_size,
    'opt_values_factors' : opt_values,
    'include_ew_month_vola' : True,
    'include_ew_year_vola' : True,
    'include_sample_covmat_trace' : True,
    'include_allstocks_year_avgvola' : True,
    'include_allstocks_month_avgvola' : True,
    'include_factors' : True,
    'include_ts_momentum_allstocks' : True,
    'include_ts_momentum_var_allstocks' : True,
    'include_ewma_year' : True,
    'include_ewma_month' : True,
    'include_mean_of_correls' : True,
    'include_iqr' : True,
    'additional_inputs' : opt_v3
}

# 10.39
p2 = {
    'pf_size' : pf_size,
    'opt_values_factors' : opt_values,
    'include_ew_month_vola' : True,
    'include_ew_year_vola' : True,
    'include_sample_covmat_trace' : True,
    'include_allstocks_year_avgvola' : False,
    'include_allstocks_month_avgvola' : False,
    'include_factors' : False,
    'include_ts_momentum_allstocks' : True,
    'include_ts_momentum_var_allstocks' : True,
    'include_ewma_year' : True,
    'include_ewma_month' : True,
    'include_mean_of_correls' : True,
    'include_iqr' : True,
    'additional_inputs' : opt_v3
}

X = re_hf.load_additional_train_data(**p2)
res = re_hf.general_single_output_LGBMRegression_Lagged(X=X, Y=Y, len_train=len_train, cur_params=lgbm_params, single_train=False, expanding=expanding)
mapped_res = re_hf.map_preds_to_factors(res, all_factors)
Y_eval = all_rawres[num_ev]
re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train)