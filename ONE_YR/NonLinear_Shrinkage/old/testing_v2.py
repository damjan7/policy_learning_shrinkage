import pandas as pd
import numpy as np
import pickle
import os

from collections import defaultdict

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([5,6,7,8,9, 10, 11, 12, 13,14,15])

from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs
from helpers import eval_function_new

import regression_evaluation_funcs as re_hf
import helper_functions_NL_RL as NL_hf
from collections import Counter



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


p2 = {'boosting_type': 'gbdt',
 'learning_rate': 0.05,
 'n_estimators': 200,
 'num_leaves': 31,
 'reg_lambda': 5}

num_ev=10

model_preds_ALL = {}
for num_ev in [10]:
    if 1 == 1:
        Y = all_res[num_ev]  # for first test
        Y = allres_min_idxes_BIASED[num_ev].astype(float)
        opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
        X = re_hf.load_additional_train_data(
            pf_size=pf_size,
            opt_values_factors=opt_values,
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
        res = re_hf.basic_multi_output_LGBM_Lagged_Expanding(X, Y.to_numpy(), len_train, p2)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        model_preds_ALL[num_ev] = mapped_res
        r2 = mapped_res


# argmin signal from allrawres
signals = []
for i in range(all_rawres[10].shape[0] // 21):
    tmp = all_rawres[10].iloc[21 * i: 21 * (i + 1), :]
    tmp = tmp.std()
    signal = tmp.idxmin()
    signals.append(signal)
list.insert(signals, 0, '0.5')
signals.pop()
signals = np.repeat(signals, 21)

def basic_multi_output_LGBM_v2(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*i : len_train + 21*(i-1), :]
            y_train = Y[21*i : len_train + 21*(i-1), :]
            x_test = X[len_train + 21*i : len_train + 21*(i+1), :]
            #x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, verbose=-1,
                              num_leaves=4, max_depth=3, learning_rate=0.5, n_estimators=2)
            )
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.std(axis=0).argmin())

print("done")


from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor


params = {
    "boosting_type": ['gbdt', 'dart'],
    'num_leaves': [31, 62, 93, 125],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200, 500],
    'reg_lambda': [0.0, 0.1, 1, 5]
}

# best:
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

model_preds_ALL = {}
for num_ev in [10]:
    if 1 == 1:
        #opt_factors = allres_min_idxes_full[num_ev]
        Y = all_res[num_ev]  # for first test
        opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
        X = re_hf.load_additional_train_data(
            pf_size=pf_size,
            opt_values_factors=opt_values
        )
        res = re_hf.basic_multi_output_LGBM_Lagged_version3(X, Y.to_numpy(), len_train, p2)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        model_preds_ALL[num_ev] = mapped_res
        r1 = mapped_res

new_data = [[0.1 for i in range(21)] for i in range(16)]
new_data_df = pd.DataFrame(new_data).T
new_data_df.columns = all_res[num_ev].columns
new_df = pd.concat((new_data_df, all_res[num_ev])).iloc[:-21,:]

model_preds_ALL = {}
for num_ev in [10]:
    if 1 == 1:
        #opt_factors = allres_min_idxes_full[num_ev]
        Y = all_res[num_ev]  # for first test
        opt_values = np.diag(Y.loc[:, allres_min_idxes_BIASED[num_ev]])[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(7.0, 21))
        X = re_hf.load_additional_train_data(
            pf_size=pf_size,
            opt_values_factors=opt_values,
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
        res = re_hf.basic_multi_output_LGBM_v2(X, Y.to_numpy(), len_train, p1)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        model_preds_ALL[num_ev] = mapped_res

x = 0
model_preds_ALL_df = pd.DataFrame(model_preds_ALL)

model_preds_ALL_sorted = {}
for k in (1, 5, 10, 25, 50):
    model_preds_ALL_sorted[k] = model_preds_ALL[k]

model_preds_ALL_df_v2 = pd.DataFrame(model_preds_ALL_sorted)

path = r"/ONE_YR/NonLinear_Shrinkage/results"

results = {}
results_returns = {}
for num_ev in num_eigenvalues:
    Y_eval = all_rawres[num_ev]
    res = re_hf.evaluate_all_factor_preds(model_preds_ALL_sorted[num_ev], Y_eval, len_train)
    results[num_ev] = res
    rets = Y_eval.iloc[len_train:].loc[:, model_preds_ALL_sorted[num_ev]]
    results_returns[num_ev] = np.diag(rets)

results_returns_df = pd.DataFrame(results_returns)

results_returns_df.to_csv(path + "\\lgbm_model_returns.csv", index=False)
model_preds_ALL_df_v2.to_csv(path + "\\lgbm_model_predictions.csv", index=False)


print("done")


# (9.458794917094664, 10.277402364121595)
# i think this should work with non lagged version
# because all the local measures are bast on past monthly and yearly data
# yes it would, but only predicting the 1 day in advance,
# otherwise im using data to which I don't have access at the current rebalancing
# now let's try the same but just predict 1 day not 21 days,
# which makes more sense as I want to use the 1 day prediction to form my PF for the
# next 21 days
model_preds_ALL = {}
for num_ev in [10]:
    if 1 == 1:
        Y = all_res[num_ev]  # for first test
        opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
        X = re_hf.load_additional_train_data(
            pf_size=pf_size,
            opt_values_factors=opt_values,
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
        res = re_hf.basic_multi_output_LGBM_NonLagged(X, Y.to_numpy(), len_train, p2)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        model_preds_ALL[num_ev] = mapped_res
        r1 = mapped_res

# now with different 1 day prediction
model_preds_ALL = {}
for num_ev in [10]:
    if 1 == 1:
        Y = all_res[num_ev]  # for first test
        opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
        X = re_hf.load_additional_train_data(
            pf_size=pf_size,
            opt_values_factors=opt_values,
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
        res = re_hf.basic_multi_output_LGBM_NonLagged(X, Y.to_numpy(), len_train, p2)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        model_preds_ALL[num_ev] = mapped_res
        r1 = mapped_res

# 1 day prediction
# (9.555334302488792, 10.366293998093063)
model_preds_ALL = {}
for num_ev in [10]:
    if 1 == 1:
        Y = all_res[num_ev]  # for first test
        opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
        X = re_hf.load_additional_train_data(
            pf_size=pf_size,
            opt_values_factors=opt_values,
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
        res = re_hf.basic_multi_output_LGBM_NonLagged_OneDayPred(X, Y.to_numpy(), len_train, p2)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        model_preds_ALL[num_ev] = mapped_res
        r2 = mapped_res


model_preds_ALL = {}
for num_ev in [10]:
    if 1 == 1:
        Y = all_res[num_ev]  # for first test
        opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
        opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
        X = re_hf.load_additional_train_data(
            pf_size=pf_size,
            opt_values_factors=signals,
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
        res = re_hf.basic_multi_output_LGBM_NonLagged_testV1(X, Y.to_numpy(), len_train, p2)
        mapped_res = re_hf.map_preds_to_factors(res, all_factors)
        model_preds_ALL[num_ev] = mapped_res
        r1 = mapped_res


'''
Try out training with a single dataset but using LGBM (not rolling)
'''
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
Y = all_res[num_ev].to_numpy()  # for first test
opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
X = re_hf.load_additional_train_data(
    pf_size=pf_size,
    opt_values_factors=opt_values,
    additional_inputs=new_df

)

model_predictions = []
for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
    if len_train + 21 * (i + 1) > X.shape[0]:
        #model_predictions = np.repeat(model_predictions, 21)
        pass
    else:
        if i == 0:
            x_train = X[21 * i: len_train + 21 * (i - 1), :]
            y_train = Y[21 * i: len_train + 21 * (i - 1), :]
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, verbose=-1, **p2)
            )
            regr = regr.fit(x_train, y_train)

        x_test = X[len_train + 21 * (i - 1): len_train + 21 * i, :]
        #x_test = np.ascontiguousarray(X[len_train + 21*i].reshape(-1,1))
        preds = regr.predict(x_test)
        print(f"fitted model in iteration {i} out of {5313 // 21}")
        model_predictions.append(preds.mean(axis=0).argmin())

model_predictions = np.repeat(model_predictions, 21)
mapped_res = re_hf.map_preds_to_factors(model_predictions, all_factors)
Y_eval = all_rawres[num_ev]
re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train)


xyz={"pf_size": 500,
     "opt_values_factors": opt_values}

X = re_hf.load_additional_train_data(**xyz)

X = re_hf.load_additional_train_data(
    pf_size=pf_size,
    opt_values_factors=opt_values,
    additional_inputs=new_df

)


'''
Try out training with a single dataset but using LGBM (not rolling)
ADDITIONALLY TRY NEW Y THAT DOESN'T NEED LAGS?
'''
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
Y = all_rawres[num_ev].to_numpy()  # for first test
opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
X = re_hf.load_additional_train_data(
    pf_size=pf_size,
    opt_values_factors=opt_values,
)
model_predictions = []
for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
    if len_train + 21 * (i + 1) > X.shape[0]:
        #model_predictions = np.repeat(model_predictions, 21)
        pass
    else:
        if i == 0:
            x_train = X[21 * i: len_train + 21 * (i), :]
            y_train = Y[21 * i: len_train + 21 * (i), :]
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, verbose=-1, **p2)
            )
            regr = regr.fit(x_train, y_train)

        x_test = X[len_train + 21 * (i - 1): len_train + 21 * i, :]
        #x_test = np.ascontiguousarray(X[len_train + 21*i].reshape(-1,1))
        preds = regr.predict(x_test)
        print(f"fitted model in iteration {i} out of {5313 // 21}")
        model_predictions.append(preds.std(axis=0).argmin())

model_predictions = np.repeat(model_predictions, 21)

mapped_res = re_hf.map_preds_to_factors(model_predictions, all_factors)
Y_eval = all_rawres[num_ev]
re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train)

#
for i in range(100):
    r = np.random.randint(0, 16, 253)
    r = np.repeat(r,21)
    mapped_res = re_hf.map_preds_to_factors(r, all_factors)
    print(re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train))