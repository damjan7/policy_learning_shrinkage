import pandas as pd
import numpy as np
import pickle
import os
from collections import  Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import psutil
from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs
from helpers import eval_function_new
import helpers_linear_shrinkage as hf_ls
from ONE_YR.NonLinear_Shrinkage import regression_evaluation_funcs as re_hf
from sklearn.preprocessing import StandardScaler, Normalizer


psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6,7,8,9])


base_folder_path = r'H:\\all\\RL_Shrinkage_2024'

# IMPORT SHRK DATASETS
shrk_data_path = None
pf_size = 500

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\cov2Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = fr"{base_folder_path}\helpers"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too
start_date = str(optimal_shrk_data['date'].iloc[0])
start_date = start_date[0:4] + '-' + start_date[4:6] + "-" + start_date[6:]
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]

cvc_shrk = optimal_shrk_data['shrk_factor'].values[5040:]

len_train = 5040
end_date = fixed_shrk_data.shape[0]
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]


all_res = fixed_shrk_data.iloc[:, 2:].copy() * 100
all_factors = all_res.columns.astype(float).values
Y = all_res.idxmin(axis=1).values.astype(float)
opt_values = all_res.idxmin(axis=1).values.astype(float)[:-21]
opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(0.15, 21))
Y = np.array(re_hf.map_factors_to_preds(Y.reshape(-1), all_factors))
opt_values = np.array(re_hf.map_factors_to_preds(opt_values, all_factors))
opt_v3 = np.diag(all_res.loc[:, all_res.idxmin(axis=1).values])[:-21]
opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))


params = {
'pf_size' : pf_size,
'opt_values_factors' : opt_values,
'include_ts_momentum_var_allstocks': False,
'include_ts_momentum_allstocks': True,
'include_sample_covmat_trace': True,
'include_mean_of_correls': False,
'include_iqr': False,
'include_factors': False,
'include_ewma_year': False,
'include_ewma_month': True,
'include_ew_year_vola': False,
'include_ew_month_vola': True,
'include_allstocks_year_avgvola': True,
'include_allstocks_month_avgvola': False,
'additional_inputs' : opt_v3
}

X = re_hf.load_additional_train_data(**params)

'''
sc = StandardScaler().fit(X[:len_train, :])
X = sc.transform(X)
sc = StandardScaler().fit(Y[:len_train].reshape(-1, 1))
Y = sc.transform(Y.reshape(-1, 1))
'''
res = re_hf.general_single_output_ElasticNet_Lagged(X=X, Y=Y, len_train=5040)
#res = sc.inverse_transform(res)
#res = (np.round(res.reshape(-1, ), 0)).astype(int)
mapped_res = np.array(re_hf.map_preds_to_factors(res, all_factors)).astype(float)
r = eval_function_new.eval_fct_networkonly_1YR(mapped_res, rets_full, permnos, 0, val_indices_results)

eval_function_new.myplot(mapped_res[val_idxes_shrkges], cvc_shrk[val_idxes_shrkges])

res_df = pd.DataFrame({"ElasticNet_CVC": mapped_res, "CVC": cvc_shrk}, index=pd.to_datetime(permnos.index[5040:], format="%Y%m%d") )
res_df = res_df.iloc[list(range(0, 5313, 21)), :]  # because we only may change intensity on every rebalancing date
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 10})
ax.plot(res_df)
ax.set(xlabel='Year', ylabel='Shrinkage Intensity')
ax.grid()
path = r"H:\all\RL_Shrinkage_2024\ONE_YR\Linear_Shrinkage\Plots"
fig.savefig(path + f"\\elasticNet_cvc_p{pf_size}.svg")

#mapped_res = re_hf.map_preds_to_factors(res, all_factors)
#Y_eval = all_rawres
#res = re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train)



def eval_model_outputs_CVC(res, show_fig=True):
    res = [100 if r > 100 else r for r in res]
    res = [0 if r < 0 else r for r in res]
    res = re_hf.map_preds_to_factors(res, all_factors)
    Y_eval = all_rawres
    Y_eval.columns = Y_eval.columns.astype(str)
    res_evaluated2 = re_hf.evaluate_all_factor_preds(res, Y_eval, len_train)
    datetime_index = pd.to_datetime(permnos.index, format="%Y%m%d")
    res_df = pd.DataFrame(np.array(res, dtype=float), index=datetime_index[5040:])
    fig = px.line(res_df, height=500)
    print("Result:", res_evaluated2)
    if show_fig:
        fig.show()