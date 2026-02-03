import pandas as pd
import numpy as np
import pickle
import os

from collections import defaultdict

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6])


from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf

from helpers import eval_funcs_multi_target
from helpers import eval_funcs

base_folder_path = r'/'
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

## ACTUAL QIS SHRINKAGES AS USED BY MODEL
## APPLY FACTOR TO THESE
qis_shrkges_path = r"/ONE_YR/preprocessing/QIS_shrinkages.csv"
qis_shrkges_v2 = pd.read_csv(qis_shrkges_path)

p1 = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\sample_eigenvalues.csv"
p2 = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\QIS_deltas.csv"
sample_evs = pd.read_csv(p1)
qis_deltas = pd.read_csv(p2).iloc[:, -251:]
qis_deltas_full = pd.read_csv(p2).iloc[:, :]
qis_deltas.columns = sample_evs.columns
qis_shrkges = qis_deltas/sample_evs

#qis_deltas_full.iloc[:, -251:] = qis_deltas_full.iloc[:, -251:].to_numpy() / sample_evs.to_numpy()
#qis_deltas_full = qis_deltas_full.mul((sample_evs.sum(axis=1) / qis_deltas_full.sum(axis=1)), axis=0)

qis_deltas_full = qis_deltas_full.mul((sample_evs.sum(axis=1) / qis_deltas_full.sum(axis=1)), axis=0)
#qis_deltas_full.iloc[:, -251:] = qis_deltas_full.iloc[:, -251:].to_numpy() / sample_evs.to_numpy()

# The actual QIS Eigenvalues
qis_deltas_full = qis_shrkges_v2.copy()


############## NEW IMPLEMENTATION

all_factors = [1.0, 2.0, 2.1, 2.3, 2.5, 2.7, 2.9]

all_factors = [1.0, 2]
# qis_deltas_full.iloc[idx, :].rolling(10).mean().fillna(qis_deltas_full.iloc[idx, 0])
k = [0.1 + i*0.004 for i in range(500)]
qis_new = qis_deltas_full.copy().mul(k, axis=1)


reb_date_indices = [21*i for i in range(10353//21)]

all_res = defaultdict(list)
all_rawres = defaultdict(list)

for idx in range(permnos.shape[0]):
    try:
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
    except:
        print("f2")

    N, p = past_ret_mat.shape
    sample = pd.DataFrame(np.matmul(past_ret_mat.T.to_numpy(), past_ret_mat.to_numpy())) / (N - 1)
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)              #sort df by column index
    temp1 = dfu.to_numpy()
    temp3 = dfu.T.to_numpy().conjugate()

    qis_base = qis_deltas_full.iloc[idx, :]
    for factor in all_factors:
        if factor == 1:
            qis = qis_base.copy()
        else:
            qis = qis_new.iloc[idx, :]

        #qis = qis_base.copy()
        #qis.iloc[-10:] = qis.iloc[-10:] * factor


        temp2 = np.diag(qis)
        sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
        try:
            weights = hf.calc_global_min_variance_pf(sigmahat)
        except:
            print("f")
        all_res[factor].append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        if idx % 21 == 0:
            all_rawres[factor] += list(fut_ret_mat @ weights)

print("done")
########## DONE





# try 1:
qis_v2 = qis_deltas_full.copy()
qis_v2.iloc[:, :] = qis_v2.iloc[:, :] * 0.7

qis_v3 = qis_deltas_full.copy()
qis_v3.iloc[:, :] = qis_v3.iloc[:, :] * 0.8

qis_v4 = qis_deltas_full.copy()
qis_v4.iloc[:, :] = qis_v4.iloc[:, :] * 0.9

qis_v5 = qis_deltas_full.copy()
qis_v5.iloc[:, :] = qis_v5.iloc[:, :] * 1.1

qis_v6 = qis_deltas_full.copy()
qis_v6.iloc[:, :] = qis_v6.iloc[:, :] * 1.2

qis_v7 = qis_deltas_full.copy()
qis_v7.iloc[:, :] = qis_v7.iloc[:, :] * 1.3

reb_date_indices = [21*i for i in range(10353//21)]

# *0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3
res1, res2, res3, res4, res5, res6, res7 = [], [], [], [], [], [], []
raw_res1, raw_res2, raw_res3, raw_res4, raw_res5, raw_res6, raw_res7 = [], [], [], [], [], [], []
# run and store the performance for every date in train set
for idx in range(permnos.shape[0]):
    try:
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
    except:
        print("f2")
    N, p = past_ret_mat.shape
    sample = pd.DataFrame(np.matmul(past_ret_mat.T.to_numpy(), past_ret_mat.to_numpy())) / (N - 1)
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)              #sort df by column index
    temp1 = dfu.to_numpy()
    temp3 = dfu.T.to_numpy().conjugate()

    qis_lst = [qis_deltas_full.iloc[idx, :], qis_v2.iloc[idx, :], qis_v3.iloc[idx, :],
               qis_v4.iloc[idx, :], qis_v5.iloc[idx, :],
               qis_v6.iloc[idx, :], qis_v7.iloc[idx, :]]
    for j, qis in enumerate(qis_lst):
        #temp2 = qis*(sum(lambda1)/sum(qis))
        temp2 = np.diag(qis)
        sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1,temp2),temp3))
        try:
            weights = hf.calc_global_min_variance_pf(sigmahat)
        except:
            print("f")
        if j==0:
            res1.append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        elif j==1:
            res2.append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        elif j==2:
            res3.append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        elif j==3:
            res4.append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        elif j==4:
            res5.append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        elif j==5:
            res6.append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        elif j==6:
            res7.append(np.std(fut_ret_mat @ weights) * np.sqrt(252))
        if idx % 21 == 0:
            if j == 0:
                raw_res1 += list(fut_ret_mat @ weights)
            elif j == 1:
                raw_res2 += list(fut_ret_mat @ weights)
            elif j==2:
                raw_res3 += list(fut_ret_mat @ weights)
            elif j==3:
                raw_res4 += list(fut_ret_mat @ weights)
            elif j==4:
                raw_res5 += list(fut_ret_mat @ weights)
            elif j==5:
                raw_res6 += list(fut_ret_mat @ weights)
            elif j==6:
                raw_res7 += list(fut_ret_mat @ weights)


# more for testing if some variation results in low pf sds
raw_res_test1, raw_res_test2, raw_res_test3 = [], [], []

qis_testing_v1 = qis_shrkges.copy()
qis_testing_v1.iloc[:, -5:] = qis_testing_v1.iloc[:, -5:] * 1.4
qis_testing_v2 = qis_shrkges.copy()
qis_testing_v2.iloc[:, -5:] = qis_testing_v2.iloc[:, -5:] * 1.5
qis_testing_v3 = qis_shrkges.copy()
qis_testing_v3.iloc[:, -5:] = qis_testing_v3.iloc[:, -5:] * 1.6


for idx in range(0, permnos.shape[0], 21):
    try:
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
    except:
        print("f2")

    N, p = past_ret_mat.shape
    sample = pd.DataFrame(np.matmul(past_ret_mat.T.to_numpy(), past_ret_mat.to_numpy())) / (N - 1)
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)              #sort df by column index
    temp1 = dfu.to_numpy()
    temp3 = dfu.T.to_numpy().conjugate()

    qis_lst = [qis_testing_v1.iloc[idx, :], qis_testing_v2.iloc[idx, :], qis_testing_v3.iloc[idx, :]]
    for j, qis in enumerate(qis_lst):
        temp2 = np.diag(qis)
        sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1,temp2),temp3))
        try:
            weights = hf.calc_global_min_variance_pf(sigmahat)
        except:
            print("f")
        if j==0:
            raw_res_test1 += list(fut_ret_mat @ weights)
        elif j==1:
            raw_res_test2 += list(fut_ret_mat @ weights)
        else:
            raw_res_test3 += list(fut_ret_mat @ weights)

print('d')

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