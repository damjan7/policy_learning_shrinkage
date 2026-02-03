import pandas as pd
import numpy as np
import pickle
import os

from collections import defaultdict

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6,7,8,9])


from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs


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
qis_shrkges_path = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\QIS_shrinkages.csv"
qis_shrkges_v2 = pd.read_csv(qis_shrkges_path)
qis_deltas_full = qis_shrkges_v2.copy()


reb_date_indices = [21*i for i in range(10353//21)]

qis_evs = []
sample_evs = []
cvc_evs = []

# 8324, 8325

for pf_size in [500]:
    qis_evs = []
    sample_evs = []
    permnos = pd.read_pickle(
        fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
    rets_full = pd.read_pickle(
        fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")
    for idx in range(0, permnos.shape[0]):
        try:
            past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
            past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
            past_ret_mat = past_ret_mat.fillna(0)
            fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
        except:
            print("f2")

        #temp1, temp2, qis_base = estimators.QIS_for_RL(past_ret_mat)


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


        temp1, temp2, qis_base = estimators.QIS_for_RL(past_ret_mat)


        #qis_base = qis_deltas_full.iloc[idx, :].copy()


        qis_evs.append(qis_base)
        sample_evs.append(lambda1)


    path = rf"H:\all\RL_Shrinkage_2024\ONE_YR\NonLinear_Shrinkage\transformed_qis_eigenvalues"

    qis_evs_df = pd.DataFrame(qis_evs)
    qis_evs_df.to_csv(path + fr"\qis_evs_df_p{pf_size}.csv")

    sample_evs_df = pd.DataFrame(sample_evs)
    sample_evs_df.to_csv(path + fr"\sample_evs_df_p{pf_size}.csv")


sample_evs_df = pd.DataFrame(sample_evs)
intensities_v1 = qis_evs_df.iloc[:, :-251]
intensities_v2 = qis_evs_df.iloc[:, -251:] / sample_evs_df.iloc[:, -251:]
intensities_full = pd.concat([intensities_v1, intensities_v2], axis=1)


# QIS reweighted
exp_weights =[1 - 0.5 ** (1 + 0.01*i) for i in range(500)]
exp_weights = exp_weights[::-1]
qis_evs_reweighted = qis_evs_df * exp_weights
qis_intensities_reweighted = qis_evs_reweighted.copy()
qis_intensities_reweighted.iloc[:, -251:] = qis_intensities_reweighted.iloc[:, -251:] / sample_evs_df.iloc[:, -251:]

print("done")
########## DONE

# short analysis
import seaborn as sns

'''
# mean of full intensities, for largest 50 eigenvalues
intensities_full.iloc[:, -50:].mean()

ax = sns.lineplot(intensities_full.iloc[:, -100:].mean())
ax.set(title="Largest 100 Eigenvalues Considered",
       ylabel= "Mean of EV(QIS) / EV(Sample) Across all Rebalancing Dates")
ax.show()


ax = sns.lineplot(intensities_full.iloc[:, :].mean())
ax.set(title="All Eigenvalues Considered",
       ylabel= "Mean of EV(QIS) / EV(Sample) [=Shrinkage Inensity] Across all Rebalancing Dates")
ax.show()

intensities_full.index = pd.to_datetime(permnos.index, format="%Y%m%d")
ax = sns.lineplot(intensities_full.mean(axis=1))
ax.set(title="Plot of Average EV(QIS) / EV(Sample) [=Shrinkage Inensity] Across Time",
       ylabel= "Mean of Shrinkage Intensity Across all Eigenvalues")
ax.show()

ax2 = sns.lineplot(intensities_full.iloc[:, -1])
ax2.set(title="Plot of Single EV(QIS) / EV(Sample) [=Shrinkage Inensity] Across Time",
       ylabel= "Single Shrinkage Intensity for Eigenvalue XY")
ax2.show()
'''


cov2para_evs = []
for idx in range(0, permnos.shape[0]):
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
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0

    shrinkage, sample, target = estimators.cov2Para(past_ret_mat)
    sigmahat =shrinkage*target+(1-shrinkage)*sample

    lambda1, u = np.linalg.eigh(sigmahat)  # use Cholesky factorisation
    lambda1 = lambda1.real.clip(min=0)  # reset negative values to 0

    cov2para_evs.append(lambda1)

cov2para_evs_df = pd.DataFrame(cov2para_evs)
cov2para_intensities = cov2para_evs_df.copy()
cov2para_intensities.iloc[:, -251:] = cov2para_evs_df.iloc[:, -251:].values / sample_evs_df.iloc[:, -251:].values

z=0

path = rf"H:\all\RL_Shrinkage_2024\ONE_YR\NonLinear_Shrinkage\transformed_qis_eigenvalues"
qis_evs_df.to_csv(path + r"\qis_evs_df.csv")
sample_evs_df.to_csv(path + r"\sample_evs_df.csv")
cov2para_evs_df.to_csv(path + r"\cov2para_evs_df.csv")

'''
ax = sns.lineplot(cov2para_intensities.mean())
ax.set(title="All Eigenvalues Considered (COV2PARA/SAMPLE)",
       ylabel= "Mean of EV(Cov2Para) / EV(Sample) [=Shrinkage Inensity] Across all Rebalancing Dates")
ax.show()


sample_evs_df.index = pd.to_datetime(permnos.index, format="%Y%m%d")
ax = sns.lineplot(sample_evs_df.iloc[:, -251:].mean(axis=1))
ax.set(title="Grand Mean of Sample Eigenvalues For Each Trading Date")
ax.show()
'''