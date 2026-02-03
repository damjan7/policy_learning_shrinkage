import pandas as pd

import pickle
import numpy as np
from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6,7,8,9,10])

from collections import defaultdict


def create_fixed_shrk_datasets(path, end_date, p, out_pf_sample_period_length, estimation_window_length,
                         out_path_shrk, out_path_dat, estimator):

    print(f"Begin with p = {p}...")
    # just load data if they already exist
    rets_permnos_path = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\rets_permnos_1Y"
    with open(rf"{rets_permnos_path}/returns_full_1Y_p{p}.pickle", 'rb') as f:
        rets_full = pd.read_pickle(f)
    with open(rf"{rets_permnos_path}/permnos_1Y_p{p}.pickle", 'rb') as f:
        permnos = pd.read_pickle(f)


    #past ret matrix --> list of past ret matrices
    shrk_intensities_v2 = np.round(np.linspace(0, 1, 101), 2)
    colnames = list(shrk_intensities_v2.astype(str))
    res = [["date", "hist_vola"] + colnames]

    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]
    assert add_idx == 259, "ADD IDX IS NOT 259, CHECK DATES"

    shrk_rawres_all = defaultdict(list)
    for idx in range(rets_full.shape[0]-259-21):
        shrk_res = []
        date = permnos.index[idx]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + 259 - 252: idx + 259, :]

        # check if any of the cols has more than 10 nans:
        if any(past_ret_mat.isna().sum() >= 20):  # idx 1026/7/8, 7108/9/10, 8622/3/4
            print("we have more than xy NaN's in a idx:", idx)

        # maybe a fix for now only
        # fill nans of past ret mats and demean
        past_ret_mat = past_ret_mat.fillna(0)
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())

        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + 259: idx + 259 + 21, :]
        if any(fut_ret_mat.isna().sum() > 0):
            print("we have more than 0 NaN's in a idx:", idx)

        shrk_est, sample, target = estimator(past_ret_mat)
        for shrk in shrk_intensities_v2:

            
            new_shrk_est = shrk
            covmat_est = new_shrk_est * target + (1-new_shrk_est) * sample
            # based on covmat --> calc pf std (and maybe return)
            pf_ret, pf_std = hf.calc_pf_std_return(covmat_est, fut_ret_mat)
            shrk_res.append(pf_std)

            weights = hf.calc_global_min_variance_pf(covmat_est)
            

            if idx % 21 == 0:
                new_shrk_est = shrk
                covmat_est = new_shrk_est * target + (1 - new_shrk_est) * sample
                weights = hf.calc_global_min_variance_pf(covmat_est)
                rawres = list(fut_ret_mat @ weights)
                shrk_rawres_all[shrk] += rawres

        # also want historical (can choose days) vola (and in future maybe different factors)
        # hist_vola = hf.get_historical_vola(past_price_matrices[idx], days=60)
        hist_vola = 0
        res.append([date, hist_vola] + shrk_res)
        if idx % 500 == 0:
            print(f"done {idx} out of {rets_full.shape}")

    df = pd.DataFrame(res)
    df.columns = df.iloc[0, :]
    df = df.drop(0)
    with open(rf"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\training_dfs\PF{p}/fixed_shrkges_{estimator.__name__}_p{p}.pickle", 'wb') as pickle_file:
        pickle.dump(df, pickle_file)

    rawres_df = pd.DataFrame(shrk_rawres_all)
    with open(rf"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\training_dfs\PF{p}/fixed_shrkges_rawres_{estimator.__name__}_p{p}.pickle", 'wb') as pickle_file:
        pickle.dump(rawres_df, pickle_file)


##### Let's call the function to create the necessary data frames
in_path = r"H:\all\RL_Shrinkage_2024\CRSP_2022_03.csv"
end_date = 20220331  # create it for the full data set
estimation_window_length = -99
out_of_sample_period_length = -99
pf_size = 225  # [30, 50, 100, 225, 500]
return_data_path1 = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets"
return_data_path2 = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
estimator = estimators.cov2Para
estimator = estimators.cov1Para

# in_path = None, if the necessary matrices already exist
for pf_size in [30, 50, 100, 225]:
    create_fixed_shrk_datasets(path=in_path,
                         end_date=end_date,
                         p=pf_size,
                         out_pf_sample_period_length=out_of_sample_period_length,
                         estimation_window_length=estimation_window_length,
                         out_path_shrk=return_data_path1,
                         out_path_dat=return_data_path2,
                         estimator=estimator
                         )

print("done")

