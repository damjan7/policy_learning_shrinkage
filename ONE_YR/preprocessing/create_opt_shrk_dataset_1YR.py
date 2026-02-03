import pandas as pd

import pickle
import numpy as np
from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6,7,8,9])


def create_fixed_shrk_datasets(path, end_date, p, out_pf_sample_period_length, estimation_window_length,
                         out_path_shrk, out_path_dat, estimator):
    print(f"NOTE ADD IDX IS HARD-CODED TO 259, MAY CHECK IF IT IS OK")
    print(f"Begin with p = {p}...")
    # just load data if they already exist
    with open(rf"{path}/returns_full_1Y_p{p}.pickle", 'rb') as f:
        rets_full = pd.read_pickle(f)
    with open(rf"{path}/permnos_1Y_p{p}.pickle", 'rb') as f:
        permnos = pd.read_pickle(f)


    #only 1 as I do not consider more
    shrk_factors = [1.0]
    # loop through all factors and the past data
    # can add everything to 1 data

    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]
    assert add_idx == 259, "ADD IDX IS NOT 259, CHECK DATES"


    for factor in shrk_factors:
        res = [["date", "shrk_factor", "hist_vola", "pf_return", "pf_std"]]
        ## ADD IDX IS HARD-CODED AS IT SHOULD NOT CHANGE
        for idx in range(rets_full.shape[0]-259-21):
            past_return_matrix = rets_full[permnos.iloc[idx]].iloc[idx + 259 - 252: idx + 259, :]
            past_return_matrix = past_return_matrix.fillna(0)
            past_return_matrix = past_return_matrix.sub(past_return_matrix.mean())
            future_return_matrix = rets_full[permnos.iloc[idx]].iloc[idx + 259: idx + 259 + 21, :]

            '''
            if 13621 in past_return_matrix.columns:
                past_return_matrix = past_return_matrix.drop(columns=[13621])
                future_return_matrix = future_return_matrix.drop(columns=[13621])
            if 46842 in past_return_matrix.columns:
                past_return_matrix = past_return_matrix.drop(columns=[46842])
                future_return_matrix = future_return_matrix.drop(columns=[46842])
            '''

            shrk_est, sample, target = estimator(past_return_matrix)
            new_shrk_est = factor * shrk_est
            covmat_est = new_shrk_est * target + (1-new_shrk_est) * sample
            # based on covmat --> calc pf std (and maybe return)
            pf_ret, pf_std = hf.calc_pf_std_return(covmat_est, future_return_matrix)

            # also want historical (can choose days) vola (and in future maybe different factors)
            # hist_vola = hf.get_historical_vola(past_price_matrices[idx], days=60)
            hist_vola = 0
            date = permnos.index[idx]
            res.append([date, new_shrk_est, hist_vola, pf_ret, pf_std])
        # save to pandas dataframe and then to disk, for each factor separately
        df = pd.DataFrame(res)
        df.columns = df.iloc[0, :]
        df = df.drop(0)
        with open(rf"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\training_dfs\PF{p}/{estimator.__name__}_factor-{factor}_p{p}.pickle", 'wb') as pickle_file:
            pickle.dump(df, pickle_file)

estimator = estimators.cov2Para
pf_size = 500
in_path = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\rets_permnos_1Y"
estimator = estimators.cov1Para

for pf_size in [30, 50, 100, 225]:
    create_fixed_shrk_datasets(in_path, 0, pf_size, 0, 0, 0, 0, estimator)
print("done")