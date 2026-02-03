from helpers import helper_functions as hf
from helpers import rl_covmat_ests_for_dataset as estimators
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def eval_avg_covmat_estimators(rets_full, permnos, val_indices):
    rets = []
    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]

    for i in range(len(val_indices)):
        idx = val_indices[i]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]

        shrk, sample, target = estimators.cov2Para(past_ret_mat)
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        sample_covmat = sample
        qis_covmat_est = estimators.QIS(past_ret_mat)

        covmat_combo = 0.4*cov2para_covmat_est + 0.1*sample_covmat + 0.5*qis_covmat_est
        covmat_combo = qis_covmat_est

        weights = hf.calc_global_min_variance_pf(covmat_combo)

        rets += list(fut_ret_mat @ weights)

    res = results = {
        "combo estimator pf return" : round(np.mean(rets) * 252 * 100, 2) ,
        "combo estimator pf sd" : round(np.std(rets) * np.sqrt(252) *100, 2) ,
    }
    return res


def eval_shrk_estimators(rets_full, permnos, val_indices, estimators):
    rets = {}
    shrkges = {}
    for estimator in estimators:
        rets[f"{estimator.__name__}"] = []
        shrkges[f"{estimator.__name__}"] = []

    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]

    for i in range(len(val_indices)):
        idx = val_indices[i]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]

        for estimator in estimators:
            if estimator.__name__ != "QIS":
                shrk, sample, target = estimator(past_ret_mat)
                covmat_est = shrk * target + (1 - shrk) * sample
            else:
                covmat_est = estimator(past_ret_mat)
            weights = hf.calc_global_min_variance_pf(covmat_est)
            rets[f"{estimator.__name__}"] += list(fut_ret_mat @ weights)
            shrkges[f"{estimator.__name__}"].append(shrk)

    for estimator in estimators:
        print(estimator.__name__)
        tmp = round(np.std(rets[estimator.__name__]) * np.sqrt(252) * 100, 2)
        print(f"Std. Dev.: {tmp}")


    return rets, shrkges