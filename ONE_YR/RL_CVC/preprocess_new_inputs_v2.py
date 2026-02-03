import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# plotting
import matplotlib.pyplot as plt

from helpers import eval_funcs, eval_function_new


'''
This Script contains some functions to pre-process and create additional inputs for our algorithm

'''

def get_sample_covmats(rets_full, permnos):
    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]
    sample_covmats = []
    for idx in range(permnos.shape[0]):
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        sample_covmat = np.array(past_ret_mat.T @ past_ret_mat)
        sample_covmat = sample_covmat[np.triu_indices(sample_covmat.shape[0])]
        sample_covmats.append(sample_covmat)

    return sample_covmats


## Nur schon die 1Y und 21D historische vola von dem Equally-Weighted Portfolio des eligible universe
def get_new_inputs(rets_full, permnos):
    '''
    Calculates following new inputs:
     - allstocks_.. : calculates the avg volatility for each timeseries of stocks on their own
     - EW_.. : calculates the volatility of the equal weighted portfolio
     - momentum_.. : calculates time-series momentum (ratio of stocks with pos trend)
            --> for now momentum avg and std across all stocks for 1 month
    '''
    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]
    sample_covmats = []

    trace_list = []
    EW_year_vola_list = []
    EW_month_vola_list = []
    EW_ewma_year_list = []
    EW_ewma_month_list = []
    allstocks_month_avgvola = []
    allstocks_year_avgvola = []
    allstocks_q075_month = []
    allstocks_q025_month = []
    momentum_allstocks_list = []
    momentum_var_allstocks_list = []
    mean_of_correls_list = []

    """    
    'pf_size' : pf_size,
    'opt_values_factors' : opt_values,
    'include_ts_momentum_var_allstocks': False,
    'include_ts_momentum_allstocks': True,
    'include_sample_covmat_trace': True,
    'include_mean_of_correls': True,
    'include_iqr': False,
    'include_factors': False,
    'include_ewma_year': False,
    'include_ewma_month': True,
    'include_ew_year_vola': False,
    'include_ew_month_vola': True,
    'include_allstocks_year_avgvola': True,
    'include_allstocks_month_avgvola': False
    """

    for idx in range(permnos.shape[0]):
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        sample_covmat = np.array(past_ret_mat.T @ past_ret_mat)

        # get off diagonal elements of correl matrix
        sample_stds = np.sqrt(np.diag(sample_covmat))
        helper1 = np.outer(sample_stds, sample_stds)
        correl_matrix = sample_covmat / helper1
        off_diag_correls_sorted = np.sort(correl_matrix[np.triu_indices(permnos.shape[1], k=1)])[::-1]
        mean_of_correls = off_diag_correls_sorted[0:int(off_diag_correls_sorted.shape[0])].mean()
        mean_of_correls_list.append(mean_of_correls)

        # calculating some new inputs:
        trace = np.trace(sample_covmat)
        trace_list.append(trace)

        allstocks_yr_vola = past_ret_mat.std(axis=0)
        allstocks_month_vola = past_ret_mat.iloc[-21:, :].std(axis=0)

        allstocks_year_avgvola.append(allstocks_yr_vola.mean() * np.sqrt(252) )
        allstocks_month_avgvola.append(allstocks_month_vola.mean() * np.sqrt(252))

        allstocks_q075_month.append(allstocks_month_vola.quantile(0.75) * np.sqrt(252) )
        allstocks_q025_month.append(allstocks_month_vola.quantile(0.25) * np.sqrt(252) )

        #
        momentum_allstocks = (past_ret_mat.iloc[-21:, :] > 0).sum().mean()
        momentum_var_allstocks = (past_ret_mat.iloc[-21:, :] > 0).sum().std()
        momentum_allstocks_list.append(momentum_allstocks)
        momentum_var_allstocks_list.append(momentum_var_allstocks)

        # equal weighted portfolio calculation
        daily_EW_rets = (past_ret_mat / past_ret_mat.shape[1]).sum(axis=1)
        EW_year_vola = np.std(daily_EW_rets) * np.sqrt(252)
        EW_month_vola = np.std(daily_EW_rets.iloc[-21:]) * np.sqrt(252)

        decay = 0.1
        EW_ewma_year = daily_EW_rets.ewm(alpha=decay).mean().iloc[-1] * np.sqrt(252)
        EW_ewma_month = daily_EW_rets.iloc[-21:].ewm(alpha=decay).mean().iloc[-1] * np.sqrt(252)

        EW_year_vola_list.append(EW_year_vola)
        EW_month_vola_list.append(EW_month_vola)

        EW_ewma_year_list.append(EW_ewma_year)
        EW_ewma_month_list.append(EW_ewma_month)



    # the following are all not standardizesss
    # standardize either with some pandas fct, that isnt forward peaking
    # or do it manually in a loop
    mydict = {}
    mydict['sample_covmat_trace'] = trace_list
    mydict['EW_Year_vola'] = EW_year_vola_list
    mydict['EW_Month_vola'] = EW_month_vola_list
    mydict['allstocks_q075_month'] = allstocks_q075_month
    mydict['allstocks_q025_month'] = allstocks_q025_month
    mydict['allstocks_year_avgvola'] = allstocks_year_avgvola
    mydict['allstocks_month_avgvola'] = allstocks_month_avgvola
    mydict['momentum_allstocks'] = momentum_allstocks_list
    mydict['momentum_var_allstocks'] = momentum_var_allstocks_list
    mydict['EW_ewma_year'] = EW_ewma_year_list
    mydict['EW_ewma_month'] = EW_ewma_month_list
    mydict['mean_of_correls'] = mean_of_correls_list

    return mydict

