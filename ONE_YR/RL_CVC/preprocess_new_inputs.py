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
    '''
    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]
    sample_covmats = []

    trace_list = []
    EW_year_vola_list = []
    EW_month_vola_list = []
    allstocks_month_avgvola = []
    allstocks_year_avgvola = []
    allstocks_q075_month = []
    allstocks_q025_month = []
    momentum_allstocks_list = []
    momentum_var_allstocks_list = []

    for idx in range(permnos.shape[0]):
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        sample_covmat = np.array(past_ret_mat.T @ past_ret_mat)

        # calculating some new inputs:
        trace = np.trace(sample_covmat)
        trace_list.append(trace)

        allstocks_yr_vola = past_ret_mat.std(axis=1)
        allstocks_month_vola = past_ret_mat.iloc[-21:, :].std(axis=1)

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

        EW_year_vola_list.append(EW_year_vola)
        EW_month_vola_list.append(EW_month_vola)

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

    return mydict






