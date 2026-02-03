import pandas as pd
import plotly.express as px
import numpy as np
import os
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import pickle


import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6,7,8,9])

os.chdir(r'H:\all\RL_Shrinkage_2024')
from ONE_YR.NonLinear_Shrinkage import regression_evaluation_funcs as re_hf
from helpers import eval_function_new
from helpers import helper_functions as hf


## load permnos, and extract the dates for which we need the market caps

PF_SIZE = 30
base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
# IMPORT SHRK DATASETS
pf_size = PF_SIZE  # DONT CHANGE HERE!!
permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

rebdates_strings = permnos.index[list(range(5040, permnos.shape[0], 21))]
permnos.index = pd.to_datetime(permnos.index, format="%Y%m%d")
rebdates = permnos.index[list(range(5040, permnos.shape[0], 21))]
assert len(rebdates) == 253, "Rebdates is not of correct length.."



# load CRSP data ?
from helpers import helper_functions_RL as hf_rl
def create_data_matrices(path, end_date, p, out_pf_sample_period_length, estimation_window_length, out_path):
    """
    This function takes the path to the raw data, two other inputs, and saves all the necessary dataframes
    NOTE: past return matrices are DE-MEANED, future matrices are NOT
    :param path: path to the raw data
    :param end_date: end date we want to consider
    :param p: num of stocks considered for building the portfolio
    :return:
    """
    df, trading_days, rebalancing_days, start_date = hf_rl.load_preprocess_rebdates_only(path=path, end_date=end_date,
                                                                           out_of_sample_period_length=out_pf_sample_period_length,
                                                                           estimation_window_length=estimation_window_length,
                                                                           rebdates=rebdates_strings)

    # start_date is returned but nowhere used :-)
    # returns just every trading day, hence no estimation window length is needed
    #rebalancing_days_full = hf_rl.get_full_rebalancing_dates_matrix(rebalancing_days)
    #p_largest_stocks = hf_rl.get_p_largest_stocks_all_reb_dates_V2(df, rebalancing_days_full, p)

    #with open(rf"./rets_permnos_1Y/permnos_1Y_p{p}.pickle", 'wb') as pickle_file:
    #    pickle.dump(p_largest_stocks, pickle_file)

    p_largest_stocks = pd.read_pickle(rf"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{p}.pickle")

    # get unique permnos and create full return matrix, containing permnos that are needed
    unique_permnos = np.unique(np.concatenate(p_largest_stocks.values, axis=0))
    df2 = df[df['PERMNO'].isin(unique_permnos)]

    df3 = df2.pivot(columns='PERMNO', index='date', values='MARKET_CAP')
    # df3 = df3.fillna(0)  --> this serves as a control mechanism --> just fill NaN's in past ret mats in the other functions directly
    with open(rf"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\rets_permnos_1Y/oos_rebdate_marketcaps_1Y_p{p}.pickle", 'wb') as pickle_file:
        pickle.dump(df3, pickle_file)

    print("done")


##### Let's call the function to create the necessary data frames
in_path = r"H:\all\RL_Shrinkage_2024\CRSP_2022_03.csv"
end_date = 20220302
estimation_window_length = -99
out_of_sample_period_length = -99
pf_size = PF_SIZE 
return_data_path = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\rets_permnos_1Y"

create_data_matrices(path=in_path,
                     end_date=end_date,
                     p=pf_size,
                     out_pf_sample_period_length=out_of_sample_period_length,
                     estimation_window_length=estimation_window_length,
                     out_path=return_data_path
                     )

