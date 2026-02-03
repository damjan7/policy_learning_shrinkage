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
p.cpu_affinity([0,1,2,3,4,5,6,7,8,9,10])

os.chdir(r'H:\all\RL_Shrinkage_2024')
from helpers import helper_functions as hf
from ONE_YR_long_only.NonLinear_Shrinkage import regression_evaluation_funcs as re_hf
from helpers import eval_function_new
from helpers import rl_covmat_ests_for_dataset as estimators

# LOAD EIGENVALUES
PF_SIZE = 500


base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
# IMPORT SHRK DATASETS
pf_size = PF_SIZE  # DONT CHANGE HERE!!
permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR_long_only\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR_long_only\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\ONE_YR_long_only\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov1Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{base_folder_path}\ONE_YR_long_only\preprocessing\training_dfs\PF{pf_size}\cov1Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)


qis_evs = pd.read_csv(fr"H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\transformed_qis_eigenvalues\qis_evs_df_p{PF_SIZE}.csv", index_col=0)
sample_evs = pd.read_csv(fr"H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\transformed_qis_eigenvalues\sample_evs_df_p{PF_SIZE}.csv", index_col=0)

from preprocess_new_inputs_v2 import get_new_inputs


x = get_new_inputs(rets_full, permnos)

print("done")