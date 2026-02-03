# %%
import pandas as pd
import plotly.express as px
import numpy as np
import os
from collections import defaultdict
from collections import Counter

import matplotlib.pyplot as plt

# %%
import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6])

# %%
os.chdir(r'H:\all\RL_Shrinkage_2024')
from helpers import helper_functions as hf

# %%
import glob

path = r'H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\transformed_qis_eigenvalues'
extension = 'csv'
os.chdir(path)
filenames = glob.glob('*.{}'.format(extension))
print(filenames)



def get_rawres(eigenvalue_dict, modelnames: list):
    tmp_res = defaultdict(list)
    tmp_rawres = defaultdict(list)
    for idx in range(0, permnos.shape[0]):
        try:
            past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21 * 12 * 1: idx + add_idx, :]
            past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
            past_ret_mat = past_ret_mat.fillna(0)
            fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
        except:
            print("Some Error..")
            
        N, p = past_ret_mat.shape
        sample = pd.DataFrame(np.matmul(past_ret_mat.T.to_numpy(), past_ret_mat.to_numpy())) / (N - 1)
        lambda1, u = np.linalg.eigh(sample)
        lambda1 = lambda1.real.clip(min=0)
        dfu = pd.DataFrame(u,columns=lambda1)
        dfu.sort_index(axis=1,inplace = True)
        temp1 = dfu.to_numpy()
        temp3 = dfu.T.to_numpy().conjugate()

        for cur_modelname in modelnames:
            qis = eigenvalue_dict[cur_modelname].iloc[idx, :]
            temp2 = np.diag(qis)
            sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
            try:
                weights = hf.calc_global_min_variance_pf_long_only(sigmahat)
            except:
                print("Some Other Error..")
            # store results
            tmp_res[cur_modelname].append(np.std(fut_ret_mat @ weights, ddof=1) * np.sqrt(252) * 100)
            if idx % 21 == 0:
                tmp_rawres[cur_modelname] += list(fut_ret_mat @ weights)

        if idx % 250 == 0:
            print(f"done {idx} out of {permnos.shape[0]}")

    return tmp_rawres, tmp_res



for PF_SIZE in [225,100, 50, 30]:
    model_names = ["qis",  "sample"]
    evs_dfs = {}
    qis_evs_path = r"H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\transformed_qis_eigenvalues"
    evs_dfs["qis"] = pd.read_csv(qis_evs_path + f'\\qis_evs_df_p{PF_SIZE}.csv', index_col=0)
    evs_dfs["sample"] = pd.read_csv(qis_evs_path + f'\\sample_evs_df_p{PF_SIZE}.csv', index_col=0)

    base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
    permnos = pd.read_pickle(
        fr"{base_folder_path}\ONE_YR_long_only\preprocessing\rets_permnos_1Y\permnos_1Y_p{PF_SIZE}.pickle")


    def get_new_eigenvalues(qis_eigenvalues, sample_eigenvalues, intensity_of_intensity):
        intensity = qis_eigenvalues.copy()
        if qis_eigenvalues.shape[1] == 500:
            intensity.iloc[: , -251:] = qis_eigenvalues.iloc[: , -251:] / sample_eigenvalues.iloc[: , -251:]
        else:
            intensity = qis_eigenvalues / sample_eigenvalues
        intensity_delta = intensity - 1
        intensity_delta_new = intensity_delta * intensity_of_intensity
        intensity_new = intensity_delta_new + 1
        qis_evs_new = intensity_new * sample_eigenvalues
        if qis_eigenvalues.shape[1] == 500:
            qis_evs_new.iloc[: , 0:250] = intensity_of_intensity * qis_eigenvalues.iloc[: , 0:250]

        # get Rotation Points
        intens_df = qis_eigenvalues / sample_eigenvalues
        if qis_eigenvalues.shape[1] == 500:  # only 1 year case ofcourse
            intens_df = intens_df.fillna(0)
        right_rotation_idx = np.argmin( np.abs(intens_df-1) , axis=1)
        left_rotation_idx = right_rotation_idx - 1
        rotation_point_sample_evs = 0.5 * np.diag(sample_eigenvalues.iloc[:, left_rotation_idx]) + 0.5 * np.diag(sample_eigenvalues.iloc[:, right_rotation_idx])
        rotation_points = 0.5 * np.diag(intens_df.iloc[:, left_rotation_idx]) + 0.5 * np.diag(intens_df.iloc[:, right_rotation_idx])
        rotation_points = rotation_points * rotation_point_sample_evs

        # Force all eigenvalues to be in decreasing order (from the largest one)
        idx_is_increasing = qis_evs_new.diff(axis=1).fillna(0) < 0
        qis_evs_new[idx_is_increasing] = np.nan
        qis_evs_new = qis_evs_new.bfill(axis=1)

        # check if values left (right) of rotation point are smaller (larger) than rotation point
        # check: is idx < left and value_at_idx > value_at_left --> then np.nan and ffill it for left (and bfill for right)
        for i in range(qis_evs_new.shape[0]):
            tmp = qis_evs_new.iloc[i, :].copy()
            left_bool = tmp[0:left_rotation_idx[i]] > rotation_points[i]
            right_bool = tmp[left_rotation_idx[i]:] < rotation_points[i]
            # change those evs on right that are smaller than rotation point
            tmp_right = tmp[left_rotation_idx[i]:]
            tmp_right[right_bool] = np.nan
            tmp_right.bfill(inplace=True) 
            tmp_right.ffill(inplace=True)  # in case the largest ev is NAN, we need to additionally ffill
            # change those evs on left that are LARGER than rotation point
            tmp_left = tmp[0:left_rotation_idx[i]]
            tmp_left[left_bool] = np.nan
            tmp_left.ffill(inplace=True) 
            tmp_left.bfill(inplace=True) # in case the smallest ev is NAN, we need to additionally bfill
            tmp_left.fillna(rotation_points[i], inplace=True)  # in case all eigenvalues left of rotation point are larger than rotation point
            # change qis_evs_new row
            qis_evs_new.iloc[i, :] = np.concatenate([tmp_left, tmp_right])

        # a few correction checks
        assert any(qis_evs_new.diff(axis=1).fillna(0) < 0), "Eigenvalues are not monotonically decreasing!"
        assert any(qis_evs_new.isna()), "There are NaN's in the QIS Eigenvalue Matrix!"
    
        return qis_evs_new

    intensity_of_intensity_list = np.arange(0.0, 2.01, 0.05).round(2)
    intensity_of_intensity_list = np.arange(0.0, 0.5, 0.05).round(2)
    print(f"Calculating new Eigenvalues.. for intensities in {intensity_of_intensity_list}")
    qis_evs_new = {}
    intensities_new = {}
    for intensity_of_intensity in intensity_of_intensity_list:
        qis_evs_new[intensity_of_intensity] = get_new_eigenvalues(evs_dfs["qis"], evs_dfs["sample"], intensity_of_intensity)
    print("Finished Calculating new Eigenvalues")

    import pickle
    base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
    # IMPORT SHRK DATASETS
    pf_size = PF_SIZE  # DONT CHANGE HERE!!
    permnos = pd.read_pickle(
        fr"{base_folder_path}\ONE_YR_long_only\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
    rets_full = pd.read_pickle(
        fr"{base_folder_path}\ONE_YR_long_only\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

    fixed_shrk_name = 'cov1Para'
    opt_shrk_name = 'cov1Para'
    with open(rf"{base_folder_path}\ONE_YR_long_only\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_{fixed_shrk_name}_p{pf_size}.pickle", 'rb') as f:
        fixed_shrk_data = pickle.load(f)
    with open(rf"{base_folder_path}\ONE_YR_long_only\preprocessing\training_dfs\PF{pf_size}\{opt_shrk_name}_factor-1.0_p{pf_size}.pickle", 'rb') as f:
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

    # %%

    print("tmp calculating now; may change the code as this calcs only for 0 to 0.5 int of int")
    tmp_rawres, tmp_res =  get_rawres(qis_evs_new, modelnames = intensity_of_intensity_list)
    all_res = pd.DataFrame(tmp_res.copy())
    all_rawres = pd.DataFrame(tmp_rawres.copy())
    out_path = r"H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\intensity_of_intensity_data"
    print("DONE")

    out_path = r"H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\intensity_of_intensity_data"
    all_res.to_csv(out_path + f"\\all_res_0_045_p{PF_SIZE}_v2.csv")
    all_rawres.to_csv(out_path + f"\\all_rawres_0_045_p{PF_SIZE}_v2.csv")
    
    if 1==2:
        print(f"creating res and rawres matrices for PF SIZE: {PF_SIZE}..")
        try:
            ioi_path = r"H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\intensity_of_intensity_data"
            all_res = pd.read_csv(ioi_path + f"\\all_res_p{PF_SIZE}_v2.csv", index_col=0)
            all_rawres = pd.read_csv(ioi_path + f"\\all_rawres_p{PF_SIZE}_v2.csv", index_col=0)
        except:
            tmp_rawres, tmp_res =  get_rawres(qis_evs_new, modelnames = intensity_of_intensity_list)
            all_res = pd.DataFrame(tmp_res.copy())
            all_rawres = pd.DataFrame(tmp_rawres.copy())
            out_path = r"H:\all\RL_Shrinkage_2024\ONE_YR_long_only\NonLinear_Shrinkage\intensity_of_intensity_data"
            all_res.to_csv(out_path + f"\\all_res_p{PF_SIZE}_v2.csv")
            all_rawres.to_csv(out_path + f"\\all_rawres_p{PF_SIZE}_v2.csv")

    print(f"Finished for pf size {PF_SIZE}!")

"""all_res_base  = pd.read_csv(ioi_path + f"\\all_res_p{PF_SIZE}_v2.csv", index_col=0)
all_rawres_base = pd.read_csv(ioi_path + f"\\all_rawres_p{PF_SIZE}_v2.csv", index_col=0)

all_res_full = pd.concat([all_res, all_res_base],axis=1)
all_rawres_full = pd.concat([all_rawres, all_rawres_base],axis=1)

all_res_full.to_csv(out_path + f"\\all_res_p{PF_SIZE}_v2.csv")
all_rawres_full.to_csv(out_path + f"\\all_rawres_p{PF_SIZE}_v2.csv")"""