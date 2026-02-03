from helpers import helper_functions as hf
from helpers import rl_covmat_ests_for_dataset as estimators
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quick_eval_val(len_train, end_date, mapped_shrkges, val_dataset, rets_full, permnos, plot=False, calc_res=False):
    val_indices_correct = (len_train, end_date)
    val_indices_results = [val_indices_correct[0] + 21 * i for i in
                           range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
    val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]

    mapped_shrkges_v2 = np.array(mapped_shrkges)[val_idxes_shrkges]
    cvc_shrk = val_dataset.optimal_shrk_data['shrk_factor']

    if plot == True:
        myplot(cvc_shrk, mapped_shrkges)
    if calc_res == True:
        val_eval = eval_fct_networkonly_1YR(mapped_shrkges_v2, rets_full, permnos, 0, val_indices_results)
        return val_eval

def quick_eval_train(len_train, end_date, train_preds, train_dataset, rets_full, permnos, calc_res = False):
    train_shrkges = [i / 100 for i in train_preds]
    myplot(train_dataset.optimal_shrk_data['shrk_factor'], train_shrkges)

    train_indices_correct = (0, len_train)
    train_indices_results = [train_indices_correct[0] + 21 * i for i in
                             range((train_indices_correct[-1] - train_indices_correct[0]) // 21)]
    train_idxes_shrkges = [0 + 21 * i for i in range((train_indices_correct[-1] - train_indices_correct[0]) // 21)]

    if calc_res == True:
        train_eval = eval_fct_new_1YR(np.array(train_shrkges)[train_idxes_shrkges], rets_full, permnos, 0,
                                                    train_indices_results)
        return train_eval

def eval_fct_new_1YR(val_preds, rets_full, permnos, reb_days, val_indices):
    weighted_rets_model = []
    weighted_rets_cov2para = []
    weighted_rets_qis = []
    sample_covmat_only = []

    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]

    for i in range(len(val_indices)):
        idx = val_indices[i]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
        #fut_ret_mat = fut_ret_mat.sub(fut_ret_mat.mean())

        shrk, sample, target = estimators.cov2Para(past_ret_mat)
        model_covmat_est = val_preds[i] * target + (1-val_preds[i]) * sample
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        sample_covmat = sample
        qis_covmat_est = estimators.QIS(past_ret_mat)

        weights_model = hf.calc_global_min_variance_pf(model_covmat_est)
        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)
        weights_sample = hf.calc_global_min_variance_pf(sample_covmat)

        weighted_rets_model += list(fut_ret_mat @ weights_model)
        weighted_rets_cov2para += list(fut_ret_mat @ weights_cov2para)
        weighted_rets_qis += list(fut_ret_mat @ weights_qis)
        sample_covmat_only += list(fut_ret_mat @ weights_sample)

    res = results = {
        "network pf return" : round(np.mean(weighted_rets_model) * 252 * 100, 2) ,
        "cov2para pf return" : round(np.mean(weighted_rets_cov2para) * 252 * 100, 2) ,
        "qis pf return" : round(np.mean(weighted_rets_qis) * 252 * 100, 2) ,
        "sample covmat return": round(np.mean(sample_covmat_only) * 252 * 100, 2) ,
        "network pf sd" : round(np.std(weighted_rets_model) * np.sqrt(252) *100, 6) ,
        "cov2para pf sd" : round(np.std(weighted_rets_cov2para) * np.sqrt(252) *100, 2) ,
        "qis pf sd" : round(np.std(weighted_rets_qis) * np.sqrt(252) *100, 2) ,
        "sample covmat sd": round(np.std(sample_covmat_only) * np.sqrt(252) *100, 2) ,
    }
    return res


def eval_fct_new_5YR(val_preds, rets_full, permnos, reb_days, val_indices):
    weighted_rets_model = []
    weighted_rets_cov2para = []
    weighted_rets_qis = []
    sample_covmat_only = []

    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]

    for i in range(len(val_indices)):
        idx = val_indices[i]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*5: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
        #fut_ret_mat = fut_ret_mat.sub(fut_ret_mat.mean())

        shrk, sample, target = estimators.cov2Para(past_ret_mat)
        model_covmat_est = val_preds[i] * target + (1-val_preds[i]) * sample
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        sample_covmat = sample
        qis_covmat_est = estimators.QIS(past_ret_mat)

        weights_model = hf.calc_global_min_variance_pf(model_covmat_est)
        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)
        weights_sample = hf.calc_global_min_variance_pf(sample_covmat)

        weighted_rets_model += list(fut_ret_mat @ weights_model)
        weighted_rets_cov2para += list(fut_ret_mat @ weights_cov2para)
        weighted_rets_qis += list(fut_ret_mat @ weights_qis)
        sample_covmat_only += list(fut_ret_mat @ weights_sample)

    res = results = {
        "network pf return" : round(np.mean(weighted_rets_model) * 252 * 100, 2) ,
        "cov2para pf return" : round(np.mean(weighted_rets_cov2para) * 252 * 100, 2) ,
        "qis pf return" : round(np.mean(weighted_rets_qis) * 252 * 100, 2) ,
        "sample covmat return": round(np.mean(sample_covmat_only) * 252 * 100, 2) ,
        "network pf sd" : round(np.std(weighted_rets_model) * np.sqrt(252) *100, 6) ,
        "cov2para pf sd" : round(np.std(weighted_rets_cov2para) * np.sqrt(252) *100, 2) ,
        "qis pf sd" : round(np.std(weighted_rets_qis) * np.sqrt(252) *100, 2) ,
        "sample covmat sd": round(np.std(sample_covmat_only) * np.sqrt(252) *100, 2) ,
    }
    return res

def myplot(*args):
    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(args[0]))
    for arg in args:
        ax.plot(x, arg)
    plt.legend()
    plt.show()


def eval_fixed_shrkges(shrkges, val_indices, rets_full, permnos):
    res_full = {}
    for shrk in shrkges:
        val_preds = [shrk for _ in range(len(val_indices))]
        res = eval_fct_new_5YR(val_preds, rets_full, permnos, 0, val_indices)
        res_full[f'{shrk}'] = res['network pf sd']

    return res_full


def get_n_smallest_indices(nparray, n):
    min_indices = np.argpartition(nparray, n - 1)[:n]
    return min_indices


def eval_fct_networkonly_1YR(val_preds, rets_full, permnos, reb_days, val_indices):
    weighted_rets_model = []
    reb_date_1 = permnos.index[0]
    add_idx = np.where(rets_full.index == reb_date_1)[0][0]

    for i in range(len(val_indices)):
        idx = val_indices[i]
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]

        shrk, sample, target = estimators.cov2Para(past_ret_mat)
        model_covmat_est = val_preds[i] * target + (1-val_preds[i]) * sample

        weights_model = hf.calc_global_min_variance_pf(model_covmat_est)
        weighted_rets_model += list(fut_ret_mat @ weights_model)

    res = results = {
        "network pf return" : round(np.mean(weighted_rets_model) * 252 * 100, 2) ,
        "network pf sd" : round(np.std(weighted_rets_model) * np.sqrt(252) *100, 6) ,
    }
    return res


def plot_and_save(plot_title, xlabel, ylabel, *data, outpath=None, show=False):
    import matplotlib.pyplot as plt
    import matplotlib
    font = {'size': 16}
    matplotlib.rc('font', **font)

    x_labels = [2000] * 39 + [2004] * 39 + [2008] * 39 + [2012] * 39 + [2016] * 39 + [2020] * 39 + [2022] * 20

    plt.figure().set_figwidth(12)

    x = [i for i in range(len(data[0][0]))]
    for d in data:
        plt.plot(x, d[0], label=d[1])

    # Add title and labels to axes
    plt.title(f'{plot_title}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')

    # custom label ticks
    #selected_ticks = [x[10], x[49], x[88], x[127], x[166], x[205], x[244]]
    #seletced_labels = [x_labels[10], x_labels[49], x_labels[88], x_labels[127], x_labels[166], x_labels[205],
    #                   x_labels[244]]
    #plt.xticks(selected_ticks, seletced_labels)

    # Add a legend
    plt.legend()

    # Show the plot
    if show == True:
        plt.show()
    else:
        out_path = r"C:\Users\kostovda\OneDrive - Luzerner Kantonsspital\Anlagen\code\V5\1YR\PLOTS"
        plt.savefig(out_path + f'/plot_XYXY.svg', format='svg', bbox_inches='tight')