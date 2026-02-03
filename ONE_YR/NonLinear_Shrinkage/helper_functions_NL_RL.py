import pandas as pd
import numpy as np
import pickle
import os

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6])


from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from ONE_YR.NonLinear_Shrinkage import regression_evaluation_funcs as re_hf

from helpers import eval_funcs_multi_target
from helpers import eval_funcs

def run_model_wrapper(
        num_ev,
        X,
        Y,
        model_func,
        all_rawres,
        all_factors,
        len_train=5040,
):

    model_preds = model_func(X, Y.to_numpy(), len_train)
    res = re_hf.map_preds_to_factors(model_preds, all_factors)
    Y_eval = all_rawres[num_ev]
    res_final = re_hf.evaluate_all_factor_preds(res, Y_eval, len_train)
    return res_final