import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor


# note, paths are hard coded in load_additional_train_data
def load_additional_train_data(
        pf_size,
        opt_shrinkage_intensities,
        include_ew_month_vola=True,
        include_ew_year_vola=True,
        include_sample_covmat_trace=True,
        include_allstocks_year_avgvola=True,
        include_allstocks_month_avgvola=True,
        include_factors=True,
        include_ts_momentum_allstocks=True,
        include_ts_momentum_var_allstocks=True,
        include_ewma_year=True,
        include_ewma_month=True,
        include_mean_of_correls=True,
        include_iqr=True

):
    base_folder_path = r'H:\\all\\RL_Shrinkage_2024'
    factor_path = fr"{base_folder_path}\helpers"
    factors = pd.read_csv(factor_path + "/all_factors.csv")
    factors = factors.pivot(index="date", columns="name", values="ret")
    start_date = '19810112'  # = str(optimal_shrk_data['date'].iloc[0])
    start_date = start_date[0:4] + '-' + start_date[4:6] + "-" + start_date[6:]
    start_idx = np.where(factors.index == start_date)[0][0]
    factors = factors.iloc[start_idx:start_idx + 10353, :]

    new_inputs_folder = fr"{base_folder_path}\ONE_YR\RL_CVC\new_inputs"
    new_inputs = pd.read_csv(new_inputs_folder + fr"\new_inputs_v2_p{pf_size}.csv").to_dict(orient='list')

    inputs = []
    if opt_shrinkage_intensities is not None:
        inputs.append(opt_shrinkage_intensities.tolist())

    if include_ewma_month == True:
        x1 = new_inputs['EW_Month_vola']
        inputs.append(x1)
    if include_ewma_year == True:
        x2 = new_inputs['EW_Year_vola']
        inputs.append(x2)
    if include_sample_covmat_trace == True:
        x3 = new_inputs['sample_covmat_trace']
        inputs.append(x3)
    if include_allstocks_year_avgvola == True:
        x7 = new_inputs['allstocks_year_avgvola']
        inputs.append(x7)
    if include_allstocks_month_avgvola == True:
        x8 = new_inputs['allstocks_month_avgvola']
        inputs.append(x8)
    if include_ts_momentum_allstocks == True:
        x10 = new_inputs['momentum_allstocks']
        inputs.append(x10)
    if include_ts_momentum_var_allstocks == True:
        x11 = new_inputs['momentum_var_allstocks']
        inputs.append(x11)
    if include_ew_year_vola == True:
        x12 = new_inputs['EW_ewma_year']
        inputs.append(x12)
    if include_ew_month_vola == True:
        x13 = new_inputs['EW_ewma_month']
        inputs.append(x13)
    if include_mean_of_correls == True:
        x14 = new_inputs['mean_of_correls']
        inputs.append(x14)
    if include_iqr == True:
        x5 = new_inputs['allstocks_q075_month']
        x6 = new_inputs['allstocks_q025_month']
        iqr = [i - j for i, j in zip(x5, x6)]
        inputs.append(iqr)

    X = np.array(inputs).T
    if include_factors == True:
        x9 = factors
        X = np.concatenate((X, x9), axis=1)
    return X

def basic_multi_output_elastic_net_NonLagged(X, Y, len_train, scale=False):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*i : len_train + 21*i, :]
            y_train = Y[21*i : len_train + 21*i, :]
            x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x_train, y_train)
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def basic_multi_output_LGBM_NonLagged(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*i : len_train + 21*i, :]
            y_train = Y[21*i : len_train + 21*i, :]
            x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, **cur_params, verbose=-1)
            )
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())


def basic_multi_output_LGBM_single_training(X, Y, len_train):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    train_size = 5040
    t = train_size - len_train
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else: #train model
            x_train = X[21*i + t : len_train + 21*i, :]
            y_train = Y[21*i + t : len_train + 21*i, :]
            x_test = X[len_train + 21*i : len_train + 21*(i+1), :]
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            if i==0: # i.e. train model one single time
                regr = MultiOutputRegressor(
                    LGBMRegressor(random_state=123, verbose=-1)
                )
                regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            model_predictions.append((preds.argmin(axis=1))[0])

from lightgbm import LGBMRegressor
def basic_multi_output_LGBM(X, Y, len_train, scale=False):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*i : len_train + 21*(i-1), :]
            y_train = Y[21*i : len_train + 21*(i-1), :]
            x_test = X[len_train + 21*i : len_train + 21*(i+1), :]
            #x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(LGBMRegressor(random_state=123, verbose=-1,
                                                      num_leaves=2, max_depth=2, learning_rate=0.5, n_estimators=2,
                                                      )).fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313 // 21}")
            model_predictions.append(preds.std(axis=0).argmin())
            if i == X.shape[0] // 21 - 1:
                print("y")

def basic_multi_output_LGBM_test(X, Y, len_train, scale=False):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*i : len_train + 21*i, :]
            y_train = Y[21*i : len_train + 21*i, :]
            x_test = X[len_train + 21*i : len_train + 21*(i+1), :]
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            if i // 21 == 0:
                regr = MultiOutputRegressor(LGBMRegressor(random_state=123, verbose=-1)).fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313 // 21}")
            model_predictions.append(preds.std(axis=0).argmin())
