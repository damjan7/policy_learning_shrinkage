import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier

# note, paths are hard coded in load_additional_train_data
def load_additional_train_data(
        pf_size, 
        opt_values_factors,
        include_ew_month_vola=False,
        include_ew_year_vola=False,
        include_sample_covmat_trace=False,
        include_allstocks_year_avgvola=False,
        include_allstocks_month_avgvola=False,
        include_factors=False,
        include_ts_momentum_allstocks=False,
        include_ts_momentum_var_allstocks=False,
        include_ewma_year=False,
        include_ewma_month=False,
        include_mean_of_correls=False,
        include_iqr=False,
        additional_inputs=None
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
    if opt_values_factors is not None:
        inputs.append(opt_values_factors.tolist())

    if include_ewma_month==True:
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
    if  include_ew_year_vola == True:
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
    if include_factors==True:
        x9 = factors
        X = np.concatenate((X, x9), axis=1)

    if additional_inputs is not None:
        len_add_inputs = len(additional_inputs)
        additional_inputs = np.column_stack(additional_inputs)
        new = additional_inputs.reshape(10353, -1)
        X = np.concatenate((X, new), axis=1)

    return X


def load_additional_train_data_test(
        pf_size,
        all_rawres,
        include_ew_month_vola=False,
        include_ew_year_vola=False,
        include_sample_covmat_trace=False,
        include_allstocks_year_avgvola=False,
        include_allstocks_month_avgvola=False,
        include_factors=False,
        include_ts_momentum_allstocks=False,
        include_ts_momentum_var_allstocks=False,
        include_ewma_year=False,
        include_ewma_month=False,
        include_mean_of_correls=False,
        include_iqr=False

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

    if all_rawres is not None:
        new = []
        for i in range(all_rawres.shape[0] // 21):
            tmp = all_rawres.iloc[0 + 21 * i: 21 * (i + 1), :]
            tmp = tmp.std()
            new.append(tmp)
            
        new = np.array(new)
        new2 = np.repeat(new, 21, axis=0)
        X = np.concatenate((X, np.array(new2)), axis=1)
    return X

from sklearn.preprocessing import StandardScaler
def basic_multi_output_elastic_net_Lagged(X, Y, len_train, scale=False):
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
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x_train, y_train)
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

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

def basic_multi_output_elastic_net_NonLagged_testV1(X, Y, len_train, scale=False):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*(i) : len_train + 21*(i-1) - 1, :]
            y_train = Y[21*(i+1) : len_train + 21*i - 1, :]
            x_test = X[len_train + 21*i - 1, :].reshape(1, -1)
            #x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x_train, y_train)
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.argmin())


def basic_multi_output_elastic_net_NonLagged_OnlyRebdates(X, Y, len_train, scale=False):
    model_predictions = []
    for i in range(10353 // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > 10353:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[i:240+i, :]
            y_train = Y[i:240+i, :]
            x_test = np.ascontiguousarray(X[240 + i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x_train, y_train)
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.argmin())

def basic_multi_output_elastic_net_NonLagged_old(X, Y, len_train, scale=False):
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
            #x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x_train, y_train)
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())


def basic_multi_output_elastic_net_new(X, Y, len_train, scale=False):
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
            regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x_train, y_train)
            preds = regr.predict(x_test)
            model_predictions.append((preds.argmin(axis=1))[0])

def basic_multi_output_elastic_net_Lagged_version2(X, Y, len_train):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*i : len_train + 21*(i-1), :]
            y_train = Y[21*i : len_train + 21*(i-1), :]
            x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            regr = MultiOutputRegressor(
                ElasticNet(random_state=123)
            )
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def basic_multi_output_elastic_net_Lagged_version3(X, Y, len_train):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if i % 10 == 0:
                x_train = X[21*i : len_train + 21*(i-1), :]
                y_train = Y[21*i : len_train + 21*(i-1), :]
                x_test = X[len_train + 21*(i-1):len_train + 21*i, :]
                regr = MultiOutputRegressor(
                    ElasticNet(random_state=123)
                )
                regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def basic_multi_output_elastic_net_Lagged_Expanding(X, Y, len_train):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if i % 1 == 0:
                x_train = X[0  : len_train + 21*(i-1), :]
                y_train = Y[0 : len_train + 21*(i-1), :]
                x_test = X[len_train + 21*(i-1):len_train + 21*i, :]
                regr = MultiOutputRegressor(
                    ElasticNet(random_state=123)
                )
                regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

from lightgbm import LGBMRegressor
def basic_multi_output_LGBM(X, Y, len_train, scale=False):
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
            regr = MultiOutputRegressor(LGBMRegressor(random_state=123)).fit(x_train, y_train)
            preds = regr.predict(x_test)
            model_predictions.append(preds.mean(axis=0).argmin())


def basic_multi_output_LGBM_v2(X, Y, len_train, cur_params):
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
            #x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, **cur_params, verbose=-1)
            )
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def basic_multi_output_LGBM_NonLagged_testV1(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*(i) - 1 : len_train + 21*(i-1) - 1, :]
            y_train = Y[21*(i) + 1: len_train + 21*(i-1), :]
            x_test = X[len_train + 21*i - 1, :].reshape(1, -1)
            #x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, **cur_params, verbose=-1)
            )
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.argmin())

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


def basic_multi_output_LGBM_Lagged_version2(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*i : len_train + 21*(i-1), :]
            y_train = Y[21*i : len_train + 21*(i-1), :]
            x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, **cur_params, verbose=-1)
            )
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def basic_multi_output_LGBM_Lagged_version3(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if 1 == 1:
                x_train = X[21*i : len_train + 21*(i-1), :]
                y_train = Y[21*i : len_train + 21*(i-1), :]
                regr = MultiOutputRegressor(
                    LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                )
                regr = regr.fit(x_train, y_train)
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def basic_multi_output_LGBM_Lagged_Classification(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if 1 == 1:
                x_train = X[21*i : len_train + 21*(i-1), :]
                y_train = Y[21*i : len_train + 21*(i-1),]
                regr = LGBMClassifier(random_state=123, **cur_params, verbose=-1)

                regr = regr.fit(x_train, y_train)
            x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def basic_multi_output_LGBM_Lagged_Expanding_Classification(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if 1 == 1:
                x_train = X[0 : len_train + 21*(i-1), :]
                y_train = Y[0 : len_train + 21*(i-1),]
                regr = LGBMClassifier(random_state=123, **cur_params, verbose=-1)

                regr = regr.fit(x_train, y_train)
            x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def basic_multi_output_LGBM_Lagged_Expanding(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if 1 == 1:
                x_train = X[0 : len_train + 21*(i-1), :]
                y_train = Y[0 : len_train + 21*(i-1), :]
                regr = MultiOutputRegressor(
                    LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                )
                regr = regr.fit(x_train, y_train)
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def basic_multi_output_LGBM_Lagged_SingleTrain(X, Y, len_train, cur_params):
    #assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if i == 0:
                x_train = X[21*i : len_train + 21*(i-1), :]
                y_train = Y[21*i : len_train + 21*(i-1), ]
                regr = MultiOutputRegressor(
                    LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                )
                regr = regr.fit(x_train, y_train)

            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.mean(axis=0).argmin())

def general_single_output_LGBMRegression_Lagged(X, Y, len_train, cur_params, single_train=False, expanding=False, train_size=None):
    model_predictions = []
    if train_size is None:
        train_size = len_train
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else: # TRAINING
            if single_train == False:
                if expanding == False:
                    x_train = X[len_train - train_size + 21*i : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size + 21*i : len_train + 21*(i-1), ]
                else:
                    x_train = X[len_train - train_size : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size : len_train + 21*(i-1), ]
                regr = LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                regr = regr.fit(x_train, y_train)
            else:
                if i == 0:
                    x_train = X[len_train - train_size + 21 * i: len_train + 21 * (i - 1), :]
                    y_train = Y[len_train - train_size + 21 * i: len_train + 21 * (i - 1), ]
                    regr = LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                    regr = regr.fit(x_train, y_train)
            ## TRAINING DONE
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            '''
            if preds < 0:
                preds = 0
            if preds > 30:
                preds = 30
            else:
                pass
            '''
            preds = int(np.round(preds, 0))

            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

'''
def general_single_output_LGBMRegression_Lagged(X, Y, len_train, cur_params, single_train=False, expanding=False, train_size=None):
    model_predictions = []
    if train_size is None:
        train_size = len_train
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else: # TRAINING
            if single_train == False:
                if expanding == False:
                    x_train = X[len_train - train_size + 21*i : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size + 21*i : len_train + 21*(i-1), ]
                else:
                    x_train = X[len_train - train_size : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size : len_train + 21*(i-1), ]
                regr = LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                regr = regr.fit(x_train, y_train)
            else:
                if i == 0:
                    x_train = X[len_train - train_size + 21 * i: len_train + 21 * (i - 1), :]
                    y_train = Y[len_train - train_size + 21 * i: len_train + 21 * (i - 1), ]
                    regr = LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                    regr = regr.fit(x_train, y_train)
            ## TRAINING DONE
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            if preds < 0:
                preds = 0
            if preds > 100:
                preds = 100
            else:
                preds = int(np.round(preds, 0))

            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)
'''

def general_single_output_LGBMRegression_Lagged_TEST(X, Y, cur_params, len_train=240):
    model_predictions = []
    idxes = [i for i in range(0, 10353, 21)]
    X = X[idxes]
    Y = Y[idxes]
    for i in range(5313 // 21):  # is too long which is why we have a if clause
        if i ==  (5313 // 21) - 1:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else: # TRAINING
            x_train = X[ 0 + i : len_train + (i-1), :]
            y_train = Y[ 0 + i : len_train + (i-1), ]
            regr = LGBMRegressor(random_state=123, **cur_params, verbose=-1)
            regr = regr.fit(x_train, y_train)

            ## TRAINING DONE
            x_test = np.ascontiguousarray(X[len_train + i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            if preds < 0:
                preds = 0
            if preds > 101:
                preds = 101
            else:
                preds = int(np.round(preds, 0))

            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)


def general_single_output_ElasticNet_Lagged_old(X, Y, len_train, single_train=False, expanding=False):
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else: # TRAINING
            if single_train == False:
                if expanding == False:
                    x_train = X[21*i : len_train + 21*(i-1), :]
                    y_train = Y[21*i : len_train + 21*(i-1), ]
                else:
                    x_train = X[0 : len_train + 21*(i-1), :]
                    y_train = Y[0 : len_train + 21*(i-1), ]
                regr = ElasticNet(random_state=123)
                regr = regr.fit(x_train, y_train)
            else:
                if i == 0:
                    x_train = X[21 * i: len_train + 21 * (i - 1), :]
                    y_train = Y[21 * i: len_train + 21 * (i - 1), ]
                    regr = LGBMRegressor(random_state=123)
                    regr = regr.fit(x_train, y_train)
            ## TRAINING DONE
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            if preds < 0:
                preds = 0
            if preds > 100:
                preds = 100
            else:
                preds = int(np.round(preds, 0))

            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def general_single_output_ElasticNet_Lagged(X, Y, len_train, single_train=False, expanding=False, train_size=None, model_params={} ):
    """
    contains the lag of 21*(i-1) in the training data since the target values
    contain information of the future!
    """
    model_predictions = []
    if train_size is None:
        train_size = len_train
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:  # TRAINING
            if single_train == False:
                if expanding == False:
                    x_train = X[len_train - train_size + 21*i : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size + 21*i : len_train + 21*(i-1), ]
                else:
                    x_train = X[len_train - train_size : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size : len_train + 21*(i-1), ]
                regr = ElasticNet(random_state=123, **model_params)
                regr = regr.fit(x_train, y_train)
            else:
                if i == 0:
                    x_train = X[len_train - train_size + 21 * i: len_train + 21 * (i - 1), :]
                    y_train = Y[len_train - train_size + 21 * i: len_train + 21 * (i - 1), ]
                    regr = ElasticNet(random_state=123, **model_params)
                    regr = regr.fit(x_train, y_train)
            ## TRAINING DONE
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)

            preds = int(np.round(preds, 0))

            #print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def general_single_output_RF_Lagged(X, Y, len_train, single_train=False, expanding=False, train_size=None,
                                    model_params={
                                        'n_estimators':30,
                                        'max_depth': 10
                                    } ):

    model_predictions = []
    if train_size is None:
        train_size = len_train
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:  # TRAINING
            if single_train == False:
                if expanding == False:
                    x_train = X[len_train - train_size + 21*i : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size + 21*i : len_train + 21*(i-1), ]
                else:
                    x_train = X[len_train - train_size : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size : len_train + 21*(i-1), ]
                regr = RandomForestRegressor(random_state=123, **model_params)
                regr = regr.fit(x_train, y_train)
            else:
                if i == 0:
                    x_train = X[len_train - train_size + 21 * i: len_train + 21 * (i - 1), :]
                    y_train = Y[len_train - train_size + 21 * i: len_train + 21 * (i - 1), ]
                    regr = RandomForestRegressor(random_state=123, **model_params)
                    regr = regr.fit(x_train, y_train)
            ## TRAINING DONE
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)

            preds = int(np.round(preds, 0))

            #print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def general_single_output_last_optimal_Lagged(X, Y, len_train, single_train=False, expanding=False, train_size=None,
                                    model_params={} ):

    model_predictions = []
    if train_size is None:
        train_size = len_train
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:  # TRAINING
            if single_train == False:
                if expanding == False:
                    x_train = X[len_train - train_size + 21*i : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size + 21*i : len_train + 21*(i-1), ]
                else:
                    x_train = X[len_train - train_size : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size : len_train + 21*(i-1), ]
            else:
                if i == 0:
                    x_train = X[len_train - train_size + 21 * i: len_train + 21 * (i - 1), :]
                    y_train = Y[len_train - train_size + 21 * i: len_train + 21 * (i - 1), ]
            ## TRAINING DONE
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))

            preds = y_train[-1]
            preds = int(np.round(preds, 0))

            #print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def general_single_output_ElasticNet_Lagged_CrossValidation(X, Y, len_train, single_train=False, expanding=False, train_size=None, model_params={}):
    model_predictions = []
    if train_size is None:
        train_size = len_train
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:  # TRAINING
            if single_train == False:
                if expanding == False:
                    x_train = X[len_train - train_size + 21*i : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size + 21*i : len_train + 21*(i-1), ]
                else:
                    x_train = X[len_train - train_size : len_train + 21*(i-1), :]
                    y_train = Y[len_train - train_size : len_train + 21*(i-1), ]
                regr = ElasticNet(random_state=123, **model_params)
                regr = regr.fit(x_train, y_train)
            else:
                if i == 0:
                    x_train = X[len_train - train_size + 21 * i: len_train + 21 * (i - 1), :]
                    y_train = Y[len_train - train_size + 21 * i: len_train + 21 * (i - 1), ]
                    regr = ElasticNet(random_state=123, **model_params)
                    regr = regr.fit(x_train, y_train)
            ## TRAINING DONE
            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)

            preds = int(np.round(preds, 0))

            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def basic_multi_output_LGBMClassification_Lagged_SingleTrain(X, Y, len_train, cur_params):
    #assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            if i == 0:
                x_train = X[21*i : len_train + 21*(i-1), :]
                y_train = Y[21*i : len_train + 21*(i-1), ]
                regr = LGBMClassifier(random_state=123, **cur_params, verbose=-1)
                regr = regr.fit(x_train, y_train)

            x_test = X[len_train + 21 * (i - 1):len_train + 21 * i, :]
            x_test = np.ascontiguousarray(X[len_train + 21 * i, :].reshape(1, -1))
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds)

def basic_multi_output_LGBM_NonLagged_OneDayPred(X, Y, len_train, cur_params):
    assert type(Y) == np.ndarray and type(X) == np.ndarray, "Y or X are not a np ndarray"
    model_predictions = []
    for i in range(X.shape[0] // 21):  # is too long which is why we have a if clause
        if len_train + 21*(i+1) > X.shape[0]:
            model_predictions = np.repeat(model_predictions, 21)
            return model_predictions
        else:
            x_train = X[21*(i+1)-1 : len_train + 21*i-1, :]
            y_train = Y[21*(i+1) : len_train + 21*i, :]
            x_test = np.ascontiguousarray(X[len_train + 21*i -1, :].reshape(1, -1))
            #x_test = np.ascontiguousarray(X[len_train + 21*i, :].reshape(1, -1))
            #y_test = Y[len_train + 21*i : len_train + 21*(i+1)]
            regr = MultiOutputRegressor(
                LGBMRegressor(random_state=123, **cur_params, verbose=-1)
            )
            regr = regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            print(f"fitted model in iteration {i} out of {5313//21}")
            model_predictions.append(preds.argmin())


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
                    LGBMRegressor(random_state=123)
                )
                regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            model_predictions.append(preds.std(axis=0).argmin())

def basic_multi_output_LGBM_yearly_training(X, Y, len_train, cur_params):
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
            if i%10==0: # i.e. train model one single time
                regr = MultiOutputRegressor(
                    LGBMRegressor(random_state=123, **cur_params, verbose=-1)
                )
                regr.fit(x_train, y_train)
            preds = regr.predict(x_test)
            model_predictions.append(preds.std(axis=0).argmin())




def map_preds_to_factors(model_preds, all_factors):
    argmin_to_factor_mapping = {}
    for i, f in enumerate(all_factors):
        argmin_to_factor_mapping[i] = str(f)
    model_preds = list(map(lambda x: argmin_to_factor_mapping[x], model_preds))
    return model_preds

def map_factors_to_preds(preds_as_floats, all_factors):
    all_nums = np.arange(len(all_factors))
    mapping = {}
    for i, f in enumerate(all_factors):
        mapping[f] = all_nums[i]
    model_preds = list(map(lambda x: mapping[x], preds_as_floats))
    return model_preds


def evaluate_all_factor_preds(model_preds, Y_eval, len_train):
    '''
    Tested function for model_preds = '1.0' for all idces --> same as QIS
    '''
    assert len_train == Y_eval.shape[0] - len(model_preds), "check inputs to the function!"
    Y_eval = Y_eval.iloc[len_train:, :]
    returns = np.diag(Y_eval.loc[:, model_preds])
    returns_sd = returns.std() * np.sqrt(252) * 100
    returns_means = returns.mean() * 252 * 100
    return round(returns_means, 3), round(returns_sd, 3)