import pandas as pd
import numpy as np
import pickle
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# plotting
import matplotlib.pyplot as plt
import psutil

psutil.cpu_count()
p = psutil.Process()
#p.cpu_affinity()  # get
p.cpu_affinity([0,1,2])


from helpers import eval_funcs, eval_function_new

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

base_folder_path = r'H:\\all\\RL_Shrinkage_2024'

# IMPORT SHRK DATASETS
shrk_data_path = None
pf_size = 500

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\1YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{base_folder_path}\1YR\preprocessing\training_dfs\PF{pf_size}\cov2Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

permnos = pd.read_pickle(
    fr"{base_folder_path}\1YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\1YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = fr"{base_folder_path}\helpers"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too

start_date = str(optimal_shrk_data['date'].iloc[0])
start_date = start_date[0:4] + '-' + start_date[4:6] + "-" + start_date[6:]
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]

# SET MANUALLY or implement a check if they already exist
new_inputs_folder = fr"{base_folder_path}\1YR\RL_CVC\new_inputs"
force_create = False
create_new_inputs = False
if not os.path.exists(new_inputs_folder + fr"\new_inputs_v2_p{pf_size}.csv") or force_create:
    create_new_inputs = True
if create_new_inputs == True:
    from preprocess_new_inputs_v2 import get_new_inputs
    new_inputs = get_new_inputs(rets_full, permnos)
    ni_df = pd.DataFrame.from_dict(new_inputs)
    ni_df.to_csv(new_inputs_folder + fr"\new_inputs_v2_p{pf_size}.csv", index=False)
    print("created new inputs")
else:
    new_inputs = pd.read_csv(new_inputs_folder + fr"\new_inputs_v2_p{pf_size}.csv").to_dict(orient='list')

## TEST
rolled_factors = True
if rolled_factors == True:
    factors = factors.rolling(21, 1).mean()

#### LOAD ADDITIONAL INPUTS AND CREATE INPUT VECTORS
cvc_shrk = optimal_shrk_data['shrk_factor'].values[5040:]
x1 = new_inputs['EW_Month_vola']
x2 = new_inputs['EW_Year_vola']
x3 = new_inputs['sample_covmat_trace']
x4 = optimal_shrk_data['shrk_factor'].values
x5 = new_inputs['allstocks_q075_month']
x6 = new_inputs['allstocks_q025_month']
x7 = new_inputs['allstocks_year_avgvola']
x8 = new_inputs['allstocks_month_avgvola']
x9 = factors
x10 = new_inputs['momentum_allstocks']
x11 = new_inputs['momentum_var_allstocks']

iqr = [i-j for i,j in zip(x5, x6)]
oracle = fixed_shrk_data.iloc[:, 2:].astype(float).idxmin(axis=1).astype(float)
cvc_full = optimal_shrk_data.shrk_factor.values

oracle_rollmeans = oracle.rolling(21).mean().fillna(0.3)[:-21]
oracle_rollmeans = pd.concat((oracle_rollmeans, pd.Series(0.3 for _ in range(21)) )).to_numpy()

X2 = np.vstack((x1, x3, x4, iqr, x10, x11, oracle_rollmeans.reshape(1, -1))).T
X2 = np.concatenate((X2, x9, oracle_rollmeans.reshape(-1,1)), axis=1)
#### DONE

#### SMOOTH CVC SHRINKAGES
smooth_cvc_shrkges = True
if smooth_cvc_shrkges == True:
    x4_smooth = []
    idces = np.where( ((x4[1:, ] - x4[:-1, ]) > 0.05) | (x4[1:] > 0.6))
    mean_no_outliers = np.mean(np.delete(x4, idces[0]+1))
    x4[idces[0]+1] = mean_no_outliers
    for i in range(x4.shape[0]):
        if i in (idces[0] + 1):
            x4_smooth.append( np.mean(x4[i - 100:i + 100] ))
        else:
            x4_smooth.append(x4[i])
    x4_smooth = np.array(x4_smooth).reshape(-1, 1)
    eval_function_new.myplot(x4_smooth)
    cvc_smooth = x4_smooth
#### DONE

len_train = 5040
end_date = fixed_shrk_data.shape[0]
Y = fixed_shrk_data.iloc[:, 3:].values
X_old = np.vstack((x1, x2, x3, x4)).T
X = np.vstack((x1, x2, x3, x4, x5, x6, x7, x8)).T
X = np.concatenate((X, x9), axis=1)

# temp here
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]



X = np.vstack((x1, x2, x3, x4, x5, x6, x7, x8, x10, x11)).T
X = np.concatenate((X, x9), axis=1)
# X = np.concatenate((X, pd.DataFrame(bias_factors)), axis=1 )

# Xtest and Ytest are then used for continuous model updates
Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y[0:len_train, :]
Ytest = Y[len_train:, :]

#### calculate Bias Factor as proposed by Gianluca
biasfactor_full_ds = np.mean(oracle) / np.mean(cvc_full)
biasfactor_half_ds = np.mean(oracle[:len_train]) / np.mean(cvc_full[:len_train])
cvc_bias_corrected = [c * biasfactor_full_ds if c*biasfactor_full_ds < 1 else 1 for c in cvc_full]
cvc_bias_corrected_hdf = [c * biasfactor_half_ds if c*biasfactor_half_ds < 1 else 1 for c in cvc_full]
# np.array(cvc_bias_corrected)[len_train:][val_idxes_shrkges]
### new cvc bias_corrected
# cvc_bias_corrected = [c * b if c*b < 1 else 1 for c,b in zip(cvc_full, bias_factors)]
#### DONE

# ADDITIONAL update cycles
w = Xtest.shape[0] // 21
preds_shrkg = []


###### CREATE BIAS AND USE AS INPUT
oracle_train = oracle[:len_train]
oracle_test = oracle[len_train:]
cvc_train = cvc_full[:len_train]
cvc_test = cvc_full[len_train:]
cvc_preds = cvc_test[val_idxes_shrkges]
bias_factors = []

for i in range(42):
    bias_factors.append(0.5)

for i in range(cvc_train.shape[0]-42):
    oracle_mean = np.mean(oracle_train.to_numpy()[21 + i: 42 + i])
    cvc_mean = np.mean(cvc_train[21 + i: 42 + i])
    bias_factor = oracle_mean / cvc_mean
    bias_factors.append(bias_factor)

for i in range(cvc_test.shape[0]):
    if i == 0 or i == 1:
        oracle_mean = np.mean(oracle_train.to_numpy()[-42 + i: oracle_train.shape[0] - 21 + i ])
        cvc_mean = np.mean(cvc_train[-42 + i: oracle_train.shape[0] - 21 + i ])
        bias_factor = oracle_mean / cvc_mean
        bias_factors.append(bias_factor)
    else:
        oracle_mean = np.mean(oracle_test.to_numpy()[0 + i - 2 : 21+(i-1)])
        cvc_mean = np.mean(cvc_test[0 + i - 2 : 21+(i-1)])
        bias_factor = oracle_mean / cvc_mean
        bias_factors.append(bias_factor)
########



##############################
# W/O FORWARD LOOKING BIAS and smaller train sizes
X = np.vstack((cvc_full, x1, x2, x3, x4, x5, x6, x7, x8, iqr, x10, x11, oracle_rollmeans.reshape(1, -1))).T
X = np.concatenate((X, x9), axis=1)  # x9 are the factors
X = np.concatenate((X, pd.DataFrame(bias_factors)), axis=1)
Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y[0:len_train, :]
Ytest = Y[len_train:, :]
preds_shrkg = []
for i in range(0, w):
    if i == 0:
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain[-21:, :], Ytrain[-21:, :])
        preds = regr.predict(Xtest[0, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1) / 100)[0])
    elif i == 1:
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain, Ytrain)
        preds = regr.predict(Xtest[21*i, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1) / 100)[0])
    else:
        Xtrain = np.concatenate((Xtrain, Xtest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        Ytrain = np.concatenate((Ytrain, Ytest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain, Ytrain)
        preds = regr.predict(Xtest[21*i, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1) / 100)[0])
r = eval_function_new.eval_fct_networkonly_1YR(np.array(preds_shrkg), rets_full, permnos, 0, val_indices_results)


##############################
# W/O FORWARD LOOKING BIAS but WITH NORMALIZED DATA
# x1,x2 EW month and year vola, x3 covmat trace
# x4 = optimal_shrk_facs, x5,x6 quartiles of allstocks volas
# x7, x8 allstocks volas,  x10,x11 momentum indicators
X = np.vstack((cvc_full, x1, x2, x3, x4, x5, x6, x7, x8, iqr, x10, x11, oracle_rollmeans.reshape(1, -1))).T
X = np.concatenate((X, x9), axis=1)  # x9 are the factors
X = np.concatenate((X, pd.DataFrame(bias_factors)), axis=1)
#X = X[:, [1,2,5,6,10,11,12,13]]
#X = np.concatenate((X, x9), axis=1)
Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y[0:len_train, :]
Ytest = Y[len_train:, :]
preds_shrkg = []
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
train_size = 5000
for i in range(0, w):
    if i == 0:
        x = Xtrain[-train_size:-21, :]
        sc.fit(x)
        x = sc.transform(x)
        #weights = [1 + i / 500 for i in range(x.shape[0])]
        weights = [1 for i in range(x.shape[0])]
        weights[-1000:] = [2 for i in range(1000)]
        weights[-500:] = [4 for i in range(500)]
        x_test = sc.transform(Xtest[0, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x, Ytrain[-train_size:-21, :], sample_weight=np.array(weights) )
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1) / 100)[0])
    elif i == 1:
        x = Xtrain[-train_size:]
        sc.fit(x)
        x = sc.transform(x)
        #weights = [1 + i / 1000 for i in range(x.shape[0])]
        weights = [1 for i in range(x.shape[0])]
        weights[-1000:] = [2 for i in range(1000)]
        weights[-500:] = [4 for i in range(500)]
        x_test = sc.transform(Xtest[21*i, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x, Ytrain[-train_size: ], sample_weight=np.array(weights))
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1) / 100)[0])
    else:
        if i == 2: # just once because we overwrite Xtrain
            Xtrain = Xtrain[-train_size: ]
            Ytrain = Ytrain[-train_size: ]
        Xtrain = np.concatenate((Xtrain, Xtest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        Ytrain = np.concatenate((Ytrain, Ytest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        x = Xtrain
        sc.fit(x)
        x = sc.transform(x)
        #weights = [1 + i / 1000 for i in range(x.shape[0])]
        weights = [1 for i in range(x.shape[0])]
        weights[-1000:] = [2 for i in range(1000)]
        weights[-500:] = [4 for i in range(500)]
        x_test = sc.transform(Xtest[21*i, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain, Ytrain, sample_weight=np.array(weights))
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1) / 100)[0])
r = eval_function_new.eval_fct_networkonly_1YR(np.array(preds_shrkg), rets_full, permnos, 0, val_indices_results)
######## DONE

# compare to benchmarks
# eval_function_new.eval_fct_new_1YR([0.3 for i in range(len(val_indices_results))], rets_full, permnos, 0, val_indices_results)


