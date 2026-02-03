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
p.cpu_affinity([0,1,2,3,4,5])


from helpers import eval_funcs, eval_function_new

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

base_folder_path = r'/'

# IMPORT SHRK DATASETS
shrk_data_path = None
pf_size = 500


permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = fr"{base_folder_path}\helpers"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too
fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\cov2Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

start_date = str(optimal_shrk_data['date'].iloc[0])
start_date = start_date[0:4] + '-' + start_date[4:6] + "-" + start_date[6:]
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]

############################# EVALUATION HELPER FUNCTION
def eval_triplet_preds(chosen_model_lst, qis_res_test):
    # chosen model a list with numbers corresp. to the chosen model
    # base_qis_res = round(np.std(qis_res_test['base']) * np.sqrt(252) * 100, 2)
    mod_rets = []
    base_qis_res_v2 = []
    for idx, i in enumerate(range(0, permnos.shape[0]-len_train, 21)):
        j=i+21
        mod_rets += list( (qis_res_test.iloc[i:j, chosen_model_lst[idx]]).to_numpy() )
        base_qis_res_v2 += list( (qis_res_test['base'].iloc[i:j, ]).to_numpy() )
    r1 = round(np.std(mod_rets) * np.sqrt(252) * 100, 2)
    r2 = round(np.std(base_qis_res_v2) * np.sqrt(252) * 100, 2)
    return r1, r2
#############################3

# SET MANUALLY or implement a check if they already exist
new_inputs_folder = fr"{base_folder_path}\ONE_YR\RL_CVC\new_inputs"
force_create = False
create_new_inputs = False
if not os.path.exists(new_inputs_folder + fr"\new_inputs_v2_p{pf_size}.csv") or force_create:
    create_new_inputs = True
if create_new_inputs == True:
    from ONE_YR.RL_CVC.preprocess_new_inputs_v2 import get_new_inputs
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
x12 = new_inputs['EW_ewma_year']
x13 = new_inputs['EW_ewma_month']
x14 = new_inputs['mean_of_correls']

#qis_results looks into the future
#i.e. datapoint 1 (t=1) can only be used from time t=21
qis_results = pd.read_csv(r"/ONE_YR/preprocessing/qis_results_full_v3.csv")
qis_eval_res = pd.read_csv(r"/ONE_YR/preprocessing/qis_results_for_eval_v3.csv") # for evaluation
tmp_means = pd.DataFrame( [qis_results.iloc[:252,:].mean() for i in range(21)] )
qis_results = pd.concat([tmp_means, qis_results]).iloc[:, ]


'''
DATA DESCRIPTION
- qis_results: contains for each time point i the mean of the next 21 days of returns. This is done as the mean of the
                next 21 days is more robust than just taking a few of the next few returns. Further, this is why
                we use it for training and we added 21 datapoints to the beginning so we can actually use it for training.
                This should be the main input signal but also the "Y" of our procedudre 
- qis_eval_res: contains for every 21nd datapoint the returns for the next 21 days. Should go in line with the QIS results
                in the column "base". If this is true, it is correctly coded.
                
--> maybe instead of 
'''


# idea 1: train 3 models and always picke the one with largest predicted returns
# or try to learn

#### DEFINE DATA ############
len_train = 5040
end_date = fixed_shrk_data.shape[0]

Y = qis_results.iloc[21:, :]
X_old = np.vstack((x1, x2, x3, x4)).T
X = np.vstack((x1, x2, x3, x4, x5, x6, x7, x8)).T
X = np.concatenate((X, x9), axis=1)
# add prev. pf sds of the 3 versions
X = np.concatenate((X, qis_results.iloc[0:-21,]), axis=1)
# Xtest and Ytest are then used for continuous model updates
################################

# temp here
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]


##############################
# W/O FORWARD LOOKING BIAS and smaller train sizes
Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y.to_numpy()[0:len_train, :]
Ytest = Y.to_numpy()[len_train:, :]
w = Xtest.shape[0] // 21
preds_shrkg = []
train_size = 5000
for i in range(0, w):
    if i == 0:
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain[-train_size:-21, :], Ytrain[-train_size:-21, :])
        preds = regr.predict(Xtest[0, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    elif i == 1:
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain[-train_size:,], Ytrain[-train_size:,])
        preds = regr.predict(Xtest[21*i, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    else:
        Xtrain = np.concatenate((Xtrain, Xtest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        Ytrain = np.concatenate((Ytrain, Ytest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain, Ytrain)
        preds = regr.predict(Xtest[21*i, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
res_model, res_qis = eval_triplet_preds(preds_shrkg, qis_eval_res.iloc[len_train:,])
#####################################################


##############################
# W/O FORWARD LOOKING BIAS and smaller train sizes
Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y.to_numpy()[0:len_train, :]
Ytest = Y.to_numpy()[len_train:, :]
w = Xtest.shape[0] // 21
preds_shrkg = []
train_size = 5000
for i in range(0, w):
    if i == 0:
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain[-train_size:-21, :], Ytrain[-train_size:-21, :])
        preds = regr.predict(Xtest[0, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    elif i == 1:
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain[-train_size:,], Ytrain[-train_size:,])
        preds = regr.predict(Xtest[21*i, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    else:
        Xtrain = np.concatenate((Xtrain, Xtest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        Ytrain = np.concatenate((Ytrain, Ytest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain, Ytrain)
        preds = regr.predict(Xtest[21*i, :].reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
res_model, res_qis = eval_triplet_preds(preds_shrkg, qis_eval_res.iloc[len_train:,])
########################


Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y.to_numpy()[0:len_train, :]
Ytest = Y.to_numpy()[len_train:, :]
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
        weights[-600:] = [2 for i in range(600)]
        weights[-300:] = [4 for i in range(300)]
        x_test = sc.transform(Xtest[0, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x, Ytrain[-train_size:-21, :], sample_weight=np.array(weights) )
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    elif i == 1:
        x = Xtrain[-train_size:]
        sc.fit(x)
        x = sc.transform(x)
        #weights = [1 + i / 600 for i in range(x.shape[0])]
        weights = [1 for i in range(x.shape[0])]
        weights[-600:] = [2 for i in range(600)]
        weights[-300:] = [4 for i in range(300)]
        x_test = sc.transform(Xtest[21*i, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x, Ytrain[-train_size: ], sample_weight=np.array(weights))
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    else:
        if i == 2: # just once because we overwrite Xtrain
            Xtrain = Xtrain[-train_size: ]
            Ytrain = Ytrain[-train_size: ]
        Xtrain = np.concatenate((Xtrain, Xtest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        Ytrain = np.concatenate((Ytrain, Ytest[0 + 21*(i-2):21*(i-1), :]), axis=0)[21:, :]
        x = Xtrain
        sc.fit(x)
        x = sc.transform(x)
        #weights = [1 + i / 600 for i in range(x.shape[0])]
        weights = [1 for i in range(x.shape[0])]
        weights[-600:] = [2 for i in range(600)]
        weights[-300:] = [4 for i in range(300)]
        x_test = sc.transform(Xtest[21*i, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain, Ytrain, sample_weight=np.array(weights))
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
res_model, res_qis = eval_triplet_preds(preds_shrkg, qis_eval_res.iloc[len_train:,])
######## DONE

#######

### SAME AS ABOVE BUT TRAIN DS INCREASES WITH EVERY ITER
Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y.to_numpy()[0:len_train, :]
Ytest = Y.to_numpy()[len_train:, :]
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
        weights[-600:] = [2 for i in range(600)]
        weights[-300:] = [4 for i in range(300)]
        x_test = sc.transform(Xtest[0, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x, Ytrain[-train_size:-21, :], sample_weight=np.array(weights) )
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    elif i == 1:
        x = Xtrain[-train_size:]
        sc.fit(x)
        x = sc.transform(x)
        #weights = [1 + i / 600 for i in range(x.shape[0])]
        weights = [1 for i in range(x.shape[0])]
        weights[-600:] = [2 for i in range(600)]
        weights[-300:] = [4 for i in range(300)]
        x_test = sc.transform(Xtest[21*i, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(x, Ytrain[-train_size: ], sample_weight=np.array(weights))
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
    else:
        if i == 2: # just once because we overwrite Xtrain
            Xtrain = Xtrain[-train_size: ]
            Ytrain = Ytrain[-train_size: ]
        Xtrain = np.concatenate((Xtrain, Xtest[0 + 21*(i-2):21*(i-1), :]), axis=0)
        Ytrain = np.concatenate((Ytrain, Ytest[0 + 21*(i-2):21*(i-1), :]), axis=0)
        x = Xtrain
        sc.fit(x)
        x = sc.transform(x)
        #weights = [1 + i / 600 for i in range(x.shape[0])]
        weights = [1 for i in range(x.shape[0])]
        weights[-600:] = [2 for i in range(600)]
        weights[-300:] = [4 for i in range(300)]
        x_test = sc.transform(Xtest[21*i, :].reshape(1, -1))
        regr = MultiOutputRegressor(ElasticNet(random_state=123)).fit(Xtrain, Ytrain, sample_weight=np.array(weights))
        preds = regr.predict(x_test.reshape(1, -1))
        preds_shrkg.append((preds.argmin(axis=1))[0])
res_model, res_qis = eval_triplet_preds(preds_shrkg, qis_eval_res.iloc[len_train:,])
######## DONE


########## TAKE THE SHRINKAGE INTENSITIES THAT RESULTED IN THE MINIMUM 21 AVG STD DEV
### WITHOUT FORWARD LOOKING BIAS

qis_results = pd.read_csv(r"/ONE_YR/preprocessing/qis_results_full_v3.csv")
qis_eval_res = pd.read_csv(r"/ONE_YR/preprocessing/qis_results_for_eval_v3.csv") # for evaluation
tmp_means = pd.DataFrame( [qis_results.iloc[:252,:].mean() for i in range(21)] )
qis_results = pd.concat([tmp_means, qis_results]).iloc[:, ]


res1 = []
for i in range(5040, qis_eval_res.shape[0], 21):
    curmin = (qis_results.iloc[i+21]).idxmin()
    res1 += list(qis_eval_res.iloc[i:i+21][curmin])
np.std(res1) * np.sqrt(252) * 100



############### RANDOM TESTING STUFF
from sklearn.preprocessing import LabelBinarizer
print("done")
