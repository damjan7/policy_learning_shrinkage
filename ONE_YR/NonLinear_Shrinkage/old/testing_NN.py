import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([5,6,7,8,9, 10, 11, 12, 13,14,15])

import pandas as pd
import numpy as np
import pickle
import os

from collections import defaultdict

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([5,6,7,8,9, 10, 11, 12, 13,14,15])

from helpers import rl_covmat_ests_for_dataset as estimators
from helpers import helper_functions as hf
from helpers import eval_funcs_multi_target
from helpers import eval_funcs
from helpers import eval_function_new

import regression_evaluation_funcs as re_hf
import helper_functions_NL_RL as NL_hf
from collections import Counter



# define factors
all_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
               1.9, 2.0]
# num eigenvalues to modify
num_eigenvalues = [1, 5, 10, 25, 50]

# again all_res stores the results of next 21 days so we can use it for training
# but only with a gap of 21 days to avoid forward looking bias of course
'''
Description:
all_res is the 21 day lead std dev of the returns of the minvar portfolio
all_rawres is the return of each day
'''
all_rawres = {}
all_res = {}
qis_grid_data = r"H:\all\RL_Shrinkage_2024\ONE_YR\preprocessing\qis_eigenvalue_grid_data"
for num_ev in num_eigenvalues:
    df = pd.read_csv(qis_grid_data + f"\\qis_grid_allres_{num_ev}_evs.csv")
    all_res[num_ev] = df
    df = pd.read_csv(qis_grid_data + f"\\qis_grid_all_rawres_{num_ev}_evs.csv")
    all_rawres[num_ev] = df

print("loaded data")

# Calculates the OOS results for each factor and each number of eigenvalues
# while keeping the factor constant for the whole OOS period
factor_results = {}
for k, v in all_rawres.items():
    tmp = round(v.iloc[5040:,].std() * np.sqrt(252) * 100, 2)
    factor_results[k] = tmp
factor_results_df = pd.DataFrame(factor_results)


# CAN use allres_min_idxes as a signal, as it is shifted by 21 days
# so no forward looking bias
allres_min_idxes = {}
for k, v in all_res.items():
    minima = v.idxmin(axis=1)[5040-22: -22].values
    allres_min_idxes[k] = minima
allres_min_idxes_df = pd.DataFrame(allres_min_idxes)


allres_min_idxes_full = {}
for k, v in all_res.items():
    minima = v.idxmin(axis=1)[: -21].values
    minima = np.insert(minima, 0, np.repeat(["1.0"], 21))
    allres_min_idxes_full[k] = minima
allres_min_idxes_full_df = pd.DataFrame(allres_min_idxes_full)

# for sanity check: BIASED version should generally be better than
# non biased version as it is literally the minimum over the future 21 days
# so using it as a signal should outperform
allres_min_idxes_BIASED = {}
for k, v in all_res.items():
    minima = v.idxmin(axis=1).values
    allres_min_idxes_BIASED[k] = minima
allres_min_idxes_BIASED_df = pd.DataFrame(allres_min_idxes_BIASED)

## Test
tmp = []
for i in range(allres_min_idxes[10].shape[0]):
    t1 = all_rawres[10].iloc[5040:].loc[:, allres_min_idxes[10][i]].iloc[i]
    tmp.append(t1)
print( np.std(tmp) * np.sqrt(252) * 100 )
    # 10.254910152698425
# or yielding the same result:
np.diag(all_rawres[10].iloc[5040:, ].loc[:, allres_min_idxes[10]]).std() * np.sqrt(252) * 100


# simple argmin rule, can be used as a benchmark
res_simple_argmin_rule = {}
for num_ev in num_eigenvalues:
    tmp = np.diag(all_rawres[num_ev].iloc[5040:, ].loc[:, allres_min_idxes[num_ev]])
    res_simple_argmin_rule[num_ev] = tmp
res_simple_argmin_rule = pd.DataFrame(res_simple_argmin_rule)


res_simple_RAWRES_argmin_rule = {}
for num_ev in num_eigenvalues:
    rawres_min_signal = all_rawres[num_ev].rolling(window=21, min_periods=1).mean().idxmin(axis=1)
    rawres_min_signal.pop(rawres_min_signal.index[-1])
    rawres_min_signal = pd.concat([ pd.Series(['1.0']), rawres_min_signal])
    tmp = (all_rawres[num_ev].loc[:, rawres_min_signal ])
    tmp = np.diag(tmp)[5040:]
    res_simple_RAWRES_argmin_rule[num_ev] = tmp
res_simple_RAWRES_argmin_rule = pd.DataFrame(res_simple_RAWRES_argmin_rule)
# now run a simple model, i.e., regression:
# i.e. as I did before; all factors --> run a regression for each factor
# to predict what factor to use at each time point
# run multioutputregressor again, but also need additional train data for that

# Load additional train data as before (see LR_rolling_eval.py)
pf_size = 500
len_train = 5040



class ActorCritic(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fcMid = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.state_action_head = nn.Linear(int(hidden_size/2), num_actions)

        self.SingleLayerNet =  nn.Linear(hidden_size, num_actions)
        # probabilistic mapping from states to actions
        #self.critic_head = nn.Linear(int(hidden_size/2), 1)  # estimated state value, I don't use this for now
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        #x = self.dropout(F.relu(self.fcMid(x)))
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        state_action_value = self.SingleLayerNet(x)
        return state_action_value


class MyDataset(Dataset):
    def __init__(self, X, Y, normalize=False):
        if normalize == True:  # for now only scale factors, I don't scale them actually
            self.factors_scaler = None
        else:
            self.X = X
            self.Y = Y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        # inputs multiplied by 100 works better
        inp = torch.Tensor(
            self.X[idx, :]
        )
        labels = torch.Tensor(self.Y[idx, :])
        return inp, labels

def train_with_dataloader(normalize=False, X=None, dst=None, batch_size=None, num_epochs=20):
    # tot len of dataset is 10353
    len_train = 5040
    # len_train = int(total_num_batches * 0.7) * batch_size

    Y = all_res[num_ev].to_numpy() * 100 # for first test

    Xtrain = X[:len_train - 21, :]
    Xtest = X[list(range(len_train, 10353, 21))]

    # scale Xtrain
    if normalize == True:
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.transform(Xtest)

    train_dataset = MyDataset(Xtrain, Y[:len_train-21, :])
    val_dataset = MyDataset(X[list(range(len_train, 10353, 21))], Y[list(range(len_train, 10353, 21))])

    '''
    X_train = X[:len_train - 21, :]
    X_test = X[len_train:, :]
    y_train = Y[:len_train - 21, :]
    y_test = Y[len_train:, :]
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # Create a DataLoader
    batch_size = 32  # Example batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    '''

    # split dataset into train and validation
    total_num_batches = 10353 // batch_size
    # tot len of dataset is 10374
    end_date = 10353
    # len_train = int(total_num_batches * 0.7) * batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset)

    num_epochs=num_epochs

    FULL_EVAL = []
    validation_loss = []
    for epoch in range(1, num_epochs+1):
        train_preds = []
        val_preds = []
        actual_train_labels = []
        epoch_loss = []
        for i, data in enumerate(train_dataloader):
            X, labels = data  # labels are actually the annualized pf standard deviations [= "reward"]
            actual_train_labels.append(torch.argmin(labels).item())
            out = net(X)
            train_preds.append(torch.argmin(out).item())
            # CALC LOSS AND BACKPROPAGATE
            optimizer.zero_grad()
            loss = criterion(out[0], labels[0])   # MSE between outputs of NN and pf std --> pf std can be interpreted
            # as value of taking action a in state s, hence want my network to learn this
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        # validate at end of epoch
        # set model into evaluation mode and deactivate gradient collection
        net.eval()
        epoch_val_loss = []
        actual_argmin_validationset = []
        val_preds_v2 = []
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                X, labels = data
                out = net(X.view(1, -1))
                # add temporary penalty
                tmp = torch.tensor([dst for _ in range(101)])
                dst2 = torch.abs(tmp - X[0][-1]/100) * (dst) # even smaller, since very few stocks

                val_preds.append(torch.argmin(out[0]).item())
                loss = criterion(out, labels)
                epoch_val_loss.append(loss.item())
                actual_argmin_validationset.append(torch.argmin(labels).item())

            if epoch % 100 == 0:
                print(f"epoch {epoch}")
                print("d")

            val_preds = np.repeat(val_preds, 21)
            mapped_res = re_hf.map_preds_to_factors(val_preds, all_factors)
            Y_eval = all_rawres[num_ev]
            print(re_hf.evaluate_all_factor_preds(mapped_res, Y_eval, len_train))

            # map predictions from 1 to 21 to shrinkage intensities


params = {
    'lr': [1e-4],
    'hidden_layer_size': [128],
    'dst': [0.00],
    'batch_size': [32]
}

# PARAMETERS:
num_epochs = 20
num_features = 1  # all 13 factors + opt shrk
num_actions = 16  # since 1 col is dates, 1 col is hist vola

from sklearn.preprocessing import  StandardScaler
new_data = [[0.1 for i in range(21)] for i in range(16)]
new_data_df = pd.DataFrame(new_data).T
new_data_df.columns = all_res[num_ev].columns
normalized_lagged_allres = pd.concat((new_data_df, all_res[num_ev])).iloc[:-21,:]
scaler = StandardScaler()
normalized_lagged_allres = scaler.fit_transform(normalized_lagged_allres)

# Load Input Data
opt_values = allres_min_idxes_BIASED[num_ev].astype(float)[:-21]
opt_values = np.insert(arr=opt_values, obj=0, values=np.repeat(1.0, 21))
opt_v3 = np.diag(all_res[num_ev].loc[:, allres_min_idxes_BIASED[num_ev]])[:-21]
opt_v3 = np.insert(arr=opt_v3, obj=0, values=np.repeat(7.0, 21))
X = re_hf.load_additional_train_data(
    pf_size=500,
    opt_values_factors=opt_values,
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
    include_iqr=True,
    additional_inputs=opt_v3
)

import itertools
FULL_RES = []
torch.manual_seed(31782)
lr, hidden_layer_size, dst, batch_size = 1e-4, 32, 0.00, 32
num_features = X.shape[1]
net = ActorCritic(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
criterion = nn.MSELoss()
criterion = nn.L1Loss()
# start training
FULL_RES.append(train_with_dataloader(normalize=True, X=X, dst=dst, batch_size=batch_size, num_epochs=505))

print("done")

