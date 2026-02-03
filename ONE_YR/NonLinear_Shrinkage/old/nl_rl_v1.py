import pandas as pd
import numpy as np
import pickle
import os

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity([0,1,2,3,4,5,6])

# Torch Stuff
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from helpers import eval_funcs_multi_target
from helpers import eval_funcs

base_folder_path = r'/'
# IMPORT SHRK DATASETS
shrk_data_path = None
pf_size = 500
permnos = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"{base_folder_path}\ONE_YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{base_folder_path}\ONE_YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)

# get all the validation indices
len_train = 5040
end_date = fixed_shrk_data.shape[0]

# rets full is longer than permnos
reb_date_1 = permnos.index[0]
add_idx = np.where(rets_full.index == reb_date_1)[0][0]
#rets_full = rets_full.iloc[add_idx:, :]
# --> then for datapoint 0, last 21 days are "past_returns"

val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]


# some fct definitions and network structure
def eval_qis_shrkges(qis_shrkges, past_ret_mat, fut_ret_mat):
    n = qis_shrkges.shape[0] - 1
    sample = pd.DataFrame(np.matmul(past_ret_mat.T.to_numpy(),past_ret_mat.to_numpy()))/n
    sample = (sample + sample.T) / 2
    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)
    temp1 = dfu.to_numpy()
    temp2 = np.diag(qis_shrkges)
    temp3 = dfu.T.to_numpy().conjugate()
    # reconstruct covariance matrix
    sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1,temp2),temp3))





class Net(nn.Module):
    def __init__(self, num_features, out_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.out = nn.Linear(int(hidden_size / 2), out_size)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # get action distribution
        # action_probs = F.softmax(self.state_action_head(x), dim=1)
        # I DO NOT NEED PROBABILITEIS NOW
        x = self.out(x)
        x = F.tanh(x)
        return x

class MyDataset(Dataset):
    def __init__(self, rets_full, permnos):
        self.rets_full = rets_full
        self.permnos = permnos
        reb_date_1 = permnos.index[0]
        self.add_idx = np.where(rets_full.index == reb_date_1)[0][0]
        qis_shrkges_path = r"/ONE_YR/preprocessing/QIS_shrinkages.csv"
        self.qis_shrkges = pd.read_csv(qis_shrkges_path)

    def __len__(self):
        return self.permnos.shape[0]

    def __getitem__(self, idx):
        past_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx - 21*12*1: idx + add_idx, :]
        past_ret_mat = past_ret_mat.sub(past_ret_mat.mean())
        past_ret_mat = past_ret_mat.fillna(0)
        fut_ret_mat = rets_full[permnos.iloc[idx]].iloc[idx + add_idx: idx + add_idx + 21, :]
        qis_shrkges = self.qis_shrkges.iloc[idx, :].to_numpy()
        return torch.Tensor(qis_shrkges), torch.Tensor(past_ret_mat.to_numpy()), torch.Tensor(fut_ret_mat.to_numpy())


def train_with_dataloader(net, optimizer, criterion, num_epochs):
    batch_size = 32
    total_num_batches = permnos.shape[0] // batch_size
    len_train = 5040
    end_date = fixed_shrk_data.shape[0]

    train_dataset = MyDataset(
        rets_full.iloc[0:len_train, ],
        permnos.iloc[0:len_train, ]
    )
    val_dataset = MyDataset(
        rets_full.iloc[len_train:, ],
        permnos.iloc[len_train:, ]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    for epoch in range(1, num_epochs+1):
        train_preds = []
        val_preds = []
        actual_train_labels = []
        epoch_loss = []
        for i, data in enumerate(train_dataloader):
            qis_shrkges, past_ret, fut_ret = data  # labels are actually the annualized pf standard deviations [= "reward"]
            out = net(qis_shrkges.view(1, -1))

            # calc pf ret for qis and out, if qis > out then calc loss ELSE 0

            train_preds.append(torch.argmin(out).item())
            # CALC LOSS AND BACKPROPAGATE
            optimizer.zero_grad()
            loss = criterion(out[0], out[0])   # MSE between outputs of NN and pf std --> pf std can be interpreted
            # as value of taking action a in state s, hence want my network to learn this
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        print(f"Epoch {epoch} training done.")


lr = 1e-4
in_features, out_features, hidden_layer_size = 500, 500, 1000
net = Net(in_features, out_features, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()
train_with_dataloader(net, optimizer, criterion, 10)


print("done")