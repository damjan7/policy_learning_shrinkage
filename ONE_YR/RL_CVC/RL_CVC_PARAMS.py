import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# plotting
import matplotlib.pyplot as plt

from V5.helpers import eval_funcs, eval_function_new

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

# IMPORT SHRK DATASETS
shrk_data_path = None
pf_size = 500

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"C:\Users\kostovda\OneDrive - Luzerner Kantonsspital\Anlagen\code\V5\1YR\preprocessing\training_dfs\PF{pf_size}\fixed_shrkges_cov2Para_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"C:\Users\kostovda\OneDrive - Luzerner Kantonsspital\Anlagen\code\V5\1YR\preprocessing\training_dfs\PF{pf_size}\cov2Para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

permnos = pd.read_pickle(
    fr"C:\Users\kostovda\OneDrive - Luzerner Kantonsspital\Anlagen\code\V5\1YR\preprocessing\rets_permnos_1Y\permnos_1Y_p{pf_size}.pickle")
rets_full = pd.read_pickle(
    fr"C:\Users\kostovda\OneDrive - Luzerner Kantonsspital\Anlagen\code\V5\1YR\preprocessing\rets_permnos_1Y\returns_full_1Y_p{pf_size}.pickle")

# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = r"C:\Users\kostovda\OneDrive - Luzerner Kantonsspital\Anlagen\code\V5\helpers"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too

start_date = str(optimal_shrk_data['date'].iloc[0])
start_date = start_date[0:4] + '-' + start_date[4:6] + "-" + start_date[6:]
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]


# Even for a simple TD-Learning method, we need an estimate of the Q(s,a) or V(s) function
# As we have a continuous state space, we discretize the action space
class ActorCritic(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        ### NOTE: can share some layers of the actor and critic network as they have the same structure
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.state_action_head = nn.Linear(int(hidden_size/2), num_actions)  # probabilistic mapping from states to actions
        #self.critic_head = nn.Linear(int(hidden_size/2), 1)  # estimated state value, I don't use this for now
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # get action distribution
        # action_probs = F.softmax(self.state_action_head(x), dim=1)
        # I DO NOT NEED PROBABILITEIS NOW
        state_action_value = self.state_action_head(x)
        # how 'good' is the current state?
        #state_value = self.critic_head(x)
        return state_action_value


class MyDataset(Dataset):
    def __init__(self, factors, fixed_shrk_data, optimal_shrk_data, normalize=False):
        if normalize == True:  # for now only scale factors, I don't scale them actually
            self.factors_scaler = MinMaxScaler()
            self.factors = pd.DataFrame(self.factors_scaler.fit_transform(factors))
        else:
            self.factors = factors
        self.fixed_shrk_data = fixed_shrk_data
        self.optimal_shrk_data = optimal_shrk_data
        print("loaded")

    def __len__(self):
        return self.factors.shape[0]

    def __getitem__(self, idx):
        # inputs multiplied by 100 works better
        inp = torch.Tensor(np.append(self.factors.iloc[idx, :].values, self.optimal_shrk_data.iloc[idx, 1])) * 100
        labels = torch.Tensor(np.array(self.fixed_shrk_data.iloc[idx, 2:].values, dtype=float))
        # for labels: .view(1, -1) not needed when working with Dataset and DataLoader
        return inp, labels

def train_with_dataloader(normalize=False, dst=None, batch_size=None):
    # split dataset into train and validation
    batch_size = batch_size
    total_num_batches = factors.shape[0] // batch_size
    # tot len of dataset is 10374
    len_train = 5040
    end_date = fixed_shrk_data.shape[0]
    # len_train = int(total_num_batches * 0.7) * batch_size
    train_dataset = MyDataset(
        factors.iloc[:len_train, :],
        fixed_shrk_data.iloc[:len_train, :],
        optimal_shrk_data.iloc[:len_train, :],
        normalize=normalize
    )

    val_dataset = MyDataset(
        factors.iloc[len_train:end_date, :],
        fixed_shrk_data.iloc[len_train:end_date, :],
        optimal_shrk_data.iloc[len_train:end_date, :],
        normalize=False,
    )
    if normalize == True:
        val_dataset.factors = pd.DataFrame(train_dataset.factors_scaler.transform(val_dataset.factors))

    train_dataloader = DataLoader(train_dataset)
    val_dataloader = DataLoader(val_dataset)

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
            out = net(X.view(1, -1))
            train_preds.append(torch.argmin(out).item())
            # CALC LOSS AND BACKPROPAGATE
            optimizer.zero_grad()
            loss = criterion(out, labels)  # MSE between outputs of NN and pf std --> pf std can be interpreted
            # as value of taking action a in state s, hence want my network to learn this
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        print(f"Epoch {epoch} training done.")

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

                val_preds.append(torch.argmin(out).item())
                loss = criterion(out, labels)
                epoch_val_loss.append(loss.item())
                actual_argmin_validationset.append(torch.argmin(labels).item())


            # map predictions from 1 to 21 to shrinkage intensities
            mapped_shrkges = list(map(eval_funcs.f2_map, val_preds))

        if epoch == 2:
            print("f")

        if epoch in [4, 7, 11, 14, 17, 20, 23, 25, 27, 29, 33, 36, 39]:
            print(f'Calculating res for epoch {epoch}')
            val_indices_correct = (len_train, end_date)
            val_indices_results = [val_indices_correct[0] + 21 * i for i in
                                   range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
            val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
            mapped_shrkges_v2 = np.array(mapped_shrkges)[val_idxes_shrkges]
            #r = eval_function_new.eval_fct_networkonly_1YR(mapped_shrkges_v2, rets_full, permnos, 0, val_indices_results)
            #FULL_EVAL.append(r)
            if epoch >= 21:
                r = eval_function_new.eval_fct_networkonly_1YR(mapped_shrkges_v2, rets_full, permnos, 0,
                                                               val_indices_results)
                print(r)
                if r['network pf sd'] < 10.5:
                    print("k")
        net.train()
    return FULL_EVAL


params={
    'lr': [1e-3, 1e-4, 1e-5],
    'hidden_layer_size': [32, 64, 128, 256, 512],
    'dst': [0.0, 0.01, 0.05],
    'batch_size': [16, 32, 64]
}

params = {
    'lr': [1e-5],
    'hidden_layer_size': [128],
    'dst': [0.05],
    'batch_size': [64]
}

# PARAMETERS:
num_epochs = 41
num_features = factors.shape[1] + 1  # all 13 factors + opt shrk
num_actions = fixed_shrk_data.shape[1] - 2  # since 1 col is dates, 1 col is hist vola

import itertools
FULL_RES = []
for idx, prod in enumerate(itertools.product(*params.values())):
    torch.manual_seed(31782)
    lr, hidden_layer_size, dst, batch_size = prod[0], prod[1], prod[2], prod[3]
    net = ActorCritic(num_features, num_actions, hidden_layer_size)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    # start training
    FULL_RES.append(train_with_dataloader(normalize=False, dst=dst, batch_size=batch_size))


print("done")

