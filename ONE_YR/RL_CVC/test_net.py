import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


#x1 = np.random.randint(0, 100, size=500)
x1 = np.random.uniform(0, 1, size=500)
x2_x3 = np.random.multivariate_normal([2, 6], [[1, 0], [0, 3]], size=500)
y = np.random.multivariate_normal([10, 6], [[2, 0], [0, 1.5]], size=500)
X = np.concatenate([x1.reshape(-1, 1), x2_x3], axis=1)

class Net(nn.Module):


    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.state_action_head = nn.Linear(int(hidden_size/2), num_actions)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        state_action_value = self.state_action_head(x)
        return state_action_value

torch.manual_seed(31782)
num_epochs = 20
lr = 1e-4
num_features = X.shape[1]  # all 13 factors + opt shrk
num_actions = y.shape[1]
hidden_layer_size = 16
net = Net(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

actions = torch.tensor([0, 1], dtype=torch.float32, requires_grad=True)
all_losses=[]
for epoch in range(1, num_epochs + 1):
    for xi, yi in zip(X_tensor, y_tensor):
        out = net(xi.view(1, -1))
        optimizer.zero_grad()
        loss = criterion(xi[0], actions)
        all_losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()

    if epoch == 5:
        print("Epoch 5")

print("done")