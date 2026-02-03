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
pf_size = 100

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

# SET MANUALLY
create_new_inputs = True
new_inputs_folder = r"C:\Users\kostovda\OneDrive - Luzerner Kantonsspital\Anlagen\code\V5\1YR\RL_CVC\new_inputs"
if create_new_inputs == True:
    from preprocess_new_inputs import get_new_inputs
    new_inputs = get_new_inputs(rets_full, permnos)
    ni_df = pd.DataFrame.from_dict(new_inputs)
    ni_df.to_csv(new_inputs_folder + fr"\new_inputs_v2_p{pf_size}.csv", index=False)
    print("created new inputs")
else:
    new_inputs = pd.read_csv(new_inputs_folder + fr"\new_inputs_p{pf_size}.csv").to_dict(orient='list')


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

smooth_cvc_shrkges = False
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

iqr = [i-j for i,j in zip(x5, x6)]

Y = fixed_shrk_data.iloc[:, 3:].values

len_train = 5040
end_date = fixed_shrk_data.shape[0]
X_old = np.vstack((x1, x2, x3, x4)).T
X = np.vstack((x1, x2, x3, x4, x5, x6, x7, x8)).T
X = np.concatenate((X, x9), axis=1)

for i in range(X.shape[1]):
    if not i == 2:
        X[:, i] = X[:, i] * 100

X_old = X
X = X_old[:, [0, 1, 2, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20]]

X = X_old[:, [3, 9,10,11,12,13,14,15,16,17,18,19,20]]

X = np.vstack((x1, x2, x3, iqr, x7, x8)).T

# Variables as in paper; factors and shrk factor
X = np.concatenate((x4.reshape(-1,1), x9), axis=1)
X = np.concatenate((x4_smooth.reshape(-1,1), x9), axis=1)



Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y[0:len_train, :]
Ytest = Y[len_train:, :]
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
regr = MultiOutputRegressor(Ridge(alpha=2, random_state=123)).fit(Xtrain, Ytrain)
preds = regr.predict(Xtest)
preds_shrkg = preds.argmin(axis=1)/100


'''
val_indices_correct = (len_train, end_date)
val_indices_results = [val_indices_correct[0] + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21 * i for i in range((val_indices_correct[-1] - val_indices_correct[0]) // 21)]
### cvc_shrk = cvc_shrinkges = Xtest[:, 3]
eval_function_new.myplot(preds_shrkg[val_idxes_shrkges], cvc_shrk[val_idxes_shrkges])

r = eval_function_new.eval_fct_networkonly_1YR(preds_shrkg[val_idxes_shrkges], rets_full, permnos, 0, val_indices_results)


# save eval_fct plot:

eval_function_new.plot_and_save("CVC and Ridge with 4 parameters", "Datapoints", "Shrinkage Intensity", (preds_shrkg[val_idxes_shrkges], 'Ridge'), (cvc_shrk[val_idxes_shrkges], 'CVC'), show=True)
'''

from sklearn.ensemble import GradientBoostingRegressor
xgb = MultiOutputRegressor(GradientBoostingRegressor()).fit(Xtrain, Ytrain)
preds = xgb.predict(Xtest)
preds_shrkg = preds.argmin(axis=1)/100


from sklearn.linear_model import LogisticRegression
logreg = MultiOutputRegressor(LogisticRegression()).fit(Xtrain, Ytrain)
preds = logreg.predict(Xtest)
preds_shrkg = preds.argmin(axis=1)/100

print("done")


for i in range(1, 101):
    if i == 1:
        newdat = np.concatenate((X, np.array([i/100 for _ in range(X.shape[0])]).reshape(-1, 1) ), axis=1)
    else:
        X2 = np.concatenate((X, np.array([i/100 for _ in range(X.shape[0])]).reshape(-1, 1) ), axis=1)
        newdat  = np.concatenate((newdat, X2), axis=0)


base_shrkgs = np.array([i/100 for i in range(1, 101)] * X.shape[0]).reshape(-1, 1)
for i in range(1, 101):
    if i == 1:
        newdat = X
    else:
        newdat = np.concatenate((newdat, X))
newdat = np.concatenate((newdat, base_shrkgs), axis = 1)

for i in range(Y.shape[0]):
    if i == 0:
        Y2 = Y[i, :].reshape(-1, 1)
    else:
        Y2 = np.concatenate((Y2, Y[i, :].reshape(-1, 1)))
Y = Y2

X = newdat
len_train = 504000

Xtrain = X[0:len_train, :]
Xtest = X[len_train:, :]
Ytrain = Y[0:len_train, :]
Ytest = Y[len_train:, :]
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
regr = Ridge(alpha=1, random_state=123).fit(Xtrain, Ytrain)
preds = regr.predict(Xtest)

actual_preds = []

for i in range(0, preds.shape[0], 100):
    j = i + 100
    example_i = preds[i, j, :]
    actual_preds.append()