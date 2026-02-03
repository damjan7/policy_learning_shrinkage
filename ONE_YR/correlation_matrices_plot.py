import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load intensities for the models
cov1para_intensities = pd.read_csv(r"H:\all\RL_Shrinkage_2024\ONE_YR\Linear_Shrinkage\results\cov1para_intensities.csv", index_col=0)

# need only OOS intensities for cov1para
cov1para_intensities = cov1para_intensities.iloc[5040:, :]
nl_linear_intensities = pd.read_csv(r"H:\all\RL_Shrinkage_2024\ONE_YR\Linear_Shrinkage\results\model_linear_intensities.csv", index_col=0)
print("loaded data")


df = cov1para_intensities.reset_index(drop=True).join(nl_linear_intensities.reset_index(drop=True), lsuffix=" L", rsuffix=" PL-L")

# filter for each rebalancing date only
df = df.iloc[list(range(0, df.shape[0], 21)), :]

# create correlation matrix
plt.figure(figsize=(15, 10))
plt.tight_layout()
sns.set_theme(font_scale=1.4)
sns.heatmap(df.corr().round(2), annot=True, cmap="crest")
plt.show()
print("done")