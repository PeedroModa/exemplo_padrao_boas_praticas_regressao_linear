# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)

import warnings
warnings.filterwarnings("ignore")

# %%
CAMINHO = os.path.join(os.getcwd(), "abt_churn.csv")
df_raw = pd.read_csv(CAMINHO,sep=",")
print(f"Formato do arquivo bruto: {df_raw.shape}")
# %%
oot = df_raw[df_raw["dtRef"] == df_raw["dtRef"].max()].copy()
oot
# %%
df_train = df_raw[df_raw["dtRef"] < df_raw["dtRef"].max()].copy()
df_train.shape
# %%
features = df_train.columns[2:-1]
target = df_train.columns[-1]
# %%
X, y = df_train[features], df_train[target]
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    random_state=42,
    test_size=0.2,
    stratify=y
)
# %%
X
# %%
y
# %%
print(f"Taxa de variável resposta: {y_train.mean()}")
print(f"Taxa de variável resposta: {y_test.mean()}")
# %%
