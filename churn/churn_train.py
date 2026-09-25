# %%

# Import de libs
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# %%
#Import dos dados
CAMINHO = os.path.join(os.getcwd(), "telecom_churn.csv")
df_raw = pd.read_csv(CAMINHO,sep=",")
print(f"Formato do arquivo bruto: {df_raw.shape}")
# %%
#Colunas existentes
df_raw.columns
# %%
#Tradução das colunas para ficar mais fácil o trabalho.
traducao = {
    'Churn': 'Cancelamento',
    'AccountWeeks': 'Semanas de Conta',
    'ContractRenewal': 'Renovação do Contrato',
    'DataPlan': 'Plano de Dados',
    'DataUsage': 'Uso de Dados',
    'CustServCalls': 'Chamadas ao Atendimento',
    'DayMins': 'Minutos Diurnos',
    'DayCalls': 'Chamadas Diurnas',
    'MonthlyCharge': 'Cobrança Mensal',
    'OverageFee': 'Taxa de Excedente',
    'RoamMins': 'Minutos em Roaming'
}

df = df_raw.rename(columns=traducao)
# %%
df
# %%
print("linhas x colunas:", df.shape)
print()
# %%
df.info()

# %%
#Dividir entre features e target
features = df.columns[1:]
target = df.columns[0]
# %%
features
# %%
target
# %%

#Atribuir as features as variáveis.
X, y = df[features], df[target]
# %%
#Dividir entre treino e teste.
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
print(f"Taxa de variável resposta treino: {y_train.mean()}")
print(f"Taxa de variável resposta test: {y_test.mean()}")

# %%
df_analise = X
df_analise[target] = y
sumario = df_analise.groupby(by=target).agg(["mean", "median"]).T
# %%
sumario
# %%
sumario["diff_abs"] = sumario[0] - sumario[1]
sumario["diff_rel"] = sumario[0] / sumario[1]
sumario.sort_values(by="diff_rel", ascending=False)
# %%

#A árvore nos auxilia para entender quais variáveis estão nos ajudando. Qual delas tem a melhor importância preditiva.
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
arvore.fit(X_train,y_train)

plt.figure(dpi=700, figsize=[4,4])
tree.plot_tree(arvore, feature_names=X_train.columns,
               filled=True,
               class_names=[str(i) for i in arvore.classes_]
               )
# %%
feature_importance = pd.Series(arvore.feature_importances_,
                                index=X_train.columns).sort_values(ascending=False).reset_index()
feature_importance
# %%
feature_importance['acum'] = feature_importance[0].cumsum()
feature_importance
# %%
