# %%
import pandas as pd

df = pd.read_csv("../data/abt_churn.csv")
df.head()

# %%

# definindo out of time (safra mais recente)
oot = df[df["dtRef"]==df['dtRef'].max()].copy()
oot

# %%
df_train = df[df["dtRef"]<df['dtRef'].max()].copy()

# %%
features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %% SAMPLE

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y,
                                                                    )
print("Taxa variavel resposta Treino: ", y_train.mean())
print("Taxa variavel resposta Teste: ", y_test.mean())

# %% Explore

# Missings
X_train.isna().sum().sort_values(ascending=False)

# %%
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(['mean', 'median']).T
sumario

# %%
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'], ascending=False)

# %%
from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

feature_importances = (pd.Series(arvore.feature_importances_, 
                                 index=X_train.columns).sort_values(ascending=False)
                                 .reset_index())

feature_importances['acum.'] = feature_importances[0].cumsum()
feature_importances[feature_importances['acum.'] < 0.96]