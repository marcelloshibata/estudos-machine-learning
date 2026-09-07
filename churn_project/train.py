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

# %%
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())
best_features

# %% MODIFY
from feature_engine import discretisation, encoding
from sklearn import pipeline


# %% Discretizar
tree_discretisation = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    cv=3
)

# %% One hot
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)


# %% MODEL
from sklearn import linear_model

reg = linear_model.LogisticRegression(
    penalty=None, 
    random_state=42,
    max_iter=10000)

model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretizar', tree_discretisation),
        ('Onehot', onehot),
        ('Model', reg),
    ]
)

model_pipeline.fit(X_train, y_train)

# %%
from sklearn import metrics

y_train_predict = model_pipeline.predict(X_train)
y_train_proba = model_pipeline.predict_proba(X_train)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
print("Acuracia Treino: ", acc_train)
print("AUC Treino: ", auc_train)

# %% Teste na base de test


y_test_predict = model_pipeline.predict(X_test)
y_test_proba = model_pipeline.predict_proba(X_test)[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print("Acuracia Test: ", acc_test)
print("AUC Test: ", auc_test)

# %% Teste na OOT

y_oot_predict = model_pipeline.predict(oot[features])
y_oot_proba = model_pipeline.predict_proba(oot[features])[:,1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
print("Acuracia OOT: ", acc_oot)
print("AUC OOT: ", auc_oot)