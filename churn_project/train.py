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
from sklearn import naive_bayes
from sklearn import ensemble

model = linear_model.LogisticRegression(
    penalty=None, 
    random_state=42,
    max_iter=10000
    )
model = naive_bayes.BernoulliNB()
model = ensemble.RandomForestClassifier(
    random_state=42,
    min_samples_leaf=20,
    n_jobs=-1, # nucleos do computador a ser usado
    n_estimators=500,
    )
model = ensemble.AdaBoostClassifier(
    random_state=42,
    n_estimators=500,
    learning_rate=0.01,
)

model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretizar', tree_discretisation),
        ('Onehot', onehot),
        ('Model', model),
    ]
)

model_pipeline.fit(X_train[best_features], y_train)

# %%
from sklearn import metrics

y_train_predict = model_pipeline.predict(X_train[best_features])
y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
roc_train = metrics.roc_curve(y_train, y_train_proba)
print("Acuracia Treino: ", acc_train)
print("AUC Treino: ", auc_train)

# %% Teste na base de test


y_test_predict = model_pipeline.predict(X_test[best_features])
y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]
roc_test = metrics.roc_curve(y_test, y_test_proba)

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print("Acuracia Test: ", acc_test)
print("AUC Test: ", auc_test)

# %% Teste na OOT

y_oot_predict = model_pipeline.predict(oot[best_features])
y_oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]
roc_oot = metrics.roc_curve(oot[target], y_oot_proba)

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
print("Acuracia OOT: ", acc_oot)
print("AUC OOT: ", auc_oot)

# %%
plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.title("Curva ROC")
plt.legend([
    f"Train: {100*auc_train:.2f}",
    f"Test: {100*auc_test:.2f}",
    f"Out-of-time: {100*auc_oot:.2f}",
])