# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/dados_cerveja_nota.xlsx")

df['aprovado'] = (df['nota'] > 5).astype(int)
df

# %%
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title("Cerveja vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovação")

# %%
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

X = df[['cerveja']]
y = df['aprovado']

# Regressão logística
reg = linear_model.LogisticRegression(penalty=None, 
                                      fit_intercept=True)
reg.fit(X, y)
reg_predict = reg.predict(X.drop_duplicates())
reg_prob = reg.predict_proba(X.drop_duplicates())[:,1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(X, y)
arvore_full_predict = arvore_full.predict(X.drop_duplicates())
arvore_full_prob = arvore_full.predict_proba(X.drop_duplicates())[:,1]

arvore_depth2 = tree.DecisionTreeClassifier(random_state=42,
                                            max_depth=2)
arvore_depth2.fit(X, y)
arvore_depth2_predict = arvore_depth2.predict(X.drop_duplicates())
arvore_depth2_prob = arvore_depth2.predict_proba(X.drop_duplicates())[:,1]

# Naive bayes
nb = naive_bayes.GaussianNB()
nb.fit(X, y)
nb_predict = nb.predict(X.drop_duplicates())
nb_proba = nb.predict_proba(X.drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(X, y, 'o', color='royalblue', label='Amostra')
plt.grid(True)
plt.title("Cerveja vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovação")
plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')
plt.plot(X.drop_duplicates(), reg_predict, color='magenta', 
         label='Reg Log Predict')
plt.plot(X.drop_duplicates(), reg_prob, color='red',
         label='Reg Log Proba')
plt.plot(X.drop_duplicates(), arvore_full_predict, color='darkorange',
         label='Arvore Full Predict')
plt.plot(X.drop_duplicates(), arvore_full_prob, color='darkgreen',
         label='Arvore Full Proba')
plt.plot(X.drop_duplicates(), arvore_depth2_predict, color='lightcoral',
         label='Arvore depth2 Predict')
plt.plot(X.drop_duplicates(), arvore_depth2_prob, color='rebeccapurple',
         label='Arvore depth2 Proba')
plt.plot(X.drop_duplicates(), nb_predict, color='springgreen',
         label='Naive Bayes Predict')
plt.plot(X.drop_duplicates(), nb_proba, color='black',
         label='Naive Bayes Proba')

plt.legend()