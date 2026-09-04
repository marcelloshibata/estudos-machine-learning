# %%
import pandas as pd

df = pd.read_csv("data/dados_resposta.csv")
df = df.replace({"Sim":1, "Não":0})
df.head()

# %%
num_vars = [
    'Curte games?',
    'Curte futebol?',
    'Curte livros?',
    'Curte jogos de tabuleiro?',
    'Curte jogos de fórmula 1?',
    'Curte jogos de MMA?',
    'Idade',
]

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)",
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa feliz'] = df['Você se considera uma pessoa feliz?'].copy().astype(int)
df_analise
df_analise = df_analise.dropna(subset=['pessoa feliz'])

# %%
features = [c for c in df_analise.columns if c != 'pessoa feliz']
X = df_analise[features]
y = df_analise['pessoa feliz']

from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model

arvore = tree.DecisionTreeClassifier(random_state=42, 
                                      min_samples_leaf=6)
arvore.fit(X, y)

naive = naive_bayes.GaussianNB()
naive.fit(X, y)

reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)
reg.fit(X, y)

# %%

arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analise[['pessoa feliz']].copy()
df_predict['predict_arvore'] = arvore_predict
df_predict['proba_arvore'] = arvore.predict_proba(X)[:,1]

df_predict['predict_naive'] = naive.predict(X)
df_predict['proba_naive'] = naive.predict_proba(X)[:,1]

df_predict['predict_reg'] = reg.predict(X)
df_predict['proba_reg'] = reg.predict_proba(X)[:,1]


#%%
from sklearn import metrics

acc_arvore = metrics.accuracy_score(y, df_predict['predict_arvore'])
precisao_arvore = metrics.precision_score(y, df_predict['predict_arvore'])
recall_arvore = metrics.recall_score(y, df_predict['predict_arvore'])
roc_arvore = metrics.roc_curve(y, df_predict['proba_arvore'])
# o quão proximo de 1 a curva ROC está
auc_arvore = metrics.roc_auc_score(y, df_predict['proba_arvore'])

acc_naive = metrics.accuracy_score(y, df_predict['predict_naive'])
precisao_naive = metrics.precision_score(y, df_predict['predict_naive'])
recall_naive = metrics.recall_score(y, df_predict['predict_naive'])
roc_naive = metrics.roc_curve(y, df_predict['proba_naive'])
# o quão proximo de 1 a curva ROC está
auc_naive = metrics.roc_auc_score(y, df_predict['proba_naive'])

acc_reg = metrics.accuracy_score(y, df_predict['predict_reg'])
precisao_reg = metrics.precision_score(y, df_predict['predict_reg'])
recall_reg = metrics.recall_score(y, df_predict['predict_reg'])
roc_reg = metrics.roc_curve(y, df_predict['proba_reg'])
# o quão proximo de 1 a curva ROC está
auc_reg = metrics.roc_auc_score(y, df_predict['proba_reg'])
auc_reg

# %%
from IPython.display import display, Markdown
import re

def clean_tex(text):
    return re.sub(r'[_?#$&%]', '', str(text))

for i, classe in enumerate(naive.classes_):
    prior = naive.class_prior_[i]
    display(Markdown(f"### **Classe Y = {classe}** (Prior: {prior:.4f})"))
    
    for j, feat_name in enumerate(X.columns):
        mu = float(naive.theta_[i, j])
        var = float(naive.var_[i, j])
        clean_name = clean_tex(feat_name)
        
        formula = (
            f"$$P(\\text{{{clean_name}}} \\mid Y={classe}) = "
            f"\\frac{{1}}{{\\sqrt{{2\\pi \\cdot {var:.4f}}}}} "
            f"\\exp\\left( -\\frac{{(\\text{{{clean_name}}} - {mu:.4f})^2}}{{2 \\cdot {var:.4f}}} \\right)$$"
        )
        display(Markdown(formula))

display(Markdown(r"$$P(Y=k \mid X) \propto P(Y=k) \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} \exp\left(-\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}\right)$$"))
display(Markdown("---"))

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)
plt.plot(roc_arvore[0], roc_arvore[1], 'o-')
plt.plot(roc_naive[0], roc_naive[1], 'o-')
plt.plot(roc_reg[0], roc_reg[1], 'o-')
plt.grid(True)
plt.title("ROC Curve")
plt.xlabel("1 - Especificidade")
plt.ylabel("Recall")

plt.legend([f"Árvore: {auc_arvore:.2f}", f"Naive: {auc_naive:.2f}",
            f"Regressão: {auc_reg:.2f}"] )

# %% Serializando o modelo em um arquivo binario para ser usado em outro lugar

pd.Series({"model": reg, "features":features}).to_pickle("model_feliz.pkl")