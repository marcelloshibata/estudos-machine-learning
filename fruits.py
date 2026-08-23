# %%
import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df

# %%

# modelo do scikit que possui a árvore
from sklearn import tree
arvore = tree.DecisionTreeClassifier(random_state=42)

# %%

y = df['Fruta']
caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]

x = df[caracteristicas]

# %%

# Isso é MACHINE LEARNING
arvore.fit(x, y) # X corresponde as co-variáveis enquanto Y as respostas.

# %%
arvore.predict([[1,1,1,1]]) # Pedindo pra maquina fazer uma previsão de qual fruta será se corresponder a todas as características

# %% print da arvore de decisao
import matplotlib.pyplot as plt

plt.figure(dpi=400, figsize=[4,4])
tree.plot_tree(arvore, feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True)

# %%

proba = arvore.predict_proba([[1,1,1,1]])[0] # probabilidade 
pd.Series(proba, index=arvore.classes_) # tabela com as probabilidades e o valor das classes respectivamente