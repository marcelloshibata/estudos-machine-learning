# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja.xlsx")
df.head()

# %%
features = ['temperatura', 'copo', 'espuma', 'cor'] # caracteristicas
target = 'classe' # variavel resposta

X = df[features]
y = df[target]

# O modelo de Decision Tree do Scikit-learn nao aceita Strings em suas features, entao temos que converter pra numeros
X = X.replace({
    "mud": 1, "pint": 2,
    "sim": 1, "não": 0,
    "clara": 0, "escura": 1,
})

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)

# %% 
import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(model, feature_names=features, class_names=model.classes_,
               filled=True)