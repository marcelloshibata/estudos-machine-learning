# %%
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()
X, y = load_iris(return_X_y=True)

# %% Visualizando dataset de iris em tabelas com pandas

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['especie'] = iris.target_names[iris.target]

print(df.head(10))

#%%

decisionTree = tree.DecisionTreeClassifier(random_state=42)
decisionTree.fit(X, y) 

previsionNumber = decisionTree.predict(X)
previsionString = iris.target_names[previsionNumber] 

print(previsionString)

plt.figure(figsize=(15,10), dpi=400)
tree.plot_tree(decisionTree, filled=True, 
            feature_names=iris.feature_names, class_names=iris.target_names)

prob = decisionTree.predict_proba(X)
df_probs = pd.DataFrame(prob, columns=iris.target_names)
print(df_probs.head())