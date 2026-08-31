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

# %%

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y,
                                                                    )

# %%
print("Taxa variavel resposta Treino: ", y_train.mean())
print("Taxa variavel resposta Teste: ", y_test.mean())