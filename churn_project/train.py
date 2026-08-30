# %%
import pandas as pd

df = pd.read_csv("../data/abt_churn.csv")
df.head()

# %%

# definindo out of time (safra mais recente)
oot = df[df["dtRef"]==df['dtRef'].max()].copy()
oot

# %%
