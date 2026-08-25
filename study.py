# %%
import pandas as pd
from sklearn import linear_model
from sklearn import tree

# dados de treino
dados = {
    'horas_estudo': [1, 2, 3, 4, 5, 6, 7, 8],
    'nota_exame': [35, 40, 52, 60, 68, 75, 82, 93]
}

df = pd.DataFrame(dados)

X = df[['horas_estudo']]
y = df['nota_exame']

df.head(10)

# %%
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)
a, b = reg.intercept_, reg.coef_[0]
print(a, b)
predict_reg = reg.predict(X)

dec = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
dec.fit(X, y)
predict_dec = dec.predict(X)

# %%
import matplotlib.pyplot as plt

plt.plot(X['horas_estudo'], y, 'o')
plt.grid(True)
plt.title("Relação Horas de Estudo vs Nota")
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota")
plt.plot(X['horas_estudo'], predict_reg)
plt.plot(X['horas_estudo'], predict_dec, color='red')
plt.legend(['Sample', f'y = {a:.3f} + {b:.3f} x', 'Decision Tree'])