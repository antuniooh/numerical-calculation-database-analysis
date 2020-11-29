#Antonio Muniz  22.119.001 - 0
#Henrique Vital 22119078 - 8
#Felipe Moreno 22.119.058 - 0

#Passo 1 - Definir Database

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/concrete_data.csv')

#Passo 2 - Limpar Database
#exibir valores ausentes ou null
df.isnull().sum().sort_values(ascending=False)[:10]
print("Número de linhas e colunas no conjunto de treinamento:", df.shape)
attributes = list(df.columns)
#removendo valores nulos
df.dropna()

#preencher os nulos
df.fillna(df.mean(0))

#remover duplicados
df.drop_duplicates()

#Passo 3 - Linear Regressão Multipla
# Importing the dataset
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(regressor.intercept_) 
print(regressor.coef_)

DF = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(DF)

#Passo 4 - Gráfico Linear Regressão Multipla
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()

#Passo 5 - R2
import statsmodels.api as sm

x = sm.add_constant(y_test, prepend=True)
res = sm.OLS(y_pred,x).fit()

print("R2: " + str(res.rsquared))
print("R2 ajusted: " + str(res.rsquared_adj))

from sklearn import metrics
r_square = metrics.r2_score(y_test, y_pred)
print(r_square)

#Passo 5 - Pearson
df.corr(method='pearson')

#Passo 6 - Spearman
df.corr(method='spearman')