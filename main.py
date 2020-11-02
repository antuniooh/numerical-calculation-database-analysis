#Antonio Muniz  22.119.001 - 0
#Henrique Vital
#Felipe Moreno

#Passo 1 - Definir Database

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sns.set()

df = pd.read_csv("registro01.csv",)
df.columns = df.columns = ['x0','x1','x2','x3','x4','x5','x6',
              'x7','x8','x9', 'target']

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
x = df.iloc[:, :-1].values
y = df.iloc[:, 10].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
result = LinearRegression()
result.fit(X_train, y_train)

print(result.intercept_) 
print(result.coef_)

y_pred= result.predict(X_test)

DF = pd.DataFrame(data=y_test, columns=['y_test'])
DF['y_predict'] = y_pred
print(DF)

#Passo 4 - Gráfico Linear Regressão Multipla

from sklearn.linear_model import LinearRegression 
lm= LinearRegression() 
x = y_test.reshape(-1, 1)
result=lm.fit(x, y_pred) 
y_pred= lm.predict(x) 
plt.scatter(x, y_pred)
plt.plot(x, y_pred, color = 'red')

#Passo 5 - R2

from sklearn import metrics
r_square = metrics.r2_score(y_test, y_pred)
print(r_square)

#Passo 5 - Pearson

#Passo 6 - Spearman