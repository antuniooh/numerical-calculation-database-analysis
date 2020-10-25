import random
random.seed(42)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

sns.set()

df = pd.read_csv("concrete_data.csv",)
df.columns = ['cement','blast_furnace_slag','fly_ash','water','superplasticizer','coarse_aggregate',
              'fine_aggregate','age','concrete_compressive_strength']

#exibir valores ausentes ou null
print(df.isnull().sum().sort_values(ascending=False)[:10])
print("Número de linhas e colunas no conjunto de treinamento:", df.shape)
attributes = list(df.columns)
#removendo valores nulos
df.dropna()

#preencher os nulos
df.fillna(df.mean(0))

#remover duplicados
df.drop_duplicates()

corr= df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr,annot=True)

#REGRESSAO LINEAR MULTIPLA
y = df['cement']
x = np.column_stack((
    df['cement'],df['blast_furnace_slag'],df['fly_ash'],df['water'],df['superplasticizer'],
    df['coarse_aggregate'],df['fine_aggregate'],df['age'],df['concrete_compressive_strength']))

x = sm.add_constant(x, prepend=True)
res = sm.OLS(y,x).fit()

print(res.params)
print(res.bse)
print(res.summary())

