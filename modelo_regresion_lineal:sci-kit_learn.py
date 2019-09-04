#!/usr/bin/env python
# coding: utf-8

# In[85]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


# In[123]:


#regr = linear_model.LinearRegression()
dataset = pd.read_csv('dataset.csv')
df = pd.DataFrame(dataset)
X = df[['NS','surface_total_in_m2', 'rooms']]
Y = df["price"]
regr = linear_model.LinearRegression()
regr.fit(X, Y)

#print('Intercept: \n', regr.intercept_)
print('Coefficients all dataset: \n', regr.coef_)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2) 
regr.fit(X_train, y_train)
print('Predict: \n', regr.predict(X_test))
print('Coefficients train: \n', regr.coef_)

# Mostramos resultados
plt.plot(X_train, y_train, 'ro', color = 'red', label='Datos Originales')
plt.title('Modelo de regresion lineal')
plt.plot(X_train, regr.predict(X_train), 'ro', color='blue', label='Recta de regresion')
plt.legend()
plt.show()

