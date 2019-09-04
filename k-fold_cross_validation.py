import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
'''
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
print (X)
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

 

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
'''

data = pd.read_csv('dataset.csv')

#Seleccionamos las variables independientes
X = data[['NS','surface_total_in_m2','rooms']]

#Defino la variable dependiente
Y = data ['price']

# Create linear regression object
regr = linear_model.LinearRegression()

# KFOLD
kf= KFold(n_splits=10)

error = 0
for valores_train, valores_test in kf.split(X):
	X_train = X.iloc[valores_train]
	X_test = X.iloc[valores_test]
	y_train = Y.iloc[valores_train]
	y_test = Y.iloc[valores_test]
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
	print (y_pred)
	error += mean_squared_error(y_test, y_pred)
	print(' \n Error Cuadrático Medio: \n ', mean_squared_error(y_test, y_pred), '\n')

print ('\n Promedio de Errores CROSS VALIDATION: ', error/10)
