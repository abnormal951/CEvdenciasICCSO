#Se importan las librerias que se van a utilizar

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#se indica que datos se van a importar y de donde 
dataset = pd.read_csv('Salary_Data.csv')
# Genero n (muestras) valores de x aleatorios entre 0 y 100
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# Creo un modelo de regresión lineal

regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Podemos predecir usando el modelo
y_pred = regressor.predict(X_test)
# Representamos el ajuste (rojo) y la recta Y = beta
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()
https://github.com/ejtinajerop/mygit/blob/master/polyfit.py