# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:06:57 2019

@author: Brian
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
 
ruta_test = 'C://Users//Brian//Desktop//Titanic//test.csv'
ruta_train = 'C://Users//Brian//Desktop//Titanic//train.csv'

df_test = pd.read_csv(ruta_test)
df_train = pd.read_csv(ruta_train)

print(df_test.head())
print(df_train.head())

#Ver cantidad de detos
print('ver cantidad de datos:')
print(df_train.shape)
print(df_test.shape)

#ver tipos datos
print('tipos de datos:')
print(df_train.info())
print(df_test.info())

#ver datos faltantes del dataset
print('ver datos faltantes:')
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

#ver estadisticas:
print('ver estadisticas:')
print(df_train.describe())
print(df_test.describe())


################Prpcesamientp de los datos#########

#cambiar el dato de sexo por datos numericos

df_train['Sex'].replace(['female','male'], [0,1], inplace = True)
df_test['Sex'].replace(['female','male'], [0,1], inplace = True)

#cambiar los datos de embarque por datos numericos
df_train['Embarked'].replace(['Q','S','C'], [0, 1, 2], inplace = True)
df_test['Embarked'].replace(['Q','S','C'], [0, 1, 2], inplace = True)

#Remplazo la edad faltante por la media de edad:
print(df_train['Age'].mean())
print(df_test['Age'].mean())
promedio = 30
df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

#Crear bandas de edad
#Bandas: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100

bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1','2','3','4','5','6','7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)

#se elimina la columa de "cabin" ya que tiene muchos datos perdidos

df_train.drop(['Cabin'], axis = 1, inplace = True)
df_test.drop(['Cabin'], axis = 1, inplace = True)

#elimino las culumnas que no influyen en el analisis
df_train =  df_train.drop(['PassengerId','Name', 'Ticket'], axis = 1)
df_test = df_test.drop(['Name','Ticket'], axis = 1 )

#se elimina las filas con los datos perdidos 

df_train.dropna(axis = 0, how = 'any', inplace = True)
df_test.dropna(axis = 0, how = 'any', inplace = True)

#Verifico los datos

print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())
print(df_train.shape)
print(df_test.shape)
print(df_test.head())
print(df_train.head())

#####Aplicacion de los algoritmos de machine learning########

#Separo la columna con la información de los sobrevivientes
X = np.array(df_train.drop(['Survived'], 1))
y = np.array(df_train['Survived'])

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#Aplicando algoritmo de regresion logistica:

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print('Precision regresion logistica:')
print(logreg.score(X_train, y_train))

#Support Vector Machine

svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print('precision Soporte de Vectores:')
print(svc.score(X_train, y_train))

#Kn Neighbors

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Precision Vecinos Más Cercanos:')
print(knn.score(X_train, y_train))

###########Prediccion utilizando modelos###########

ids = df_test['PassengerId']

##Regresion Logistica

prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis=1))
out_logreg = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_logreg })
print('Predicción Regresión Logística:')
print(out_logreg.head())

#Support Vecotor Machine

prediccion_svc = svc.predict(df_test.drop('PassengerId', axis = 1))
out_svc = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_svc})
print('Prediccion soporte de vectores:')
print(out_svc.head())
#Kn Neighbors
prediccion_knn = knn.predict(df_test.drop('PassengerId', axis = 1))
out_knn = pd.DataFrame({'PassengerId' :ids, 'Survived' : prediccion_knn})
print('Predicción Vecinos más Cercanos:')
print(out_knn.head())