import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


datos = pd.read_csv('./housing.csv')
#print(datos.head())

#datos.info()

#print(datos.describe())

#datos.hist(figsize=(15, 8), bins=30, edgecolor='black')
#plt.show()

#sb.scatterplot(x=datos['latitude'], y=datos['longitude'], data=datos, hue=datos['median_house_value'], palette='coolwarm')
#plt.show()

#sb.scatterplot(x='latitude', y='longitude', data=datos[ datos.median_income > 14 ], hue=datos['median_house_value'], palette='coolwarm')
#plt.show()

#Tratamiento de la información
datos_na = datos.dropna() 
#datos_na.info()

datos_na = datos_na[datos_na['housing_median_age'] < 50]
datos_na = datos_na[datos_na['median_house_value'] < 500000]
datos_na = datos_na[datos_na['median_income'] < 15]

datos_na['ocean_proximity'].value_counts()

#print(datos_na['ocean_proximity'].value_counts())

datos_na['ocean_proximity']

#print(datos_na['ocean_proximity'])

#datos dummies
dummies = pd.get_dummies(datos_na['ocean_proximity'], dtype=int)

datos_na = pd.concat([datos_na, dummies], axis=1)

datos_na = datos_na.drop('ocean_proximity', axis=1)

#print(datos_na)

datos_na.corr()

#print(datos_na.corr())

sb.set({'figure.figsize':(15,8)})

#sb.heatmap(data=datos_na.corr(), annot=True, cmap='YlGnBu')

#plt.show()

# % de datos relacionados entre sí
datos_na['room_ratio'] = datos_na['total_bedrooms'] / datos_na['total_rooms']
datos_na['incomes_ratio'] = datos_na['median_income'] / datos_na["median_house_value"]
datos_na['househould_ratio'] = datos_na['population'] / datos_na['households']
#print(datos_na)

# % de datos relacionados entre sí en su función logarítmica
datos_na['room_ratio'] = np.log(datos_na['total_bedrooms'] / datos_na['total_rooms'])
datos_na['incomes_ratio'] = np.log(datos_na['median_income'] / datos_na["median_house_value"])
datos_na['househould_ratio'] = np.log(datos_na['population'] / datos_na['households'])
#print(datos_na)

sb.set({'figure.figsize':(15,8)})

sb.heatmap(data = datos_na.corr(), annot=True, cmap='YlGnBu')

#plt.show()

X = datos_na.drop("median_house_value", axis=1) # características de entrada
y = datos_na["median_house_value"] # etiqueta

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

modelo = LinearRegression()

#entrenamiento
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

comparativa = {"predicciones":predicciones, "valor real": y_test}
result = pd.DataFrame(comparativa)
#print(result)

score = modelo.score(X_test, y_test)
print(f'Presición: {r2_score(y_test, predicciones)}')
print(f'Desviación media: {mean_squared_error(y_test, predicciones)}')

#¿El resultado fue mejor o peor? 
#El resultado fue notablemente mejor.

#¿Por qué crees que es así? Por qué son necesarios los cambios aplicados (fundamento del porqué afecta esos cambios)
#Gracias a correlacionar datos similares entre sí y calcular su proporción lo que logró mejorar 
# el rendimiento a un puntaje superior al 80%. Y al aplicar la transformación logarítmica, 
# se eliminó el límite fijo de algunos valores, lo que permitió que el modelo mejorara aún más, 
# alcanzando un puntaje superior al 90%.