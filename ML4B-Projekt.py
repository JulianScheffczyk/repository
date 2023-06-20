import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn 

import os
import pathlib

print(os.getcwd())

## set folder for data load
data_path: str = os.path.join("C:\\", "Users", "jsche", "Desktop", "ML4B")

## load data
print("Loading: Gehen")
df_walk_1 = pd.read_json(os.path.join(data_path, "Gehen.14.05-2023-05-14_13-02-33.json"))
print("Loading: Fahrrad")
df_bike_1 = pd.read_json(os.path.join(data_path, "fahrrad_17.05-1-2023-05-17_17-48-35.json"))
print("Loading: Bus")
df_bus_1 = pd.read_json(os.path.join(data_path, "Bus_24.05-1-2023-05-24_10-49-43.json"))

#print(df_walk_1.head())

## check data types
df_types =pd.concat([df_walk_1.dtypes, df_bike_1.dtypes, df_bus_1.dtypes], axis = 1)

df_types.columns = ['Walk', 'Bike', 'Bus']
df_types

## schow koloms of data
df_walk_1.columns
df_bike_1.columns
df_bus_1.columns


df_types =pd.concat([df_walk_1.dtypes, df_bike_1.dtypes, df_bus_1.dtypes], axis = 1)

df_types.columns = ['Walk', 'Bike', 'Bus']
df_types

# data processing 

# convert time to datetime
df_walk_1['time'] = pd.to_datetime(df_walk_1['time'])
df_bike_1['time'] = pd.to_datetime(df_bike_1['time'])
df_bus_1['time'] = pd.to_datetime(df_bus_1['time'])

# set index to time

df_walk_1 = df_walk_1.set_index('time')
df_bike_1 = df_bike_1.set_index('time')
df_bus_1 = df_bus_1.set_index('time')


# zeige die ersten 2 Zeilen der Daten
#print(df_walk_1.head(2), df_bike_1.head(2), df_bus_1.head(2))

#print(df_walk_1.columns)


## die Daten nach time sortieren mit pandas.melt()
#df_walk_1 = pd.melt(df_walk_1, id_vars=['time'], value_vars=['x', 'y', 'z'], var_name='Acc', value_name='Acc_Walk')
#print(df_walk_1)
#df_walk_1 = pd.melt(df_walk_1, id_vars=['time', 'x', 'y', 'z'], var_name='Acc', value_name='Acc_Walk')
#print(df_walk_1.head(10))

# den Daten eine Reihe hinzufügen wo steht ob es sich um gehen, fahrrad oder bus handelt
df_walk_1['Activity'] = 'Walk'
df_bike_1['Activity'] = 'Bike'
df_bus_1['Activity'] = 'Auto'

#print(df_walk_1.head(2), df_bike_1.head(2), df_bus_1.head(2))
#print(df_walk_1.columns)


# die Daten in einen datensatzt verwandeln und die columns auf die anderen mit gleichen namen matchen 
#df_all= pd.concat([df_walk_1, df_bike_1, df_bus_1], axis=1)
#print(df_all.head(10))

# die datem in einem datensatzt joinen und die columns auf die anderen mit gleichen namen matchen
df_all = pd.concat([df_walk_1, df_bike_1, df_bus_1], axis=0)

print(df_all.head(10))
print(df_all.columns)

# Daten mit aktitität bike auswählen
#print(df_all[df_all['Activity'] == 'Bike']) , just for testing if the activity bike is there

# Columns auswählen
#print(df_all[['Activity', 'x', 'y', 'z']])
#print(df_all[['latitude','longitude','altitude', 'Activity']])

# colums aus dem datensatz rausnehmen ausser time und in einem neuen datensatz speichern df_all_new
df_all_new = df_all[[ 'x', 'y', 'z', 'Activity', 'speed', 'qz', 'qy', 'qx','pitch', 'roll', 'yaw' ]]

#print(df_all_new.head(10)) , just for testing if the columns are gone and the new dataset is created
#print(df_all_new[['speed']]) , there are nan values in the dataset

# die columns im df mit nan values ausgeben lassen 
#print(df_all_new.isnull())

## die nan values mit 0 ersetzen
df_all_new = df_all_new.fillna(0)
#print(df_all_new.head(10))

## die Daten im df zufällig sortieren
df_all_new = df_all_new.sample(frac=1).reset_index(drop=True)
print(df_all_new.head(10))

## one hot encoding
df_all_new = pd.get_dummies(df_all_new, columns=['Activity'])
print(df_all_new.head(10))

## die Daten in trainings und testdaten aufteilen 
from sklearn.model_selection import train_test_split    # import train_test_split function from sklearn package
X = df_all_new[['x', 'y', 'z', 'speed', 'qz', 'qy', 'qx','pitch', 'roll', 'yaw']]  # Features
y = df_all_new[['Activity_Auto', 'Activity_Bike', 'Activity_Walk']]  # Labels
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1) # 70% training and 30% test

####################
## die Daten skalieren
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

## ein ML model erstellen
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

## die Daten vorhersagen
y_pred = classifier.predict(X_test)

## die Genauigkeit des Models berechnen
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))







