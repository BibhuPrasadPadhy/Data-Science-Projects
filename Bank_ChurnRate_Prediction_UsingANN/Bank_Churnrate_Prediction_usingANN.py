# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 01:32:36 2018

@author: Bibhu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'F:\Artificial_Neural_Networks\train.csv')
df.info()
df.describe()

df_N=df[df['Exited']==0]
df_Y=df[df['Exited']==1]

count_N,count_Y=df['Exited'].value_counts()
df_N_under=df_N.sample(count_Y)
df = pd.concat([df_N_under,df_Y],axis=0)

X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense


#Creating the L=4 layer ANN
classifier = Sequential()
classifier.add(Dense(output_dim=6,kernel_initializer = 'uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,kernel_initializer = 'uniform',activation='relu'))
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compile
classifier.compile(optimizer='adam',loss ='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,epochs=8,batch_size=10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))
