# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 22:57:17 2018

@author: rtyr
"""

#Outlier Treatment function , Not used here 
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

#Essesential Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#Replacing ? with NaN first and then dropping the NaN rows
missing_values = ['?']
clevelanda = pd.read_csv('clevelanda.csv' , na_values = missing_values)
clevelanda = clevelanda.dropna()

#Matrix of Features and Output Vector
x = clevelanda.iloc[:,:-1]
y = clevelanda.iloc[:,13]

#One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder_x = OneHotEncoder(categorical_features=[2,6,10,11,12])
x = onehotencoder_x.fit_transform(x).toarray()

#Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


#Standard Scaler
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
print(model)

# make predictions
expected = y_test
predicted = model.predict(x_test)

# summarize the fit of the model
print("\n\n\n\n Classification Report = \n" ,metrics.classification_report(expected, predicted))
print(" Confusion Matrix = \n",metrics.confusion_matrix(expected, predicted))
# Accuracy Score 
from sklearn.metrics import accuracy_score
print(" Accuracy Score = " ,accuracy_score(expected, predicted))

