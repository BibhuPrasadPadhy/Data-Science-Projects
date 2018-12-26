# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 13:18:00 2018

@author: Bibhu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from scipy.stats import skew 
from scipy.stats import kurtosis


concrete = pd.read_csv('concrete.csv')
corr_matrix = concrete.corr()
x = concrete[['cement','slag','ash','water','superplastic','age']]
y = concrete.iloc[:,8]

#log transformation of dataset to remove skewness
x_new = np.sqrt(x)
y_new = np.sqrt(y)


from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 3)
x_train,x_test,y_train,y_test = train_test_split(x_new,y_new,test_size = 0.25,random_state = 3)

sc_x=StandardScaler()
#￼
#
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)


regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

actual_y_pred = np.square(y_pred)
actual_y_test = np.square(y_test)

# The coefficients
print('Coefficients', regressor.coef_)
print('Intercept',regressor.intercept_)
# The mean squared error

# MSE = 1/n(yactual- ypredicted)2
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(actual_y_test,actual_y_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test,y_pred))
print('Variance score: %.2f' % r2_score(actual_y_test,actual_y_pred))
#r2_score(y_test, y_pred)
