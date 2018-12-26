#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear regression : Strength prediction on concrete data with Outlier Treatment and skew treatment
@author: Bibhu
"""


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

concrete = pd.read_csv('concrete.csv')

#Outlier Treatment using IQR method
concrete_out = remove_outlier(concrete,'slag')
concrete_out = remove_outlier(concrete,'water')
concrete_out = remove_outlier(concrete,'superplastic')
concrete_out = remove_outlier(concrete,'strength')
concrete = concrete_out

x=concrete.iloc[:,:-1]
y=concrete.iloc[:,8]

# Square Root Transformation
x= np.sqrt(x)
y= np.sqrt(y)


# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=3)

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(X_test)


y_test = np.square(y_test)
y_pred = np.square(y_pred)

# The coefficients
print('Coefficients', regr.coef_)
print('Intercept',regr.intercept_)

# The mean squared error
# MSE = 1/n(yactual- ypredicted)2
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,y_pred))
r2_score(y_test,y_pred)

#Plot outputs
plt.hist(y_test, color='Red')
plt.hist(y_pred, color='blue')
#Residual Plot


plt.scatter(regr.predict(X_train),regr.predict( X_train)- y_train, c= 'b', s=40, alpha = 0.5)
plt.scatter(regr.predict(X_test), regr.predict(X_test)-y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax= 50)
plt.title('Residual Plot')






























