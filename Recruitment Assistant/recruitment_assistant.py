# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:33:31 2018

@author: Bibhu_Padhy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mldata_train.csv')

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
cat_var = ['role','skill1','skill2','skill3']
for i in cat_var:
    df[i] = le.fit_transform(df[i])

x = df.iloc[:,:-1]
y = df.iloc[:,6]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state = 5)

from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.fit(x_train,y_train)
print("Feature Importance",dc.feature_importances_)
pred = dc.predict(x_test)

from sklearn.metrics import classification_report , confusion_matrix ,accuracy_score
print("Classification Report\n",classification_report(pred,y_test))
print("Confusion Matrix",confusion_matrix(pred,y_test))
print("Accuracy Score",accuracy_score(pred,y_test))