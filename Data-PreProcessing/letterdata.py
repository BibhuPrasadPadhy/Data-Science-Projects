# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 08:36:07 2018

@author: Bibhu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('letterdata.csv')
letter_data = dataset.iloc[:,:].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
letter_data[:,0]=labelencoder_x.fit_transform(letter_data[:,0])

onehotcoder = OneHotEncoder(categorical_features=[0])
letter_data = onehotcoder.fit_transform(letter_data).toarray()