# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 07:14:56 2018

@author: Bibhu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Imported the dataset WAOFM(Renamed)
dataset = pd.read_csv('WAOFM.csv')

#Changed Pandas DataFrame to Numpy Object
dataset = dataset.iloc[:,:].values

#Imported StandardScaler & Imputer
from sklearn.preprocessing import StandardScaler, Imputer

#Applied Imputer on all the missing values replacing Missing value with mean
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(dataset[:,4:])
dataset[:,4:] = imputer.transform(dataset[:,4:])

#After replacing missing value in 
scaler = StandardScaler()
dataset[:,4:] = scaler.fit_transform(dataset[:,4:])

#Final Dataset
waofm_data = pd.DataFrame(dataset)
