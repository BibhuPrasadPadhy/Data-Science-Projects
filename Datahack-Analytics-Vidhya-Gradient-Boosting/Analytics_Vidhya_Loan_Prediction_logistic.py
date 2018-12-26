# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:41:13 2018

@author: Bibhu_Padhy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Import dataset and initial Analysis
df = pd.read_csv(r'C:\Users\Bibhu_Padhy\Desktop\Docs\train.csv')
df.head()
df.describe()

df1 = pd.read_csv(r'C:\Users\Bibhu_Padhy\Desktop\Docs\test.csv')
df1.head()
df1.describe()
#EDA

df['Property_Area'].value_counts()
df['Gender'].value_counts()
df['Married'].value_counts()
df['Dependents'].value_counts()
df['Education'].value_counts()
df['Self_Employed'].value_counts()

df1['Property_Area'].value_counts()
df1['Gender'].value_counts()
df1['Married'].value_counts()
df1['Dependents'].value_counts()
df1['Education'].value_counts()
df1['Self_Employed'].value_counts()

#Univariate Analysis
df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome',by=['Education','Gender'])
df['ApplicantIncome'].skew()

df['CoapplicantIncome'].hist(bins=50)
df['CoapplicantIncome'].skew()

df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount',by='Education')

df1['ApplicantIncome'].hist(bins=50)
df1.boxplot(column='ApplicantIncome',by=['Education','Gender'])
df1['ApplicantIncome'].skew()

df1['CoapplicantIncome'].hist(bins=50)
df1['CoapplicantIncome'].skew()

df1['LoanAmount'].hist(bins=50)
df1.boxplot(column='LoanAmount',by='Education')
#Data-PreProcessing
#Check for number of null values in the dataset and replace as per need

df.isnull().sum() 

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)

df.isnull().sum()

df1.isnull().sum()
df1['Gender'].fillna(df1['Gender'].mode()[0],inplace=True)
df1['Married'].fillna(df1['Married'].mode()[0],inplace=True)
df1['Dependents'].fillna(df1['Dependents'].mode()[0],inplace=True)
df1['Self_Employed'].fillna(df1['Self_Employed'].mode()[0],inplace=True)
df1['Credit_History'].fillna(df1['Credit_History'].mode()[0],inplace=True)

df1['LoanAmount'].fillna(df1['LoanAmount'].mean(),inplace=True)
df1['Loan_Amount_Term'].fillna(df1['Loan_Amount_Term'].mode()[0],inplace=True)
df1.isnull().sum()

#Outlier Treatment
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount'].hist(bins=20,color='Red')
df['LoanAmount_log'].hist(bins=20)
df['LoanAmount_log'].skew()

df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)
df['TotalIncome_log'].skew()

df1['LoanAmount_log'] = np.log(df1['LoanAmount'])
df1['LoanAmount'].hist(bins=20,color='Red')
df1['LoanAmount_log'].hist(bins=20)
df1['LoanAmount_log'].skew()

df1['TotalIncome']=df['ApplicantIncome']+df1['CoapplicantIncome']
df1['TotalIncome_log']=np.log(df1['TotalIncome'])
df1['TotalIncome_log'].hist(bins=20)
df1['TotalIncome_log'].skew()


#Label Encoder
cat_var = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
cat_var1 = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
labelencoder = LabelEncoder()

for i in cat_var:
    df[i]=labelencoder.fit_transform(df[i])
    
for i in cat_var1:
    df1[i]=labelencoder.fit_transform(df1[i])
    
x_train = df[['Dependents','Education','Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area','LoanAmount_log','TotalIncome_log']]
y_train = df['Loan_Status']

x_test =  df1[['Dependents','Education','Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area','LoanAmount_log','TotalIncome_log']]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Prediction
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


df1['Loan_Status'] = y_pred
submission = df1[['Loan_ID','Loan_Status']]

submission["Loan_Status"] = np.where(submission.Loan_Status ==1,"Y","N")

submission.to_csv(r'C:\Users\Bibhu_Padhy\Desktop\Docs\sample_submission1.csv')
# summarize the fit of the model
#print("\n\n\n\n Classification Report = \n" ,metrics.classification_report(y_test, y_pred))
#print(" Confusion Matrix = \n",metrics.confusion_matrix(y_test, y_pred))
# Accuracy Score 
#from sklearn.metrics import accuracy_score
#print(" Accuracy Score = " ,accuracy_score(y_test, y_pred))


