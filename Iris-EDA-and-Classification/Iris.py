
# coding: utf-8

# IRIS DATASET EDA AND CLASSIFICATION USING LOGISTIC REGRESSION/RANDOM FOREST/KNN/NAIVE BAYES/SVC/XGBOOST

# In[1]:


import numpy as np #numpy is used for numeric operations
import pandas as pd# Pandas is just like SQL , It is used for Data manipulation
import matplotlib.pyplot as plt #For Plots

df = pd.read_csv(r'D:\Datasets\iris-species\Iris.csv') # Import the dataset


# In[5]:


df.head() #Shows top 5 rows


# In[3]:


df.info() #Information about the dataset


# In[4]:


df.describe() #Univariare Statistics 


# In[6]:


df['Species'].value_counts()
#To check the class bias present in target variable and it is zero here which can be visualized in the bar plot below


# In[8]:


df['Species'].value_counts().plot.bar() #Frequency distribution for targer variable "Species"


# In[14]:


df.iloc[:,1:5].plot.hist(stacked='False',alpha=0.5) #Unstacked histogram distribution for numeric continuous variables


# In[15]:


df.iloc[:,1:5].hist() #Frequency distribution of each continuous feature present in Iris dataset shown via histogram


# In[16]:


df.boxplot()


# In[19]:


df.iloc[:,1:].boxplot()


# In[24]:


plt.rcParams['figure.figsize'] = [10, 10] #to control the plot size
df.iloc[:,1:].boxplot(by='Species') #Box plot grouped by species


# In[25]:


df.columns


# In[29]:


plt.rcParams['figure.figsize'] = [6,6] #to control the plot size
fig = df[df['Species']=='Iris-versicolor'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',c='green',alpha=0.8)
df[df['Species']=='Iris-virginica'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',c='red',alpha=0.8,ax=fig)
df[df['Species']=='Iris-setosa'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',c='black',alpha=0.8,ax=fig)


# In[31]:


plt.rcParams['figure.figsize'] = [6,6] #to control the plot size
fig = df[df['Species']=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',c='green',alpha=1)
df[df['Species']=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',c='red',alpha=1,ax=fig)
df[df['Species']=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',c='black',alpha=1,ax=fig)


# It is clearly evident that the features PetalWidthCm and PetalLengthCm are highly correlated and the Iris flower species can be clearly distinguished or classified if we use PetalWidthCm and PetalLengthCm as features

# Now let's start modelling.

# In[32]:


#Check for Presence of Null values
df.isnull().sum()


# The result say no null values in the dataset

# In[33]:


#Check for skewness in the numeric columns
df.skew()


# There is not much skewness present in the dataset but the distribution is not gaussian which can be seen below

# In[37]:


plt.rcParams['figure.figsize'] = [8,8]
df.iloc[:,1:].hist()


# When the features do not follow a gaussian distribution it is better to use non-parametric methods for classification

# In[51]:


x = df.iloc[:,1:5]
y = df.iloc[:,5]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)

from sklearn.linear_model import LogisticRegression #logistic Regression
model = LogisticRegression(random_state=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[52]:


model = LogisticRegression(random_state=3)


# In[53]:


model.fit(x_train,y_train)


# In[54]:


y_pred = model.predict(x_test)


# In[57]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[58]:


print(accuracy_score(y_pred,y_test))


# In[59]:


print("Classification_Report",classification_report(y_pred,y_test))


# In[60]:


print("confusion_matrix",classification_report(y_pred,y_test))


# In[65]:


from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
model = RandomForestClassifier(n_estimators=5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy Score is :",accuracy_score(y_pred,y_test))
print("Classification_Report\n",classification_report(y_pred,y_test))
print("confusion_matrix\n",classification_report(y_pred,y_test))
    


# In[66]:


from sklearn.naive_bayes import GaussianNB  #Naive Bayes Algorithm
model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy Score is :",accuracy_score(y_pred,y_test))
print("Classification_Report\n",classification_report(y_pred,y_test))
print("confusion_matrix\n",classification_report(y_pred,y_test))
    


# In[72]:


from sklearn.neighbors import KNeighborsClassifier  #KNN Alogorthm
model = KNeighborsClassifier(n_neighbors=5) 
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy Score is :",accuracy_score(y_pred,y_test))
print("Classification_Report\n",classification_report(y_pred,y_test))
print("confusion_matrix\n",classification_report(y_pred,y_test))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy Score is :",accuracy_score(y_pred,y_test))
print("Classification_Report\n",classification_report(y_pred,y_test))
print("confusion_matrix\n",classification_report(y_pred,y_test))


# In[76]:


from sklearn.svm import SVC 
model = SVC(C=1.0,kernel='rbf')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy Score is :",accuracy_score(y_pred,y_test))
print("Classification_Report\n",classification_report(y_pred,y_test))
print("confusion_matrix\n",classification_report(y_pred,y_test))


# In[78]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy Score is :",accuracy_score(y_pred,y_test))
print("Classification_Report\n",classification_report(y_pred,y_test))
print("confusion_matrix\n",classification_report(y_pred,y_test))

