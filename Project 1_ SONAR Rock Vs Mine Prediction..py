#!/usr/bin/env python
# coding: utf-8

# Importing the dependencies.

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Data PreProcessing.

# In[3]:


# Loading the data in the pandas dataframes.
sonar = pd.read_csv("sonar data.csv", header= None) # Since the data has no head/features.
sonar.head()


# In[4]:


sonar.shape


# In[6]:


sonar.describe() # Gives the statistical description of the data.


# In[11]:


# Finding total number  of rocks and mines in the dataset column 60.
sonar[60].value_counts()


# M --> Mine
# R --> Rock

# In[12]:


# Getting the description of indiviual R and M data.
sonar.groupby(60).mean()


# In[13]:


# Seperating data and the labels. -> Problem of the Supervised Learning.
X = sonar.drop(columns=60, axis=1)
Y = sonar[60]
X.head()


# In[14]:


Y.head()


# Splitting the data into train and test data.

# In[15]:


# Stratify will split the data based on R and M label equally.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, stratify=Y, random_state = 1)


# In[18]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training -> Logistic Regression.

# In[19]:


model = LogisticRegression()


# In[20]:


# Training the Logistic Regression Model with training data.
model.fit(X_train,Y_train)


# Model Evaluation based on Accuracy Score.
# 

# In[22]:


# Accuracy on the training data.
X_train_prediction = model.predict(X_train)
# Compairing X_train_prediction with the Y_train.
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[24]:


print("The accuracy score of the training data is", training_data_accuracy*100, "Percentage")


# In[27]:


# Accuracy on the testing data.
X_test_prediction = model.predict(X_test)
# Compairing X_test_prediction with the Y_test.
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[28]:


print("The accuracy score of the test data is", testing_data_accuracy*100, "Percentage")


# Making a Predictive System.

# In[45]:


input_data = (0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,0.6011,0.6470,0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,0.3862,0.3958,0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,0.0031,0.0099,0.0080,0.0107,0.0161,0.0133)
# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance.
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)


if(prediction[0] == "R"):
    print("The object is a Rock")
else:
    print("The object is a Mine")


# In[ ]:




