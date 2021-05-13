#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics (GRIP May21)
# 
# 
# # Author: Ravi charan Baraka
# 
# 
# # # Task 1 : Prediction using supervised ML
# 
# 

# # Problem statement:
# 
#        Predict the percentage of a student based on the number of study hours if a student studies for 9.25 hrs/ day.

# # importing the required libraries

# In[2]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # importing data set

# In[3]:


#reading the data using pandas
df=pd.read_csv("http://bit.ly/w-data")
print("Data imported Successfully")
df.head()


# # Understanding Data

# In[4]:


df.describe()


# In[5]:


df.shape


# In[41]:


#plotting the distribution of scores
df.plot(x='Hours',y='Scores',style='o')
plt.title('Study hours Vs percentage gained')
plt.xlabel('Hours studied')
plt.ylabel('marks scored')
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# # Cleaning the Data

# In[7]:



df.isnull().sum()


# # preparing the data

# In[8]:



x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[33]:


#split the data for training and validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
print("splitting is done")


# In[34]:


#training the algorithm(model)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
print("Training Complete")


# In[40]:


#plotting regression line
line=model.coef_*x+model.intercept_

# Plotting for the test data
plt.scatter(x, y,c="red")
plt.title('Linear Regression vs trained model')
plt.xlabel('Hours studied')
plt.ylabel('Score obtained')
plt.plot(x, line);
plt.show()


# # Predicting values

# In[22]:


y_pred = model.predict(X_test)


# In[24]:


y_pred


# # Comparing Actual Vs Predicted

# In[25]:


compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[26]:


compare


# In[29]:


#testing the accuracy of Model
result=model.score(X_test,y_test)
print(result)


# # Solution for given problem statement:

# In[30]:


hours=9.25
prediction=model.predict([[hours]])
print(prediction)


# # Evaluating the Model

# In[31]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# # Conclusion:
# For a student studying 9.25Hrs a day , the model predicts his score as 93.6917

# In[ ]:




