#!/usr/bin/env python
# coding: utf-8

#  # Author : Shaun Jose

# ## Task:Prediction using supervised Machine Learning

#    ## Importing all libraries required for this notebook

# In[30]:


import pandas as pd
import numpy as np
from sklearn import metrics 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df


#    ## Plotting the distribution of scores

# In[3]:


df.plot(x='Hours',y='Scores',style='o')
plt.title('Hour v/s Percentage')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()


#    ##  Preparing The Data

# In[4]:


X=df.Hours.values.reshape(-1,1)
X


# In[21]:


Y=df.Scores.values
Y


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#    ## Training the Model

# In[23]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
clf=LinearRegression()
clf.fit(x_train,y_train)
print('Training Completed')


# In[24]:


line = clf.coef_*X+clf.intercept_
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


#    ## Making the Predictions

# In[25]:


y_predicted=clf.predict(x_test)


#    ### Comparing Actual V/S Predicted 

# In[26]:


df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_predicted})  
df_new 


#    ## Checking The Accuracy of the model

# In[27]:


clf.score(x_test,y_test)


#    ## Predicted Score

# In[29]:


hours = 9.25
solution= clf.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(solution[0]))


#    ## Evaluating The Model

# In[36]:


print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_predicted)) 

