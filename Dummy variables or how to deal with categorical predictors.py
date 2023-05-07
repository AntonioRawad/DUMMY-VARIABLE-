#!/usr/bin/env python
# coding: utf-8

# ## Dummy variables or how to deal with categorical predictors

# # Import the relevant libraries

# In[26]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
# We can override the default matplotlib styles with those of Seaborn
sns.set()


# # Load the data

# In[27]:


raw_data = pd.read_csv('C:/Users/rawad/OneDrive/Desktop/aws Restart course/Udemy Data Science Course/exercise/1.03.+Dummies.csv')



# In[24]:


raw_data 

# attendence is categorical data we can not use it the methods , so we need to transform the yes to 1 and no to zero 
# In[29]:


data = raw_data.copy()



# In[30]:


data['Attendance'] = data['Attendance'].replace({'No': 0, 'Yes': 1})


# In[31]:


data


# In[32]:


data.describe()


# # REGRESSION EXPRESSION 

# In[33]:


y = data['GPA']
x1 = data[['SAT', 'Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())


# In[35]:


y = data['GPA']
plt.scatter(data['SAT'], y)

yhat_no = 0.6439 + 0.0014 * data['SAT']
yhat_yes = 0.8665 + 0.0014 * data['SAT']

fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='#a50026')

plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)

plt.show()


# In[ ]:


#


# In[43]:


y = data['GPA']
plt.scatter(data['SAT'], y,c=data['Attendance'],cmap='RdYlGn_r')

yhat_no = 0.6439 + 0.0014 * data['SAT']
yhat_yes = 0.8665 + 0.0014 * data['SAT']

fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='#a50026')

plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)

plt.show()


# # CLRAERLY we can assume that students who has attended the classes has a higher GPA 
# lets add the original regression line 
# In[48]:


y = data['GPA']
plt.scatter(data['SAT'], y, c=data['Attendance'], cmap='RdYlGn_r')

yhat_no = 0.6439 + 0.0014 * data['SAT']
yhat_yes = 0.8665 + 0.0014 * data['SAT']
yhat = 0.0017*data['SAT'] + 0.275

fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='#a50026')
fig = plt.plot(data['SAT'], yhat, lw=4, c='orange', label='regression line')

plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




