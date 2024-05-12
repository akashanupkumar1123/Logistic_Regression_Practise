#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[4]:


ad_data = pd.read_csv('advertising.csv')


# In[7]:


ad_data.head()


# In[ ]:





# In[8]:


ad_data.info()


# In[ ]:





# In[9]:


ad_data.describe()


# In[ ]:





# In[10]:


sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# In[ ]:





# In[11]:


sns.jointplot(x='Age', y='Area Income', data = ad_data)


# In[ ]:





# In[14]:


sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data,color='red',kind='kde');


# In[ ]:





# In[15]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')


# In[ ]:





# In[16]:


sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# In[ ]:





# In[17]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[18]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[ ]:





# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:





# In[21]:


from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[22]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:





# In[23]:


predictions = logmodel.predict(X_test)


# In[ ]:





# In[24]:


from sklearn.metrics import classification_report


# In[ ]:





# In[25]:


print(classification_report(y_test,predictions))


# In[ ]:




