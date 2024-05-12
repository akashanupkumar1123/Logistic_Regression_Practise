#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('heart.csv')


# In[3]:


df.head()


# In[ ]:





# In[4]:


df['target'].unique()


# In[ ]:





# In[5]:


df.info()


# In[ ]:





# In[6]:


df.describe().transpose()


# In[ ]:





# In[7]:


sns.countplot(x='target',data=df)


# In[ ]:





# In[8]:


df.columns


# In[ ]:





# In[9]:


# Running pairplot on everything will take a very long time to render!
sns.pairplot(df[['age','trestbps', 'chol','thalach','target']],hue='target')


# In[ ]:





# In[10]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)


# In[ ]:





# In[11]:


X = df.drop('target',axis=1)
y = df['target']


# In[ ]:





# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[ ]:





# In[14]:


scaler = StandardScaler()


# In[ ]:





# In[15]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[ ]:





# In[16]:


from sklearn.linear_model import LogisticRegressionCV 


# In[18]:


help(LogisticRegressionCV)


# In[20]:


log_model = LogisticRegressionCV()


# In[21]:


log_model.fit(scaled_X_train,y_train)


# In[ ]:





# In[22]:


log_model.C_


# In[ ]:





# In[23]:


log_model.get_params()


# In[ ]:





# In[24]:


log_model.coef_


# In[ ]:





# In[25]:


coefs = pd.Series(index=X.columns,data=log_model.coef_[0])


# In[26]:


coefs = coefs.sort_values()


# In[ ]:





# In[27]:


plt.figure(figsize=(10,6))
sns.barplot(x=coefs.index,y=coefs.values);


# In[ ]:





# In[28]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix


# In[29]:


y_pred = log_model.predict(scaled_X_test)


# In[30]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# In[31]:


plot_confusion_matrix(log_model,scaled_X_test,y_test)


# In[ ]:





# In[32]:


print(classification_report(y_test,y_pred))


# In[33]:


from sklearn.metrics import plot_precision_recall_curve,plot_roc_curve


# In[34]:


plot_precision_recall_curve(log_model,scaled_X_test,y_test)


# In[35]:


plot_roc_curve(log_model,scaled_X_test,y_test)


# In[36]:


patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]


# In[37]:


X_test.iloc[-1]


# In[38]:


y_test.iloc[-1]


# In[39]:


log_model.predict(patient)


# In[40]:


log_model.predict_proba(patient)


# In[ ]:




