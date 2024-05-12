#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('titanic_train.csv')


# In[3]:


train.head()


# In[ ]:





# In[4]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:





# In[5]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train,palette='RdBu_r')


# In[ ]:





# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train,palette='RdBu_r')


# In[ ]:





# In[8]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')


# In[ ]:





# In[10]:


sns.displot(train['Age'].dropna(),kde=False,color='darkred', bins=30)


# In[ ]:





# In[11]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[ ]:





# In[12]:


sns.countplot(x='SibSp', data=train)


# In[ ]:





# In[13]:


train['Fare'].hist(color='green', bins=40,figsize=(8,4))


# In[ ]:





# In[15]:


import cufflinks as cf
cf.go_offline()


# In[16]:


train['Fare'].iplot(kind='hist', bins= 30,color='green')


# In[ ]:





# In[17]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train,palette='winter')


# In[ ]:





# In[19]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
    
    
        if Pclass ==1:
            return 37
    
        elif Pclass ==2:
            return 29
    
        else:
            return 24
    
    else:
        return Age
    


# In[ ]:





# In[20]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:





# In[21]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:





# In[22]:


train.drop('Cabin',axis=1,inplace=True)


# In[23]:


train.head()


# In[ ]:





# In[24]:


train.dropna(inplace=True)


# In[25]:


train.info()


# In[ ]:





# In[26]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:





# In[27]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:





# In[28]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:





# In[29]:


train.head()


# In[ ]:





# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:





# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[ ]:





# In[34]:


predictions = logmodel.predict(X_test)


# In[ ]:





# In[35]:


from sklearn.metrics import classification_report


# In[36]:


print(classification_report(y_test,predictions))


# In[ ]:




