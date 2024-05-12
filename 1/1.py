#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


print(np.__version__)
print(pd.__version__)
import sys
print(sys.version)
import sklearn
print(sklearn.__version__)


# In[ ]:





# In[5]:


x = np.linspace(-6,6, num =1000)

plt.figure(figsize = (12,8))
plt.plot(x, 1/(1 + np.exp(-x)));
plt.title("sigmoid Function");


# In[ ]:





# In[6]:


tmp = [0, 0.4, 0.6, 0.8,1.0]
tmp


# In[7]:


np.round(tmp)


# In[8]:


np.array(tmp) > 0.7


# In[ ]:





# In[9]:


dataset = [[-2.0011, 0],
           [-1.4654, 0],
           [0.0965, 0],
           [1.3881, 0],
           [3.0641, 0],
           [7.6275, 1],
           [5.3324, 1],
           [6.9225, 1],
           [8.6754, 1],
           [7.6737, 1]]


# In[10]:


coef = [-0.806605464, 0.2573316]


# In[12]:


for row in dataset:
    yhat = 1.0/ (1.0 + np.exp(-coef[0] - coef[1] * row[0]))
    print("yhat {0:.4f}, yhat {1}".format(yhat, round(yhat)))


# In[ ]:





# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


dataset


# In[15]:


X = np.array(dataset)[:, 0:1]
y = np.array(dataset)[:,1]


# In[16]:


X


# In[17]:


y


# In[ ]:





# In[19]:


clf_LR = LogisticRegression(C=1.0, penalty='l2', tol=0.0001 , solver="lbfgs")


# In[20]:


clf_LR.fit(X,y)


# In[ ]:





# In[21]:


clf_LR.predict(X)


# In[ ]:





# In[22]:


clf_LR.predict_proba(X)


# In[ ]:





# In[ ]:





# In[23]:


dataset2 = [[ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.4,  0. ],
            [ 0.3,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.1,  0. ],
            [ 1.4,  1. ],
            [ 1.5,  1. ],
            [ 1.5,  1. ],
            [ 1.3,  1. ],
            [ 1.5,  1. ],
            [ 1.3,  1. ],
            [ 1.6,  1. ],
            [ 1. ,  1. ],
            [ 1.3,  1. ],
            [ 1.4,  1. ]]


# In[ ]:





# In[24]:


X = np.array(dataset2)[:, 0:1]
y = np.array(dataset2)[:, 1]


# In[25]:


clf_LR = LogisticRegression(C=1.0, penalty='l2', tol=0.0001, solver='lbfgs')

clf_LR.fit(X,y)


# In[ ]:





# In[26]:


y_pred = clf_LR.predict(X)
clf_LR.predict(X)


# In[ ]:





# In[27]:


np.column_stack((y_pred, y))


# In[ ]:





# In[28]:


clf_LR.predict(np.array([0.9]).reshape(1,-1))


# In[ ]:




