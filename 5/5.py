#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[3]:


df = pd.read_csv('iris.csv')


# In[4]:


df.head()


# In[ ]:





# In[5]:


df.info()


# In[ ]:





# In[6]:


df.describe()


# In[ ]:





# In[7]:


df['species'].value_counts()


# In[ ]:





# In[8]:


sns.countplot(df['species'])


# In[ ]:





# In[9]:


sns.scatterplot(x='sepal_length',y='sepal_width',data=df,hue='species')


# In[ ]:





# In[10]:


sns.scatterplot(x='petal_length',y='petal_width',data=df,hue='species')


# In[ ]:





# In[11]:


sns.pairplot(df,hue='species')


# In[ ]:





# In[12]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:





# In[13]:


df['species'].unique()


# In[ ]:





# In[14]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = df['species'].map({'setosa':0, 'versicolor':1, 'virginica':2})
ax.scatter(df['sepal_width'],df['petal_width'],df['petal_length'],c=colors);


# In[ ]:





# In[15]:


X = df.drop('species',axis=1)
y = df['species']


# In[ ]:





# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[18]:


scaler = StandardScaler()


# In[19]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[ ]:





# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


from sklearn.model_selection import GridSearchCV


# In[23]:


#Depending on warnings you may need to adjust max iterations allowed 
# Or experiment with different solvers
log_model = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000)


# In[24]:


# Penalty Type
penalty = ['l1', 'l2']

# Use logarithmically spaced C values (recommended in official docs)
C = np.logspace(0, 4, 10)


# In[25]:


grid_model = GridSearchCV(log_model,param_grid={'C':C,'penalty':penalty})


# In[26]:


grid_model.fit(scaled_X_train,y_train)


# In[27]:


grid_model.best_params_


# In[28]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix


# In[29]:


y_pred = grid_model.predict(scaled_X_test)


# In[30]:


accuracy_score(y_test,y_pred)


# In[31]:


confusion_matrix(y_test,y_pred)


# In[32]:


plot_confusion_matrix(grid_model,scaled_X_test,y_test)


# In[33]:


# Scaled so highest value=1
plot_confusion_matrix(grid_model,scaled_X_test,y_test,normalize='true')


# In[34]:


print(classification_report(y_test,y_pred))


# In[35]:


from sklearn.metrics import roc_curve, auc


# In[36]:


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(5,5)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


# In[37]:


plot_multiclass_roc(grid_model, scaled_X_test, y_test, n_classes=3, figsize=(16, 10))


# In[ ]:




