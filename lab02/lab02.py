#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[2]:


mnist = fetch_openml('mnist_784',version=1)


# In[3]:


mnist.frame.info()


# In[4]:


mnist.frame.head()


# In[5]:


print((np.array(mnist.data.loc[0]).reshape(28, 28) > 0).astype(int))


# In[6]:


X = mnist['data']
y = mnist['target']
y.index


# In[7]:


y2 = y.cat.set_categories(['0', '1', '2','3','4','5','6','7','8','9'], ordered=True)
boo = y2.argsort()
y2 = y2.sort_values()
X2 = X.reindex(boo)
y2.index


# In[8]:


X_train, X_test = X2[:56000], X2[56000:]
y_train, y_test = y2[:56000], y2[56000:] 
print(X_train.shape, y_train.shape) 
print(X_test.shape, y_test.shape)


# In[9]:


print(y_train.values)
y_test.values


# In[10]:


print((np.array(X.loc[25980]).reshape(28, 28) > 0).astype(int))


# In[11]:


X


# In[12]:


y


# In[13]:


X_train


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


X_train


# In[16]:


print(y_train.values)
y_test.values


# In[17]:


y_train_bin = (y_train == '0').astype(int)
y_test_bin = (y_test == '0').astype(int)


# In[18]:


sgd1 = SGDClassifier(random_state=42)
sgd1.fit(X_train,y_train_bin)


# In[19]:


acc_test = accuracy_score(y_test_bin, sgd1.predict(X_test))
acc_test


# In[20]:


acc_train = accuracy_score(y_train_bin, sgd1.predict(X_train))
acc_train


# In[21]:


acc_list = [acc_train,acc_test]
acc_list


# In[22]:


krzyz = cross_val_score(sgd1, X_train, y_train_bin, cv=3, scoring='accuracy')


# In[23]:


krzyz


# In[24]:


sgd2 = SGDClassifier(random_state=42)
sgd2.fit(X_train,y_train)


# In[25]:


blad = confusion_matrix(y_test, sgd2.predict(X_test))
blad


# In[26]:


with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(acc_list, f)


# In[27]:


with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(krzyz, f)


# In[28]:


with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(blad, f)


# In[ ]:





