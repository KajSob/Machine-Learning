#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle


# In[2]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()
Xb = data_breast_cancer.data
yb = data_breast_cancer.target


# In[3]:


from sklearn.datasets import load_iris
data_iris = load_iris()
Xi = data_iris.data
yi = data_iris.target


# In[4]:


scalerb = StandardScaler()
scaleri = StandardScaler()
XbS = scalerb.fit_transform(Xb)
XiS = scaleri.fit_transform(Xi)


# In[5]:


pcai = PCA(n_components=0.9)
pcab = PCA(n_components=0.9)
XbD = pcab.fit_transform(XbS)
XiD = pcai.fit_transform(XiS)


# In[6]:


lb = []
li = []
for b in pcab.explained_variance_ratio_:
    lb.append(b)
for i in pcai.explained_variance_ratio_:
    li.append(i)


# In[7]:


with open('pca_bc.pkl', 'wb') as f:
    pickle.dump(lb, f)

lb


# In[8]:


with open('pca_ir.pkl', 'wb') as f:
    pickle.dump(li, f)

li


# In[9]:


pcai.components_


# In[16]:


ii = []
ib = []
for o in pcai.components_:
    maks = max(o)
    mini = min(o)
    if mini*-1>maks:
        maks = mini
    for u in range(len(o)):
        if o[u] == maks:
            ii.append(u)

for z in pcab.components_:
    maks = max(z)
    mini = min(z)
    if mini*-1>maks:
        maks = mini
    print(maks)
    for a in range(len(z)):
        if z[a] == maks:
            ib.append(a)
ii


# In[17]:


ib


# In[18]:


pcab.components_


# In[19]:


with open('idx_bc.pkl', 'wb') as f:
    pickle.dump(ib, f)
with open('idx_ir.pkl', 'wb') as f:
    pickle.dump(ii, f)


# In[ ]:




