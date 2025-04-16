#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn import tree
import graphviz
import subprocess
import pickle
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[3]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[4]:


X = X.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


Xb = data_breast_cancer.data[['mean texture', 'mean symmetry']]
yb = data_breast_cancer.target

Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)


# In[6]:


for depth in range(1,21):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(Xb_train,yb_train)
    sc_train = f1_score(yb_train, clf.predict(Xb_train))
    sc_test = f1_score(yb_test, clf.predict(Xb_test))
    print(sc_train,sc_test)


# In[7]:


clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(Xb_train,yb_train)


# In[8]:


tree.export_graphviz(clf, out_file='bc.dot',rounded=True,filled=True)
subprocess.run(["dot", "-Tpng", "bc.dot", "-o" "bc.png"])


# In[9]:


klasa = [3,f1_score(yb_train, clf.predict(Xb_train)),f1_score(yb_test, clf.predict(Xb_test)),accuracy_score(yb_train, clf.predict(Xb_train)),accuracy_score(yb_test, clf.predict(Xb_test))]

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(klasa, f)
klasa


# In[10]:


for depth in range(1,21):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X_train,y_train)
    sc_train = mean_squared_error(y_train, reg.predict(X_train))
    sc_test = mean_squared_error(y_test, reg.predict(X_test))
    print(sc_train,sc_test, sc_train+sc_test)


# In[11]:


reg = DecisionTreeRegressor(max_depth=4, random_state=42)
reg.fit(X_train,y_train)


# In[12]:


regal = [4,mean_squared_error(y_train, reg.predict(X_train)),mean_squared_error(y_test, reg.predict(X_test))]

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(regal, f)
regal


# In[13]:


tree.export_graphviz(reg, out_file='reg.dot',rounded=True,filled=True)
subprocess.run(["dot", "-Tpng", "reg.dot", "-o" "reg.png"])


# In[ ]:




