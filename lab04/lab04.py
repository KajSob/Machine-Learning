#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.metrics import accuracy_score
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=False)
print(data_breast_cancer['DESCR'])


# In[3]:


data_iris = datasets.load_iris()
print(data_iris['DESCR'])


# In[4]:


Xb = data_breast_cancer["data"][:, (3, 4)]
yb = data_breast_cancer["target"]


# In[5]:


Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)


# In[6]:


svm_clf = Pipeline([
("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),
])
wo_scale = LinearSVC(C=1,loss="hinge",random_state=42)


# In[7]:


wo_scale.fit(Xb_train,yb_train)


# In[8]:


svm_clf.fit(Xb_train, yb_train)


# In[9]:


yb_train_predict_wo = wo_scale.predict(Xb_train)
yb_test_predict_wo = wo_scale.predict(Xb_test)
yb_train_predict_sc = svm_clf.predict(Xb_train)
yb_test_predict_sc = svm_clf.predict(Xb_test)


# In[10]:


listb = [accuracy_score(yb_train,yb_train_predict_wo), accuracy_score(yb_test,yb_test_predict_wo), accuracy_score(yb_train,yb_train_predict_sc), accuracy_score(yb_test,yb_test_predict_sc)]
listb


# In[11]:


with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(listb, f)


# In[12]:


Xi = data_iris["data"][:, (2, 3)]
yi = (data_iris["target"] == 2)


# In[13]:


Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, yi, test_size=0.2, random_state=42)


# In[14]:


svm_clfi = Pipeline([
("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),
])
wo_scalei = LinearSVC(C=1,loss="hinge",random_state=42)


# In[15]:


wo_scalei.fit(Xi_train,yi_train)


# In[16]:


svm_clfi.fit(Xi_train, yi_train)


# In[17]:


yi_train_predict_wo = wo_scalei.predict(Xi_train)
yi_test_predict_wo = wo_scalei.predict(Xi_test)
yi_train_predict_sc = svm_clfi.predict(Xi_train)
yi_test_predict_sc = svm_clfi.predict(Xi_test)


# In[18]:


listi = [accuracy_score(yi_train,yi_train_predict_wo), accuracy_score(yi_test,yi_test_predict_wo), accuracy_score(yi_train,yi_train_predict_sc), accuracy_score(yi_test,yi_test_predict_sc)]
listi


# In[19]:


with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(listi, f)


# In[20]:


size = 900
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[21]:


X = X.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


clf = Pipeline([
("poly_features", PolynomialFeatures(degree=4)), ("linear_svr", LinearSVR()),
])


# In[23]:


clf.fit(X_train,y_train)


# In[24]:


mse_li =[mean_squared_error(y_train, clf.predict(X_train)),mean_squared_error(y_test, clf.predict(X_test))]
mse_li


# In[25]:


svr_base = SVR(kernel="poly", degree=4)
svr_base.fit(X_train,y_train)


# In[26]:


mse_bad =[mean_squared_error(y_train, svr_base.predict(X_train)),mean_squared_error(y_test, svr_base.predict(X_test))]
mse_bad


# In[27]:


param_grid = {"coef0": [0.1, 1, 10],"C" : [0.1, 1, 10]}
search = GridSearchCV(svr_base, param_grid, scoring="neg_mean_squared_error", n_jobs=-1)
search.fit(X, y)
print(search.best_params_)


# In[28]:


svr_base = SVR(kernel="poly", degree=4,C=1,coef0=1)
svr_base.fit(X_train,y_train)


# In[29]:


mse_li.append(mean_squared_error(y_train, svr_base.predict(X_train)))
mse_li.append(mean_squared_error(y_test, svr_base.predict(X_test)))
mse_li


# In[30]:


with open('reg_mse.pkl', 'wb') as f:
    pickle.dump(mse_li, f)


# In[ ]:




