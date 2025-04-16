#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


# In[2]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[3]:


X = X.reshape(-1,1)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
X_new = np.array([[-3], [3]])
print(lin_reg.intercept_, lin_reg.coef_, "\n", lin_reg.predict(X_new))


# In[6]:


import matplotlib.pyplot as plt

# Wykres punktowy danych
plt.scatter(X_train, y_train, label='Dane treningowe')
plt.scatter(X_test, y_test, color='red', label='Dane testowe')
plt.plot(X_new, lin_reg.predict(X_new), color='black', label='Regresja liniowa')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Regresja liniowa')
plt.show()


# In[7]:


import sklearn.neighbors
knn_reg3 = sklearn.neighbors.KNeighborsRegressor(
n_neighbors=3)
knn_reg3.fit(X_train, y_train)
print(knn_reg3.predict(X_new))


# In[8]:


knn_reg5 = sklearn.neighbors.KNeighborsRegressor(
n_neighbors=5)
knn_reg5.fit(X_train, y_train)
print(knn_reg5.predict(X_new))


# In[9]:


plt.clf()
plt.scatter(X, y, c="blue")
X_new1 = np.arange(-3, 3, 0.001).reshape(-1,1)
plt.plot(X_new1, knn_reg3.predict(X_new1), c="red")
plt.show()


# In[10]:


plt.clf()
plt.scatter(X, y, c="blue")
X_new1 = np.arange(-3, 3, 0.001).reshape(-1,1)
plt.plot(X_new1, knn_reg5.predict(X_new1), c="red")
plt.show()


# In[11]:


from sklearn.preprocessing import PolynomialFeatures
poly_features2=PolynomialFeatures(degree=2,include_bias=False)
X_poly2 = poly_features2.fit_transform(X_train)
print(X_train[0], X_poly2[0])
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2, y_train)
print(lin_reg2.intercept_, lin_reg2.coef_)
print(lin_reg2.predict(poly_features2.fit_transform([[0],[2]])))


# In[12]:


plt.scatter(X_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(X_test, y_test, color='black', label='Dane testowe')
X_poly_plot2 = poly_features2.transform(X_train)
y_poly_pred2 = lin_reg2.predict(X_poly_plot2)

sorted_indices = X_train.flatten().argsort()
plt.plot(X_train[sorted_indices], y_poly_pred2[sorted_indices], color='red', label='Polynomial Regression')
plt.legend()
plt.show()


# In[13]:


poly_features3=PolynomialFeatures(degree=3,include_bias=False)
X_poly3 = poly_features3.fit_transform(X_train)
print(X_train[0], X_poly3[0])
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3, y_train)
print(lin_reg3.intercept_, lin_reg3.coef_)
print(lin_reg3.predict(poly_features3.fit_transform([[0],[2]])))


# In[14]:


plt.scatter(X_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(X_test, y_test, color='black', label='Dane testowe')
X_poly_plot3 = poly_features3.transform(X_train)
y_poly_pred3 = lin_reg3.predict(X_poly_plot3)

sorted_indices = X_train.flatten().argsort()
plt.plot(X_train[sorted_indices], y_poly_pred3[sorted_indices], color='red', label='Polynomial Regression')
plt.legend()
plt.show()


# In[15]:


poly_features4=PolynomialFeatures(degree=4,include_bias=False)
X_poly4 = poly_features4.fit_transform(X_train)
print(X_train[0], X_poly4[0])
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly4, y_train)
print(lin_reg4.intercept_, lin_reg4.coef_)
print(lin_reg4.predict(poly_features4.fit_transform([[0],[2]])))


# In[16]:


plt.scatter(X_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(X_test, y_test, color='black', label='Dane testowe')
X_poly_plot4 = poly_features4.transform(X_train)
y_poly_pred4 = lin_reg4.predict(X_poly_plot4)

sorted_indices = X_train.flatten().argsort()
plt.plot(X_train[sorted_indices], y_poly_pred4[sorted_indices], color='red', label='Polynomial Regression')
plt.legend()
plt.show()


# In[17]:


poly_features5=PolynomialFeatures(degree=5,include_bias=False)
X_poly5 = poly_features5.fit_transform(X_train)
print(X_train[0], X_poly5[0])
lin_reg5 = LinearRegression()
lin_reg5.fit(X_poly5, y_train)
print(lin_reg5.intercept_, lin_reg5.coef_)
print(lin_reg5.predict(poly_features5.fit_transform([[0],[2]])))


# In[18]:


plt.scatter(X_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(X_test, y_test, color='black', label='Dane testowe')
X_poly_plot5 = poly_features5.transform(X_train)
y_poly_pred5 = lin_reg5.predict(X_poly_plot5)

sorted_indices = X_train.flatten().argsort()
plt.plot(X_train[sorted_indices], y_poly_pred5[sorted_indices], color='red', label='Polynomial Regression')
plt.legend()
plt.show()


# In[19]:


mse_ar = [
    [mean_squared_error(y_train, lin_reg.predict(X_train)),mean_squared_error(y_test, lin_reg.predict(X_test))],
    [mean_squared_error(y_train, knn_reg3.predict(X_train)),mean_squared_error(y_test, knn_reg3.predict(X_test))],
    [mean_squared_error(y_train, knn_reg5.predict(X_train)),mean_squared_error(y_test, knn_reg5.predict(X_test))],
    [mean_squared_error(y_train, lin_reg2.predict(poly_features2.transform(X_train))),mean_squared_error(y_test, lin_reg2.predict(poly_features2.transform(X_test)))],
    [mean_squared_error(y_train, lin_reg3.predict(poly_features3.transform(X_train))),mean_squared_error(y_test, lin_reg3.predict(poly_features3.transform(X_test)))],
    [mean_squared_error(y_train, lin_reg4.predict(poly_features4.transform(X_train))),mean_squared_error(y_test, lin_reg4.predict(poly_features4.transform(X_test)))],
    [mean_squared_error(y_train, lin_reg5.predict(poly_features5.transform(X_train))),mean_squared_error(y_test, lin_reg5.predict(poly_features5.transform(X_test)))],
]


# In[20]:


idx=["lin_reg", "knn_3_reg", "knn_5_reg","poly_2_reg", "poly_3_reg", "poly_4_reg", "poly_5_reg"]
col=["train_mse", "test_mse"]
mse_df = pd.DataFrame(mse_ar,index=idx,columns=col)
mse_df


# In[21]:


mse_df.to_pickle('mse.pkl')


# In[24]:


reg = [(lin_reg, None), (knn_reg3, None), (knn_reg5, None), (lin_reg2,
poly_features2), (lin_reg3, poly_features3), (lin_reg4, poly_features4),
(lin_reg5, poly_features5)]
with open('reg.pkl', 'wb') as f:
    pickle.dump(reg, f)


# In[23]:





# In[ ]:




