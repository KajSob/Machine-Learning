#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN


# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[3]:


km8 = KMeans(n_clusters = 8, n_init = 10)
km9 = KMeans(n_clusters = 9, n_init = 10)
km10 = KMeans(n_clusters = 10, n_init = 10)
km11 = KMeans(n_clusters = 11, n_init = 10)
km12 = KMeans(n_clusters = 12, n_init = 10)


# In[4]:


y8 = km8.fit_predict(X)
y9 = km9.fit_predict(X)
y10 = km10.fit_predict(X)
y11 = km11.fit_predict(X)
y12 = km12.fit_predict(X)


# In[5]:


sil_km = [silhouette_score(X,y8),silhouette_score(X,y9),silhouette_score(X,y10),silhouette_score(X,y11),silhouette_score(X,y12)]


# In[6]:


sil_km


# In[7]:


with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(sil_km, f)


# In[8]:


neo = confusion_matrix(y, y10)


# In[9]:


neo


# In[10]:


pill = set()
for i in neo:
    pill.add(i.argmax())
pill = sorted(pill)
pill
#??


# In[11]:


with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(pill, f)
#??


# In[12]:


loompa = []
for i in range(0,300):
    for j in X:
        bom = np.linalg.norm(X[i]-j)
        if bom != 0:
            loompa.append(bom)
loompa = sorted(loompa)
umpa = loompa[:10]
umpa


# In[13]:


with open('dist.pkl', 'wb') as f:
    pickle.dump(umpa, f)


# In[15]:


s = (umpa[0]+umpa[1]+umpa[2])/3

e = s
db_l =[]
while(e<=s+(0.1*s)):
    db_l.append(DBSCAN(eps=e))
    e+=0.04*s
db_l


# In[16]:


for i in db_l:
    i.fit(X)


# In[32]:


end = []
for i in db_l:
    end.append(len(set(i.labels_)))
end


# In[30]:


with open('dbscan_len.pkl', 'wb') as f:
    pickle.dump(end, f)


# In[ ]:




