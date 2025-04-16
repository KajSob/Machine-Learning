#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import urllib.request
import gzip


if not os.path.exists('data'):
    os.makedirs('data')
    
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
urllib.request.urlretrieve(url, "data/housing.tgz")

with tarfile.open('data/housing.tgz', 'r:gz') as plik:
    plik.extractall(path='data/')

with open('data/housing.csv', 'rb') as big:
    with gzip.open('data/housing.csv.gz', 'wb') as small:
        small.writelines(big)

os.remove('data/housing.tgz')


# In[2]:


import pandas as pd
df = pd.read_csv('data/housing.csv.gz')
print(df.head())
print()
print(df.info())


# In[3]:


print(df.ocean_proximity.value_counts())
print()
print(df.ocean_proximity.describe())


# In[4]:


import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[5]:


df.plot(kind="scatter", x="longitude", y="latitude",
alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[6]:


df.plot(kind="scatter", x="longitude", y="latitude",
alpha=0.4, figsize=(7,3), colorbar=True,
s=df["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[7]:


df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index":"atrybut","median_house_value":"wspolczynnik_korelacji"}).to_csv('korelacja.csv',index=False)


# In[8]:


import seaborn as sns
sns.pairplot(df)


# In[9]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
test_size=0.2,
random_state=42)
len(train_set),len(test_set)


# In[10]:


train_set.head()


# In[16]:


test_set.head()


# In[18]:


train_set.corr(numeric_only=True)


# In[20]:


test_set.corr(numeric_only=True)


# In[21]:


test_set.to_pickle('test_set.pkl')


# In[22]:


train_set.to_pickle('train_set.pkl')


# In[ ]:




