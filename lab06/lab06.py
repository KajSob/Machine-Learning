#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
data_breast_cancer.feature_names


# In[3]:


X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


tree = DecisionTreeClassifier()
log = LogisticRegression()
knc = KNeighborsClassifier()

vot_hard = VotingClassifier(
estimators=[('tree', tree),('log', log),('knc', knc)],
voting='hard')

vot_soft = VotingClassifier(
estimators=[('tree', tree),('log', log),('knc', knc)],
voting='soft')


# In[5]:


tree.fit(X_train, y_train)
log.fit(X_train, y_train)
knc.fit(X_train, y_train)
vot_hard.fit(X_train, y_train)
vot_soft.fit(X_train, y_train)


# In[6]:


acc_l = [[accuracy_score(y_train,tree.predict(X_train)),accuracy_score(y_test,tree.predict(X_test))],
         [accuracy_score(y_train,log.predict(X_train)),accuracy_score(y_test,log.predict(X_test))],
         [accuracy_score(y_train,knc.predict(X_train)),accuracy_score(y_test,knc.predict(X_test))],
         [accuracy_score(y_train,vot_hard.predict(X_train)),accuracy_score(y_test,vot_hard.predict(X_test))],
         [accuracy_score(y_train,vot_soft.predict(X_train)),accuracy_score(y_test,vot_soft.predict(X_test))]
        ]

with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(acc_l, f)

acc_l


# In[7]:


klas = [tree,log,knc,vot_hard,vot_soft]
with open('vote.pkl', 'wb') as f:
    pickle.dump(klas, f)


# In[8]:


bag = BaggingClassifier(n_estimators=30)
bag50 = BaggingClassifier(n_estimators=30,max_samples=0.5)
past = BaggingClassifier(n_estimators=30,bootstrap = False)
past50 = BaggingClassifier(n_estimators=30,max_samples=0.5,bootstrap = False)
rand = RandomForestClassifier(n_estimators=30)
ada = AdaBoostClassifier(n_estimators=30)
grad = GradientBoostingClassifier(n_estimators=30)


# In[9]:


bag.fit(X_train, y_train)
bag50.fit(X_train, y_train)
past.fit(X_train, y_train)
past50.fit(X_train, y_train)
rand.fit(X_train, y_train)
ada.fit(X_train, y_train)
grad.fit(X_train, y_train)


# In[10]:


acc_l2 = [[accuracy_score(y_train,bag.predict(X_train)),accuracy_score(y_test,bag.predict(X_test))],
         [accuracy_score(y_train,bag50.predict(X_train)),accuracy_score(y_test,bag50.predict(X_test))],
         [accuracy_score(y_train,past.predict(X_train)),accuracy_score(y_test,past.predict(X_test))],
         [accuracy_score(y_train,past50.predict(X_train)),accuracy_score(y_test,past50.predict(X_test))],
         [accuracy_score(y_train,rand.predict(X_train)),accuracy_score(y_test,rand.predict(X_test))],
         [accuracy_score(y_train,ada.predict(X_train)),accuracy_score(y_test,ada.predict(X_test))],
         [accuracy_score(y_train,grad.predict(X_train)),accuracy_score(y_test,grad.predict(X_test))]
        ]

with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(acc_l2, f)

acc_l2


# In[11]:


klas2 = [bag,bag50,past,past50,rand,ada,grad]
with open('bag.pkl', 'wb') as f:
    pickle.dump(klas2, f)
klas2


# In[12]:


Xall = data_breast_cancer.data
y = data_breast_cancer.target

Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xall, y, test_size=0.2, random_state=42)


# In[13]:


bag_fea = BaggingClassifier(n_estimators=30, max_features=2,max_samples=0.5,bootstrap_features=False)


# In[14]:


bag_fea.fit(Xa_train,ya_train)


# In[15]:


fea_l = [accuracy_score(ya_train,bag_fea.predict(Xa_train)),accuracy_score(ya_test,bag_fea.predict(Xa_test))]

with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(fea_l, f)
fea_l


# In[16]:


klas3 = [bag_fea]
with open('fea.pkl', 'wb') as f:
    pickle.dump(klas3, f)
klas3


# In[17]:


tier = pd.DataFrame({"acc_train":[],"acc_test":[],"fea_names":[]})
bonk = bag_fea.estimators_[0]
for i in range(len(bag_fea.estimators_)):
    Xi = data_breast_cancer.data[[data_breast_cancer.feature_names[bag_fea.estimators_features_[i][0]],data_breast_cancer.feature_names[bag_fea.estimators_features_[i][1]]]]
    y = data_breast_cancer.target
    Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, y, test_size=0.2, random_state=42)

    bonk = bag_fea.estimators_[i]
    bonk.fit(Xi_train, yi_train)
    tier.loc[i] = [accuracy_score(yi_train,bonk.predict(Xi_train)),accuracy_score(yi_test,bonk.predict(Xi_test)),f"{data_breast_cancer.feature_names[bag_fea.estimators_features_[i][0]]}, {data_breast_cancer.feature_names[bag_fea.estimators_features_[i][1]]}"]
tier = tier.sort_values(["acc_train","acc_test"],ascending=False)
tier


# In[18]:


tier.to_pickle("acc_fea_rank.pkl")


# In[ ]:




