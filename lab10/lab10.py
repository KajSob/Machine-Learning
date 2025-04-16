#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle
import keras_tuner as kt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal
from scikeras.wrappers import KerasRegressor



# In[2]:


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# In[3]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
X_train.shape


# In[4]:


param_distribs = {
"model__n_hidden": range(0,4),
"model__n_neurons": range(1,101),
"model__learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
"model__optimizer": ["adam", "sgd", "nesterov"]
}


# In[5]:


def build_model(n_hidden, n_neurons, optimizer, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "momentum":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == "nesterov":
        optimizer = tf.keras.optimizers.SGD(nesterov=True, learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


# In[6]:


es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)


# In[7]:


keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[ ]:


rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=5, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid,y_valid), verbose=0)


# In[ ]:


bob = rnd_search_cv.best_params_
bob


# In[ ]:


with open('rnd_search_params.pkl', 'wb') as f:
    pickle.dump(bob, f)
with open('rnd_search_scikeras.pkl', 'wb') as f:
    pickle.dump(rnd_search_cv, f)


# In[ ]:


def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=3e-4, max_value=3e-2, sampling = "log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam","nesterov"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer == "nesterov":
        optimizer = tf.keras.optimizers.SGD(nesterov=True, learning_rate=learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
    return model 


# In[ ]:


random_search_tuner = kt.RandomSearch(build_model_kt, objective="val_mse", max_trials=10, overwrite=True, directory="my_california_housing", project_name="my_rnd_search", seed=42)


# In[ ]:


root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)


# In[ ]:


random_search_tuner.search(X_train, y_train, epochs=100, callbacks = [tb,es], validation_data=(X_valid, y_valid)) 


# In[ ]:


kt_params = {
    'n_hidden': random_search_tuner.get_best_hyperparameters()[0].get('n_hidden'),
    'n_neurons': random_search_tuner.get_best_hyperparameters()[0].get('n_neurons'),
    'learning_rate': random_search_tuner.get_best_hyperparameters()[0].get('learning_rate'),
    'optimizer': random_search_tuner.get_best_hyperparameters()[0].get('optimizer')
}
kt_params


# In[ ]:


with open('kt_search_params.pkl', 'wb') as f:
    pickle.dump(kt_params, f)


# In[ ]:


best_model = random_search_tuner.hypermodel.build(random_search_tuner.get_best_hyperparameters()[0])
best_model.compile(loss="mse", optimizer=best_model.optimizer, metrics=["mse"])
best_model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[es, tb])


# In[ ]:


best_model.save("kt_best_model.keras")


# In[ ]:




