#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
import os
from tensorflow import keras
import numpy as np
from keras.src.metrics import RootMeanSquaredError


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[3]:


X_train = X_train/255
X_test = X_test/255


# In[4]:


plt.imshow(X_train[568], cmap="binary")
plt.axis('off')
plt.show()


# In[5]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[568]]


# In[6]:


model = models.Sequential()
model.add(layers.Flatten(input_shape=[28, 28]))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[7]:


model.summary()
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# In[8]:


model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",
metrics=["accuracy"])


# In[9]:


root_logdir = os.path.join(os.curdir, "image_logs")
root_logdir


# In[10]:


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
run_logdir


# In[11]:


tb_cb= keras.callbacks.TensorBoard(get_run_logdir())
history = model.fit(X_train, y_train, epochs=20,
validation_split=0.1,
callbacks=[tb_cb])


# In[12]:


image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[13]:


#%load_ext tensorboard


# In[14]:


#%tensorboard --logdir ./image_logs


# In[15]:


model.save("fashion_clf.keras")


# In[16]:


housing = fetch_california_housing()


# In[17]:


X = housing.data
y = housing.target


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# In[19]:


modreg = models.Sequential()
normalizer = keras.layers.Normalization(input_shape=X_train.shape[1:], axis=None)
normalizer.adapt(y)
modreg.add(normalizer)
modreg.add(layers.Dense(50,activation="relu",input_shape=X_train.shape[1:]))
modreg.add(layers.Dense(50,activation="relu",input_shape=X_train.shape[1:]))
modreg.add(layers.Dense(50,activation="relu",input_shape=X_train.shape[1:]))
modreg.add(layers.Dense(1))


# In[20]:


modreg.compile(loss="mean_squared_error",
optimizer="adam",
metrics=[RootMeanSquaredError()])


# In[21]:


ea_cb = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.01,
    verbose=1)


# In[22]:


root_logdir = os.path.join(os.curdir, "housing_logs")
root_logdir


# In[23]:


get_run_logdir()


# In[24]:


tb_cb= keras.callbacks.TensorBoard(get_run_logdir())
history = modreg.fit(X_train, y_train, epochs=100,
validation_data=(X_val, y_val),
callbacks=[ea_cb,tb_cb])


# In[25]:


modreg.save("reg_housing_1.keras")


# In[26]:


m2 = models.Sequential()
normalizer = keras.layers.Normalization(input_shape=X_train.shape[1:], axis=None)
normalizer.adapt(y)
m2.add(normalizer)
m2.add(layers.Dense(100,activation="relu",input_shape=X_train.shape[1:]))
m2.add(layers.Dense(100,activation="relu",input_shape=X_train.shape[1:]))
m2.add(layers.Dense(1))


# In[27]:


m3 = models.Sequential()
normalizer = keras.layers.Normalization(input_shape=X_train.shape[1:], axis=None)
normalizer.adapt(y)
m3.add(normalizer)
m3.add(layers.Dense(20,activation="relu",input_shape=X_train.shape[1:]))
m3.add(layers.Dense(20,activation="relu",input_shape=X_train.shape[1:]))
m3.add(layers.Dense(20,activation="relu",input_shape=X_train.shape[1:]))
m3.add(layers.Dense(20,activation="relu",input_shape=X_train.shape[1:]))
m3.add(layers.Dense(20,activation="relu",input_shape=X_train.shape[1:]))
m3.add(layers.Dense(1))


# In[28]:


m2.compile(loss="mean_squared_error",
optimizer="adam",
metrics=[RootMeanSquaredError()])

m3.compile(loss="mean_squared_error",
optimizer="adam",
metrics=[RootMeanSquaredError()])


# In[29]:


tb_cb= keras.callbacks.TensorBoard(get_run_logdir())
history = m2.fit(X_train, y_train, epochs=100,
validation_data=(X_val, y_val),
callbacks=[ea_cb,tb_cb])


# In[30]:


tb_cb= keras.callbacks.TensorBoard(get_run_logdir())
history = m3.fit(X_train, y_train, epochs=100,
validation_data=(X_val, y_val),
callbacks=[ea_cb,tb_cb])


# In[31]:


m2.save("reg_housing_2.keras")
m3.save("reg_housing_3.keras")


# In[ ]:




