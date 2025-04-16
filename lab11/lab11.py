#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import partial
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


# In[2]:


[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
"tf_flowers",
split=['train[:10%]', "train[10%:25%]", "train[25%:]"],
as_supervised=True,
with_info=True)


# In[3]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[4]:


plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9)
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")
plt.show(block=False)


# In[5]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label


# In[6]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[7]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()


# In[8]:


DefaultConv2D = partial(keras.layers.Conv2D,
kernel_size=3,
activation='relu',
padding="SAME")


# In[9]:


model = keras.models.Sequential([
tf.keras.layers.Rescaling(scale=1./255),
DefaultConv2D(filters=32, kernel_size=7,
input_shape=[224, 224, 3]),
keras.layers.MaxPooling2D(pool_size=2),
DefaultConv2D(filters=64,kernel_size=3),
keras.layers.MaxPooling2D(pool_size=2),
keras.layers.Flatten(),
keras.layers.Dense(units=64, activation='relu'),
keras.layers.Dense(units=10, activation='softmax')])


# In[10]:


model.compile(loss="sparse_categorical_crossentropy",
optimizer=Adam(), metrics=["accuracy"])

history = model.fit(train_set,
validation_data=valid_set,
epochs=10)



# In[11]:


test = model.evaluate(test_set)[1]
test


# In[12]:


train = model.evaluate(train_set)[1]
train


# In[13]:


valid = model.evaluate(valid_set)[1]
valid


# In[14]:


import pickle


# In[15]:


with open('simple_cnn_acc.pkl','wb') as f:
    pickle.dump((train,valid,test),f)
model.save('simple_cnn_flowers.keras')


# In[16]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# In[17]:


base_model = tf.keras.applications.xception.Xception(
weights="imagenet",
include_top=False)


# In[19]:


tf.keras.utils.plot_model(base_model)


# In[20]:


avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)


# In[22]:


for layer in base_model.layers:
    layer.trainable = False


# In[23]:


model.compile(loss="sparse_categorical_crossentropy",
optimizer=Adam(), metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set,
epochs=5)


# In[24]:


for layer in base_model.layers:
    layer.trainable = True


# In[ ]:


model.compile(loss="sparse_categorical_crossentropy",
optimizer=Adam(), metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set,
epochs=3)


# In[ ]:


test = model.evaluate(test_set)[1]
test


# In[ ]:


train = model.evaluate(train_set)[1]
train


# In[ ]:


valid = model.evaluate(valid_set)[1]
valid


# In[ ]:


with open('xception_acc.pkl','wb') as f:
    pickle.dump((train,valid,test),f)
model.save('xception_flowers.keras')


# In[ ]:




