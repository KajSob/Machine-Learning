#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber


# In[3]:


tf.keras.utils.get_file(
"bike_sharing_dataset.zip",
"https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
cache_dir=".",
extract=True
)


# In[4]:


df = pd.read_csv('datasets/hour.csv',
parse_dates={'datetime': ['dteday', 'hr']},
date_format='%Y-%m-%d %H',
index_col='datetime')


# In[5]:


print((df.index.min(), df.index.max()))


# In[6]:


(365 + 366) * 24 - len(df)


# In[7]:


df


# In[8]:


zeros = pd.concat([df['casual'],df['registered'],df['cnt']],axis=1)
zeros =zeros.asfreq('h').fillna(0)
zeros


# In[9]:


inter = pd.concat([df['temp'],df['atemp'],df['hum'],df['windspeed']],axis=1)
inter = inter.resample('h').interpolate()
inter


# In[10]:


filler = pd.concat([df['holiday'],df['weekday'],df['workingday'],df['weathersit']],axis=1)
filler = filler.resample('h').ffill()
filler


# In[11]:


df = pd.concat([filler,inter,zeros],axis=1)


# In[12]:


df.notna().sum()


# In[13]:


df[['casual', 'registered', 'cnt', 'weathersit']].describe()


# In[14]:


df.casual /= 1e3
df.registered /= 1e3
df.cnt /= 1e3
df.weathersit /= 4


# In[15]:


df_2weeks = df[:24 * 7 * 2]
df_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[16]:


df_daily = df.resample('W').mean()
df_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[17]:


mae_daily = df['cnt'].diff(24).abs().mean() * 1e3
mae_weekly = df['cnt'].diff(24*7).abs().mean() * 1e3
mae_baseline = (mae_daily, mae_weekly)
print(mae_baseline)
with open('mae_baseline.pkl', 'wb') as f:
    pickle.dump(mae_baseline, f)


# In[18]:


cnt_train = df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df['cnt']['2012-07-01 00:00':]


# In[19]:


seq_len = 1 * 24
train_ds = tf.keras.utils.timeseries_dataset_from_array(
cnt_train.to_numpy(),
targets=cnt_train[seq_len:],
sequence_length=seq_len,
batch_size=32,
shuffle=True,
seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
cnt_valid.to_numpy(),
targets=cnt_valid[seq_len:],
sequence_length=seq_len,
batch_size=32
)


# In[20]:


model = tf.keras.Sequential([
tf.keras.layers.Dense(1, input_shape=[seq_len])
])


# In[21]:


model.compile(loss=Huber(),
optimizer=SGD(learning_rate=0.0001,momentum=0.9), metrics=["mae"])

history = model.fit(train_ds,
validation_data=valid_ds,
epochs=20)


# In[26]:


mae_linear = (model.evaluate(valid_ds)[1] * 1e3,)
mae_linear


# In[27]:


with open('mae_linear.pkl','wb') as f:
    pickle.dump(mae_linear,f)
model.save('model_linear.keras')


# In[28]:


model = tf.keras.Sequential([
tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])])


# In[29]:


model.compile(loss=Huber(),
optimizer=SGD(learning_rate=0.0001,momentum=0.9), metrics=["mae"])

history = model.fit(train_ds,
validation_data=valid_ds,
epochs=20)


# In[30]:


mae_rnn1 = (model.evaluate(valid_ds)[1] * 1e3,)
mae_rnn1


# In[31]:


with open('mae_rnn1.pkl','wb') as f:
    pickle.dump(mae_rnn1,f)
model.save('model_rnn1.keras')


# In[32]:


model = tf.keras.Sequential([
tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
tf.keras.layers.Dense(1)])


# In[33]:


model.compile(loss=Huber(),
optimizer=SGD(learning_rate=0.0001,momentum=0.9), metrics=["mae"])

history = model.fit(train_ds,
validation_data=valid_ds,
epochs=20)


# In[34]:


mae_rnn32 = (model.evaluate(valid_ds)[1] * 1e3,)
mae_rnn32


# In[35]:


with open('mae_rnn32.pkl','wb') as f:
    pickle.dump(mae_rnn32,f)
model.save('model_rnn32.keras')


# In[36]:


model = tf.keras.Sequential([
tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
tf.keras.layers.SimpleRNN(32, return_sequences=True),
tf.keras.layers.SimpleRNN(32, return_sequences=True),
tf.keras.layers.Dense(1)])


# In[37]:


model.compile(loss=Huber(),
optimizer=SGD(learning_rate=0.0001,momentum=0.9), metrics=["mae"])

history = model.fit(train_ds,
validation_data=valid_ds,
epochs=20)


# In[38]:


mae_rnn_deep = (model.evaluate(valid_ds)[1] * 1e3,)
mae_rnn_deep


# In[39]:


with open('mae_rnn_deep.pkl','wb') as f:
    pickle.dump(mae_rnn_deep,f)
model.save('model_rnn_deep.keras')


# In[40]:


last = pd.concat([df['weathersit'],df['atemp'],df['workingday'],df['cnt']],axis=1)
last


# In[41]:


cnt_train = last['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = last['2012-07-01 00:00':]


# In[42]:


seq_len = 1 * 24
train_ds = tf.keras.utils.timeseries_dataset_from_array(
cnt_train.to_numpy(),
targets=cnt_train[seq_len:],
sequence_length=seq_len,
batch_size=32,
shuffle=True,
seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
cnt_valid.to_numpy(),
targets=cnt_valid[seq_len:],
sequence_length=seq_len,
batch_size=32
)


# In[43]:


model = tf.keras.Sequential([
tf.keras.layers.SimpleRNN(32, input_shape=[24, 4]),
tf.keras.layers.Dense(4)])


# In[44]:


model.compile(loss=Huber(),
optimizer=SGD(learning_rate=0.0001,momentum=0.9), metrics=["mae"])

history = model.fit(train_ds,
validation_data=valid_ds,
epochs=20)


# In[46]:


mae_rnn_mv = (model.evaluate(valid_ds)[1] * 1e3,)
mae_rnn_mv


# In[47]:


with open('mae_rnn_mv.pkl','wb') as f:
    pickle.dump(mae_rnn_mv,f)
model.save('model_rnn_mv.keras')


# In[ ]:




