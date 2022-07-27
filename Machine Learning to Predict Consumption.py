#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset_train = pd.read_excel('demand_jan_maio.xlsx')
dataset_train.head()


# In[3]:


training_set = dataset_train.iloc[:, 0:1].values


# In[4]:


dataset_train.shape


# In[5]:


dataset_train.head()


# In[6]:


training_set


# In[7]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set[:100000])


# In[8]:


3 * 480


# In[9]:


minutesAgoUpperBound = 1440
minutesAgoLowerBound = 1540

featureSize = minutesAgoLowerBound - minutesAgoUpperBound

minutespredict = 480
X_train = []
y_train = []
for i in range(minutesAgoLowerBound, 100000 - minutespredict):
    X_train.append(training_set_scaled[i - minutesAgoLowerBound:i-minutesAgoUpperBound, 0])
    y_train.append([training_set_scaled[i + x, 0] for x in range(minutespredict)])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# In[10]:


y_train.shape


# In[11]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


# In[12]:


model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=minutespredict))
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()


# In[13]:


model.fit(X_train,y_train,epochs=100, batch_size=1000)


# In[14]:


# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
# dataset_test = pd.read_csv(url)
# real_stock_price = dataset_test.iloc[:, 1:2].values
real_Consumption = training_set[100000:100000 + minutesAgoUpperBound]


# In[15]:


# minutespredict = 480
# minutesAgo = 100


# In[16]:


# dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# inputs = dataset_train[len(dataset_total) - len(dataset_test) - 60:].values
inputs = training_set[100000 - minutesAgoLowerBound:]
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(minutesAgoLowerBound, 2 * minutesAgoUpperBound):
    X_test.append(inputs[i-minutesAgoLowerBound:i-minutesAgoUpperBound, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[17]:


predicted_Consumption = model.predict(X_test)
# predicted_Consumption = sc.inverse_transform(predicted_Consumption)


# In[35]:


plt.plot(real_Consumption[:480], color = 'black', label = 'Elec Consumption')
plt.plot(sc.inverse_transform(predicted_Consumption)[0], color = 'green', label = 'Predicted Elec Consumption')
plt.title('Electricity Consumption Prediction')
plt.xlabel('Time')
plt.ylabel('Elec Consumption')
plt.legend()
plt.show()


# In[ ]:




