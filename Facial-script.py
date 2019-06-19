#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split


# In[3]:


X = np.load('X.npy')
Y = np.load('Y.npy')
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)


# In[4]:


def evaluate_model(trainX, trainy, testX, testy):
    verbose = 1
    epochs = 100
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(32, input_shape=(n_timesteps, n_features)))
    #model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(trainX, trainy, epochs=epochs, verbose=verbose)
    _, accuracy = model.evaluate(testX, testy, verbose=verbose)
    return accuracy


# In[5]:


evaluate_model(X_train, Y_train, X_test, Y_test)


# In[ ]:




