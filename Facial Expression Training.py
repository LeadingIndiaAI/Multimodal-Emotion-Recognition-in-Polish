#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
import os


# In[104]:


path = 'Final/'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))


# In[105]:


files[0]


# In[106]:


descriptors = pd.read_csv('Descriptors.csv', index_col=0)
descriptors.head()


# In[114]:


def load_file(file):
    temp_df = pd.read_csv(file, sep=',')
    max_value = np.amax(temp_df)
    max_value = max(max_value)
    temp_df = temp_df/max_value
    #print(max_value)
    name = file[6:-4]
    output_value = descriptors.loc[descriptors['video_name']==name +'.MP4']['output'][:1].values[0]
    #video_name = descriptors.loc[descriptors['text_file']==textfile]['video_name'][:1].values[0]
    gender = descriptors.loc[descriptors['video_name']==name +'.MP4']['gender'][:1].values[0]
    #temp_df['textfile'] = textfile
    #temp_df['output'] = output_value
    #temp_df['video_name'] = video_name
    if gender == 'F':
        temp_df['gender'] = 1
    else:
        temp_df['gender'] = 0
    #temp_df.drop(['Unnamed: 152'], axis=1, inplace=True)
    return temp_df.values, output_value


# In[115]:


a, b = load_file('Final/927_0196_01.csv')


# In[116]:


a


# In[117]:


shapes = []
Y = []
for name in files:
    data, t = load_file(name)
    shapes.append(data.shape[0])
    Y.append(t)


# In[19]:


index = 0 
for element in Y:
    if element == 'Ne':
        Y[index] = 0
    elif element == 'Sa':
        Y[index] = 1
    elif element == 'Su':
        Y[index] = 2
    elif element == 'Fe':
        Y[index] = 3
    elif element == 'An':
        Y[index] = 4
    elif element == 'Di':
        Y[index] = 5
    elif element == 'Ha':
        Y[index] = 6
    index += 1


# In[20]:


from tensorflow.keras.utils import to_categorical

maxlen = max(shapes)
print(maxlen)


# In[ ]:


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data, temp = load_file(name)
        #print('The name is:', name, data.shape)
        zeros_data = np.zeros((max(shapes)-data.shape[0], 153))
        data = np.concatenate((zeros_data, data), axis=0)
        #print('Now the shape is:', name, data.shape)
        loaded.append(data)
        shapes.append(data.shape[0])
    loaded = np.stack(loaded)
    return loaded


# In[ ]:


X = load_group(files)

Y = np.asarray(Y).reshape((63,1))
Y = to_categorical(Y)
print(X.shape, Y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# In[ ]:


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


# In[ ]:


evaluate_model(X_train, Y_train, X_test, Y_test)

