#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[2]:


#!ln -s /content/gdrive/My\ Drive/Colab\ Notebooks/ML/data ./data
#get_ipython().system('ln -s /content/gdrive/My\\ Drive/Acadamics/CS725/Project/data ./data')


# In[29]:


import numpy as np
import pandas as pd
import time
import pickle
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten
from keras.utils import to_categorical 

dataframe = pd.read_csv('./data/train_pp.csv', header=None)
array = dataframe.values

X_train = array[:,:-1]
y_train = array[:,-1]
y_train = to_categorical(y_train, dtype='uint8')

dataframe = pd.read_csv('./data/test_pp.csv', header=None)
array = dataframe.values
X_test = array[:,:-1]#.reshape(array.shape[0],-1,1)
y_test = array[:,-1]
y_test = to_categorical(y_test, dtype='uint8')


# In[30]:


print(y_train.shape)


# In[31]:


n_features, n_outputs = X_train.shape[1], y_train.shape[1]

model=Sequential()
model.add(Dense(units=64,kernel_initializer='uniform',activation='relu',input_dim=n_features))
model.add(Dropout(0.5))
model.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=100,kernel_initializer='uniform',activation='relu'))

model.add(Dense(units=n_outputs,activation='softmax')) #kernel_initializer='uniform',


# In[32]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[33]:


t0= time.time()
history=model.fit(X_train,y_train,batch_size=256,epochs=10,validation_data=(X_test,y_test))
t1= time.time() - t0


# In[34]:


y_pred=model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)


# In[35]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt


print("Time duartion for training = {:.2f} seconds".format(t1))

print("Accuracy :",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred, average='weighted'))
print("Recall   :",recall_score(y_test, y_pred, average='weighted'))

cm=confusion_matrix(y_test, y_pred, normalize='true') 
labels = pickle.load(open('./data/labels.pkl','rb'))
plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True,cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.tick_params(axis='x', which='major', labelsize=12, rotation=45)
plt.tick_params(axis='y', which='major', labelsize=12, rotation=0)

class_accuracies = cm.diagonal()
np.save('./data/ffnn_class_accuracies',class_accuracies)
