#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from google.colab import drive
#drive.mount('/content/gdrive')
#get_ipython().system('ln -s /content/gdrive/My\\ Drive/Acadamics/CS725/Project/data ./data')


# In[2]:


# %% [code]
import numpy as np
import pandas as pd
import time
import pickle
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten
from keras.utils import to_categorical 

dataframe = pd.read_csv('./data/train_pp.csv', header=None)
array = dataframe.values

X_train = array[:,:-1].reshape(array.shape[0],-1,1)
y_train = array[:,-1]
y_train = to_categorical(y_train, dtype='uint8')

dataframe = pd.read_csv('./data/test_pp.csv', header=None)
array = dataframe.values
X_test = array[:,:-1].reshape(array.shape[0],-1,1)
y_test = array[:,-1]


# In[3]:


print(X_train.shape,y_train.shape)


# In[16]:


y_test_ = to_categorical(y_test, dtype='uint8')

verbose, epochs, batch_size = 2, 10, 256
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

t0= time.time()
history=model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
t1= time.time() - t0


# In[13]:


y_pred=model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)


# In[14]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

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
# np.save('./data/cnn_class_accuracies',class_accuracies)


# In[7]:


# %% [code]
from pylab import rcParams
rcParams['figure.figsize'] = 10, 4
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% [code]
y_pred=model.predict(X_test)

# %% [code]
y_test_class=np.argmax(y_test,axis=1)
y_pred_class=np.argmax(y_pred,axis=1)

# %% [code]
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test_class,y_pred_class)
accuracy=accuracy_score(y_test_class,y_pred_class)

