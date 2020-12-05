#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


#!ln -s /content/gdrive/My\ Drive/Acadamics/CS725/Project/data ./data
get_ipython().system('ln -s /content/gdrive/My\\ Drive/Colab\\ Notebooks/ML/data ./data')


# In[ ]:


import numpy as np
import pickle
import pandas as pd
import time


# In[ ]:


dataframe = pd.read_csv('./data/train_pp.csv', header=None)
array = dataframe.values
print(array.shape)
X_train = array[:,:-1]
y_train = array[:,-1]


# In[ ]:


dataframe = pd.read_csv('./data/test_pp.csv', header=None)
array = dataframe.values
X_test = array[:,:-1]
y_test = array[:,-1]
print(X_train.shape, X_test.shape)


# In[ ]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')

t0= time.time()
svclassifier.fit(X_train, y_train)
t1= time.time() - t0

y_pred = svclassifier.predict(X_test)


# In[ ]:


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
plt.tick_params(axis='x', which='major', labelsize=12)
plt.tick_params(axis='y', which='major', labelsize=12, rotation=0)

class_accuracies = cm.diagonal()
np.save('./data/svm_class_accuracies',class_accuracies)

