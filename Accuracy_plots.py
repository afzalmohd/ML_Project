#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[2]:


get_ipython().system('ln -s /content/gdrive/My\\ Drive/Acadamics/CS725/Project/data ./data')


# In[6]:


import numpy as np
import pickle
import matplotlib.pyplot as plt


# In[4]:


svm = np.load('./data/svm_class_accuracies.npy')
rf = np.load('./data/rf_class_accuracies.npy')
ffnn = np.load('./data/ffnn_class_accuracies.npy')
cnn = np.load('./data/cnn_class_accuracies.npy')


# In[21]:


fig = plt.figure(figsize=(6,4))
X = np.arange(len(svm))
labels = pickle.load(open('./data/labels.pkl','rb'))
labels.insert(0,'')
ax = fig.add_axes([0,0,1,1])
ax.bar(X - 0.30, rf, color = 'deepskyblue', width = 0.20, label='RF')
ax.bar(X - 0.10, svm, color = 'crimson', width = 0.20, label='SVM')
ax.bar(X + 0.10, ffnn, color = 'yellowgreen', width = 0.20, label='FFNN')
ax.bar(X + 0.30, cnn, color = 'coral', width = 0.20, label='CNN')
ax.set_xticklabels(labels=labels)
ax.set_ylabel('Accuarcy')
ax.set_ylim(0,1)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)


# In[23]:


fig = plt.figure(figsize=(6,4))
X = np.arange(4)
labels = ['RF','SVM','FFNN','CNN']
acc = [.868,.9518,.9596,.9606]
labels.insert(0,'')
ax = fig.add_axes([0,0,1,1])
ax.bar(X, acc, color = 'deepskyblue')
ax.set_xticklabels(labels=labels)
ax.set_ylabel('Accuarcy')
ax.set_ylim(0,1)
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

