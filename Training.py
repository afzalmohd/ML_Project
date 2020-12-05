#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[3]:


get_ipython().system('ln -s /content/gdrive/My\\ Drive/Acadamics/CS725/Project/data ./data')


# In[1]:


import numpy as np
import pickle


# In[20]:


dataframe = pd.read_csv('./data/train_pp.csv', header=None)
array = dataframe.values
print(array.shape)
X_train = array[:,:-2]
Y_train = array[:,-1].reshape(-1,1)


# In[21]:


dataframe = pd.read_csv('./data/test_pp.csv', header=None)
array = dataframe.values
X_test = array[:,:-2]
Y_test = array[:,-1].reshape(-1,1)


# In[22]:


print(X_train.shape, X_test.shape)

