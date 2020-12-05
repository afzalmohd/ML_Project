#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


get_ipython().system('ln -s /content/gdrive/My\\ Drive/Acadamics/CS725/Project/data ./data')


# In[ ]:


get_ipython().system('pip install -U scikit-learn')


# In[ ]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pickle


# In[ ]:


dataframe = pd.read_csv('./data/train.csv', header=0)
data_train = dataframe.values


# In[ ]:


names = dataframe.columns[:-2]


# In[ ]:


X_train = data_train[:,:-2]
y_train = data_train[:,-1]


# In[ ]:


dataframe = pd.read_csv('./data/test.csv', header=0)
data_test = dataframe.values
X_test = data_test[:,:-2]
y_test = data_test[:,-1]


# In[ ]:


# PCA to detect redundant features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)

le = LabelEncoder()
le.fit(y_train)
y_train_le = le.transform(y_train)
# n_pcs= pca.components_.shape[0]

# # Get the index of the most important feature on each component in the PCA analysis
# most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
# most_important_names = [names[most_important[i]] for i in range(n_pcs)]

# redundant = set(names[:-1]) - set(most_important_names)
# n_red = len(redundant)

# print("Redundant features:" + str(redundant))


# In[ ]:


X_test_std = sc.transform(X_test)
X_test_pca = pca.transform(X_test_std)
y_test_le = le.transform(y_test)


# In[28]:


labels_list = list(le.classes_)
for i, label in enumerate(labels_list):
    if 'WALKING_' in label:
        labels_list[i] = label.replace('WALKING_','')
print(labels_list)


# In[ ]:


data_train_pp = np.concatenate((X_train_pca,y_train_le.reshape(-1,1)), axis=1)
data_test_pp = np.concatenate((X_test_pca,y_test_le.reshape(-1,1)), axis=1)


# In[ ]:


print(data_train_pp.shape,data_test_pp.shape)


# In[ ]:


pd.DataFrame(data_train_pp).to_csv("./data/train_pp.csv", header=None, index=None)
pd.DataFrame(data_test_pp).to_csv("./data/test_pp.csv", header=None, index=None)


# In[29]:


pickle.dump(sc, open('./data/scaler.pkl', 'wb'))
pickle.dump(pca, open('./data/pca.pkl', 'wb'))
pickle.dump(labels_list, open('./data/labels.pkl','wb'))

